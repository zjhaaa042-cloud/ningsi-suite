"""信号处理链与可训练模型的口径测试。"""

from __future__ import annotations

import unittest

import numpy as np

from tests.helpers import temp_root
from ningsi.acquisition.simulate import SyntheticEEG
from ningsi.models.logistic import (
    FEATURE_NAMES,
    LinearModel,
    evaluate,
    feature_matrix,
    features_from_window,
    fit_logistic,
    roc_auc,
    subject_split,
)
from ningsi.signal.bands import band_power
from ningsi.signal.preprocess import preprocess
from ningsi.signal.spectrum import welch_psd
from ningsi.signal.window import analyze_window

SRATE = 250.0
T = np.arange(int(4 * SRATE)) / SRATE


def mains_power(signal) -> float:
    """工频功率按 49–51 Hz 积分；fmax 需放宽到 60 Hz 才能看到陷波前后的差异。"""
    spectrum = welch_psd(signal, SRATE, fmax=60.0)
    return band_power(spectrum.freqs, spectrum.psd, 49.0, 51.0)


def slow_power(signal, low: float = 0.5, high: float = 2.0) -> float:
    spectrum = welch_psd(signal, SRATE)
    return band_power(spectrum.freqs, spectrum.psd, low, high)


class PreprocessTest(unittest.TestCase):
    def test_notch_removes_mains(self):
        signal = 20 * np.sin(2 * np.pi * 10 * T) + 40 * np.sin(2 * np.pi * 50 * T)
        filtered, log = preprocess(signal, SRATE)
        self.assertLess(mains_power(filtered), 0.05 * mains_power(signal))
        self.assertEqual(len([s for s in log.as_dict()["stages"] if s["stage"] == "mains_notch"]), 2)

    def test_drift_and_dc_removed(self):
        signal = 30 * np.sin(2 * np.pi * 0.2 * T) + 15 * np.sin(2 * np.pi * 10 * T) + 50.0
        filtered, _ = preprocess(signal, SRATE)
        self.assertLess(abs(float(np.mean(filtered))), 1.0)
        self.assertLess(slow_power(filtered), 0.10 * slow_power(signal))

    def test_alpha_band_preserved(self):
        signal = 20 * np.sin(2 * np.pi * 10 * T)
        filtered, _ = preprocess(signal, SRATE)
        raw = welch_psd(signal, SRATE)
        out = welch_psd(filtered, SRATE)
        self.assertAlmostEqual(band_power(out.freqs, out.psd, 8, 13) / band_power(raw.freqs, raw.psd, 8, 13), 1.0, delta=0.2)

    def test_forward_backward_filter_has_no_interior_phase_shift(self):
        from ningsi.signal.preprocess import _biquad, filtfilt
        time = np.arange(int(40 * SRATE)) / SRATE
        signal = np.sin(2 * np.pi * time)
        filtered = filtfilt(signal, *_biquad("highpass", 0.5, SRATE), pad=750)
        region = slice(int(10 * SRATE), int(30 * SRATE))
        basis = np.column_stack((np.sin(2 * np.pi * time[region]), np.cos(2 * np.pi * time[region])))
        coefficients = np.linalg.lstsq(basis, filtered[region], rcond=None)[0]
        self.assertLess(abs(np.degrees(np.arctan2(coefficients[1], coefficients[0]))), 0.1)

    def test_chain_log_records_parameters(self):
        _, log = preprocess(np.zeros(int(2 * SRATE)), SRATE)
        stages = log.as_dict()["stages"]
        self.assertEqual(stages[0]["stage"], "drift_correction")
        self.assertIn("fc_hz", stages[0])
        self.assertEqual(stages[-1]["stage"], "band_pass")
        self.assertEqual(stages[-1]["band_hz"], (0.5, 45.0))
        self.assertTrue(all(stage.get("zero_phase") for stage in stages))


class ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matrix = []
        cls.labels = []
        cls.participants = []
        for subject in range(6):
            sim = SyntheticEEG(seed=100 + subject)
            for state, label in (("focused", 1), ("drowsy", 0)):
                for _ in range(10):
                    window = analyze_window(sim.window(state), sim.srate)
                    row = features_from_window(window)
                    cls.matrix.append([row[name] for name in FEATURE_NAMES])
                    cls.labels.append(label)
                    cls.participants.append(f"p{subject:02d}")
        cls.matrix = np.array(cls.matrix)
        cls.labels = np.array(cls.labels)
        cls.model = fit_logistic(cls.matrix, cls.labels, trained_subjects=("p00", "p01"))

    def test_features_are_finite(self):
        self.assertTrue(np.all(np.isfinite(self.matrix)))

    def test_training_separates_states(self):
        metrics = evaluate(self.model, self.matrix, self.labels)
        self.assertGreater(metrics["accuracy"], 0.9)
        self.assertGreater(metrics["auc"], 0.9)
        self.assertEqual(sum(metrics["confusion"].values()), len(self.labels))

    def test_auc_is_rank_based(self):
        self.assertAlmostEqual(roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4]), 1.0, places=6)
        self.assertAlmostEqual(roc_auc([0, 1], [0.5, 0.5]), 0.5, places=6)

    def test_subject_split_is_disjoint(self):
        split = subject_split([f"p{index:02d}" for index in range(6)], seed=3)
        seen = split["train"] + split["validation"] + split["test"]
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(set(seen), {f"p{index:02d}" for index in range(6)})

    def test_holdout_subject_and_persistence(self):
        split = subject_split(self.participants, seed=3)
        train = [index for index, name in enumerate(self.participants) if name in split["train"]]
        test = [index for index, name in enumerate(self.participants) if name in split["test"]]
        model = fit_logistic(self.matrix[train], self.labels[train], trained_subjects=tuple(split["train"]))
        self.assertGreater(evaluate(model, self.matrix[test], self.labels[test])["accuracy"], 0.8)
        path = model.save(temp_root("model") / "model.json")
        restored = LinearModel.load(path)
        self.assertTrue(np.allclose(restored.weights, model.weights))
        self.assertEqual(restored.trained_subjects, tuple(split["train"]))

    def test_feature_matrix_shape(self):
        sim = SyntheticEEG(seed=1)
        windows = [analyze_window(sim.window("rest"), sim.srate) for _ in range(3)]
        matrix, names = feature_matrix(windows)
        self.assertEqual(matrix.shape, (3, len(FEATURE_NAMES)))
        self.assertEqual(tuple(names), FEATURE_NAMES)


if __name__ == "__main__":
    unittest.main()
