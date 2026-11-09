"""频谱、频带、伪迹、基线与指标的口径测试。"""

from __future__ import annotations

import unittest

import numpy as np

from tests.helpers import make_baseline
from ningsi.signal.artifacts import check_channels
from ningsi.signal.bands import band_power, relative_band_powers
from ningsi.signal.baseline import can_compare
from ningsi.signal.indicators import compute_indicators, score_from_z
from ningsi.signal.spectrum import welch_psd
from ningsi.signal.window import analyze_window
from ningsi.acquisition.simulate import SyntheticEEG


class SpectrumTest(unittest.TestCase):
    def setUp(self):
        self.srate = 250.0
        self.t = np.arange(int(4 * self.srate)) / self.srate

    def test_peak_lands_in_alpha_band(self):
        signal = 20.0 * np.sin(2 * np.pi * 10 * self.t)
        spectrum = welch_psd(signal, self.srate)
        self.assertEqual(spectrum.segments, 3)
        powers = band_power(spectrum.freqs, spectrum.psd, *__import__("ningsi.config", fromlist=["BANDS"]).BANDS["alpha"])
        self.assertAlmostEqual(powers, 200.0, delta=20.0)

    def test_short_window_is_rejected(self):
        with self.assertRaises(ValueError):
            welch_psd(np.zeros(100), self.srate)

    def test_relative_powers_sum_to_one(self):
        signal = 15.0 * np.sin(2 * np.pi * 6 * self.t) + 10.0 * np.sin(2 * np.pi * 20 * self.t)
        spectrum = welch_psd(signal, self.srate)
        rel = relative_band_powers(spectrum.freqs, spectrum.psd)
        self.assertAlmostEqual(sum(rel.values()), 1.0, places=4)


class QualityTest(unittest.TestCase):
    def setUp(self):
        self.sim = SyntheticEEG(seed=5)

    def test_clean_window_passes(self):
        self.assertTrue(check_channels(self.sim.window("rest"), self.sim.srate).ok)

    def test_artifact_reasons(self):
        verdict = check_channels(self.sim.window("rest", artifact=True), self.sim.srate)
        self.assertFalse(verdict.ok)
        self.assertIn("amplitude", verdict.reasons)

    def test_flat_channel_detected(self):
        verdict = check_channels(np.zeros((1, 1000)), self.sim.srate)
        self.assertIn("flat_channel", verdict.reasons)


class IndicatorTest(unittest.TestCase):
    def setUp(self):
        self.sim, self.baseline = make_baseline()

    def _mean_score(self, state: str, count: int = 6) -> dict:
        results = [compute_indicators(analyze_window(self.sim.window(state), self.sim.srate), self.baseline) for _ in range(count)]
        return {key: sum(r.scores[key] for r in results) / len(results) for key in results[0].scores}

    def test_focused_state_raises_focus_lowers_load(self):
        scores = self._mean_score("focused")
        self.assertGreater(scores["focus"], 0.8)
        self.assertLess(scores["load"], 0.3)

    def test_drowsy_state_raises_relax(self):
        scores = self._mean_score("drowsy")
        self.assertGreater(scores["relax"], 0.8)
        self.assertLess(scores["focus"], 0.3)

    def test_loaded_state_raises_load(self):
        self.assertGreater(self._mean_score("loaded")["load"], 0.8)

    def test_unusable_window_returns_unavailable(self):
        window = analyze_window(self.sim.window("rest", artifact=True), self.sim.srate)
        result = compute_indicators(window, self.baseline)
        self.assertFalse(result.available)
        self.assertEqual(result.reason, "window_unusable")

    def test_invalid_baseline_is_rejected(self):
        from ningsi.signal.baseline import Baseline
        invalid = Baseline("baseline-v1", 250.0, 1, "sim", {}, {}, 0, False, ("no_usable_window",))
        result = compute_indicators(analyze_window(self.sim.window("rest"), self.sim.srate), invalid)
        self.assertFalse(result.available)
        self.assertEqual(result.reason, "baseline_invalid")

    def test_score_is_bounded(self):
        self.assertEqual(score_from_z(50.0), 1.0)
        self.assertLess(score_from_z(-50.0), 1e-6)
        self.assertAlmostEqual(score_from_z(0.0), 0.5, places=6)

    def test_comparability_guards(self):
        self.assertEqual(can_compare(self.baseline, 250.0, 1, "sim"), (True, "comparable"))
        self.assertEqual(can_compare(self.baseline, 128.0, 1, "sim")[1], "sample_rate_changed")
        self.assertEqual(can_compare(self.baseline, 250.0, 2, "sim")[1], "channel_count_changed")
        self.assertEqual(can_compare(self.baseline, 250.0, 1, "other")[1], "device_changed")


if __name__ == "__main__":
    unittest.main()
