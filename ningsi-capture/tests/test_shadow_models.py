from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ningsi" / "src"))
sys.path.insert(0, str(ROOT / "ningsi-capture" / "scripts"))
from ningsi.models import attlapse_shadow as shadow
from attlapse_two_channel_eval import predict as offline_predict
from attlapse_two_channel_features import extract_features


class ShadowModelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = shadow.load_candidates()

    def test_both_frozen_scores_match_offline_predictor(self):
        probes = np.random.default_rng(19).normal(size=(22, 6))
        out = [shadow.score_features(self.model, dict(zip(shadow.FEATURE_NAMES, row))) for row in probes]
        for family in shadow.FAMILIES:
            expected = offline_predict(self.model["parameters"][family], probes)
            np.testing.assert_allclose([r[family + "_score"] for r in out], expected, atol=1e-14)
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "weights.json"
            bad.write_text("{}", encoding="utf8")
            with self.assertRaisesRegex(ValueError, "checksum"):
                shadow.load_candidates(bad)

    def test_probe_feature_and_quality_parity_with_offline(self):
        rate = 250
        stamps = np.arange(46 * rate) / rate
        alpha = 6 * np.sin(2 * np.pi * 10 * stamps)
        theta = 3 * np.cos(2 * np.pi * 7 * stamps)
        beta = 3 * np.cos(2 * np.pi * 17 * stamps)
        artifact = 34 * np.sin(2 * np.pi * 1.1 * stamps)
        values = np.stack([artifact + alpha + theta, artifact + beta])
        probe = dict(participant="p", probe_id="x", label=1, source_score=1, source_answer="x",
                     source="sourcedata", rate=rate, unit="µV", channels=["Fp1","Fp2"],
                     timestamps=stamps, values_uv=values, start_sample=0, onset_sample=46*rate,
                     onset_sec=46., phase_start_sec=0., bad_channels=[])
        expected = {r["quality"]: r for r in extract_features(probe) if r["window"] == "pre18_2"}
        # Pass reversed channels in millivolts, including windows beyond the analysis interval.
        rows = [shadow.compute_window(values[::-1, end-1500:end] / 1000, stamps[end-1500:end],
                                      rate, 1000., (1, 0)) for end in range(6500, len(stamps)+1, 500)]
        result = shadow.aggregate_and_score(self.model, rows, 46., 0.)
        self.assertEqual(result["windows_total"], expected["physical"]["windows_total"])
        self.assertEqual(result["windows_usable"], expected["physical"]["windows_usable"])
        self.assertEqual(result["strict_windows_usable"], expected["strict"]["windows_usable"])
        self.assertEqual(result["strict_window_rejections"], expected["strict"]["window_rejections"])
        self.assertEqual(result["strict_qc_status"], "failed")
        self.assertEqual(result["physical_qc_status"], "passed")
        np.testing.assert_allclose([result["features"][k] for k in shadow.FEATURE_NAMES],
                                   [expected["physical"]["features"][k] for k in shadow.FEATURE_NAMES], atol=1e-10)
        self.assertIsNotNone(result["rbf_score"])

    def test_timing_gap_bad_units_flatline_and_missing_windows_abstain(self):
        rate = 250
        x = np.zeros((2, 1500))
        stamps = np.arange(1500) / rate + 4
        gap = stamps.copy()
        gap[-40:] += .15
        for times, factor, reason in ((gap, 1, "timestamp_gap_over_100ms"),
                                      (stamps, None, "unit_unconfirmed"), (stamps, 1, "input_flat_channel")):
            row = shadow.compute_window(x, times, rate, factor, (0, 1))
            self.assertIn(reason, row["physical_reasons"])
            res = shadow.aggregate_and_score(self.model, [row], 20, 0, expected_windows=7)
            self.assertIsNone(res["logistic_score"])
            self.assertIsNone(res["rbf_score"])
            self.assertEqual(res["physical_window_rejections"]["missing_window"], 6)
            self.assertEqual(res["physical_window_rejections"][reason], 1)
        with self.assertRaisesRegex(ValueError, "250 Hz"):
            shadow.compute_window(x, stamps, 512, 1., (0,1))
        self.assertIn("lookback_crosses_phase", shadow.aggregate_and_score(self.model, [], 20, 10)["reason"])


if __name__ == "__main__":
    unittest.main()
