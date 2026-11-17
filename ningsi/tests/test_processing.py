"""Shared window service preserves scientific calculations and quality gates."""

import unittest

import numpy as np

from ningsi.acquisition.simulate import SyntheticEEG
from ningsi.processing import process_window
from ningsi.signal.baseline import build_baseline
from ningsi.signal.indicators import compute_indicators
from ningsi.signal.window import analyze_window


class ProcessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = SyntheticEEG(seed=42)
        windows = [analyze_window(source.window(), 250, t_end=4 + 2 * i) for i in range(20)]
        cls.baseline = build_baseline(windows, device="test", srate=250)
        assert cls.baseline.valid

    def test_clean_result_keeps_existing_numerical_definition(self):
        raw = SyntheticEEG(seed=9).window("focused")
        original = raw.copy()
        expected = analyze_window(raw, 250, t_end=12)
        result = process_window(raw, 250, t_end=12, baseline=self.baseline)
        self.assertEqual(result.window.as_dict(), expected.as_dict())
        self.assertEqual(result.indicators.as_dict(), compute_indicators(expected, self.baseline).as_dict())
        np.testing.assert_array_equal(raw, original)

    def test_missing_short_and_nonfinite_windows_remain_unavailable(self):
        for raw, reason, samples in ((None, "no_data", 0),
                                     (np.zeros((1, 0)), "no_data", 0),
                                     (np.zeros((1, 500)), "signal_too_short", 500),
                                     (np.full((1, 1000), np.nan), "non_finite_signal", 1000)):
            with self.subTest(reason=reason, samples=samples):
                result = process_window(raw, 250, baseline=self.baseline)
                self.assertFalse(result.window.usable)
                self.assertEqual(result.window.quality.reasons, (reason,))
                self.assertEqual(result.window.quality.metrics["samples"], samples)
                self.assertFalse(result.indicators.available)
                self.assertEqual(result.indicators.reason, "window_unusable")
                self.assertIsNone(result.indicators.score("focus"))

    def test_clean_signal_needs_a_matching_valid_baseline(self):
        result = process_window(SyntheticEEG(seed=9).window(), 250)
        self.assertTrue(result.window.usable)
        self.assertEqual(result.indicators.reason, "baseline_invalid")
        mismatch = process_window(SyntheticEEG(channels=2, seed=9).window(), 250,
                                  baseline=self.baseline)
        self.assertTrue(mismatch.window.usable)
        self.assertEqual(mismatch.indicators.reason, "baseline_configuration_mismatch")
        self.assertEqual(mismatch.indicators.scores, {})

    def test_acquisition_artifact_evidence_vetoes_an_otherwise_clean_window(self):
        evidence = {"spec": "test-reference", "reasons": ["motion_reference_excursion"],
                    "motion_reference": {"unit": "g"}}
        result = process_window(SyntheticEEG(seed=9).window(), 250,
                                baseline=self.baseline, artifact_evidence=evidence)
        self.assertFalse(result.window.usable)
        self.assertEqual(result.window.quality.reasons, ("motion_reference_excursion",))
        self.assertEqual(result.window.quality.metrics["artifact_control"], evidence)
        self.assertEqual(result.indicators.scores, {})
        self.assertEqual(result.indicators.reason, "window_unusable")

    def test_reference_correction_evidence_without_rejection_preserves_scores(self):
        raw = SyntheticEEG(seed=9).window()
        plain = process_window(raw, 250, baseline=self.baseline)
        corrected = process_window(raw, 250, baseline=self.baseline,
                                   artifact_evidence={"reasons": [], "ocular_corrected_channels": [0]})
        self.assertTrue(corrected.window.usable)
        self.assertEqual(corrected.indicators.as_dict(), plain.indicators.as_dict())


if __name__ == "__main__":
    unittest.main()
