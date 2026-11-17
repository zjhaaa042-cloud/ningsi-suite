"""Frozen classifiers retain their feature contract and reject blink windows live."""

import unittest

import numpy as np

from ningsi.signal.window import analyze_window
from ningsi_studio.core.human_inference import veto_live_blinks


class HumanBlinkVetoTests(unittest.TestCase):
    rate = 250

    def setUp(self):
        self.t = np.arange(4 * self.rate) / self.rate
        self.clean = np.stack([12 * np.sin(2 * np.pi * 10 * self.t),
                               10 * np.sin(2 * np.pi * 10 * self.t + .2)])
        window = analyze_window(self.clean, self.rate, t_end=4.)
        self.assertTrue(window.usable)
        self.row = {"window": window, "features": {"sentinel": 1.25}, "start": 0., "end": 4.}
        self.profile = {"srate": self.rate, "scale_to_uv": 1., "channel_labels": ["Fp1", "Fp2"]}

    def apply(self, values, *, total=None, end=None, profile=None):
        total = values.shape[1] if total is None else total
        return veto_live_blinks(self.row, {"samples": values, "total_samples": total},
                                profile or self.profile, total if end is None else end)

    def test_clean_feature_values_and_source_are_untouched(self):
        original = self.clean.copy()
        self.assertIs(self.apply(self.clean), self.row)
        np.testing.assert_array_equal(self.clean, original)

    def test_raw_blink_veto_cannot_be_overridden_by_calibrated_focus_path(self):
        blink = 160 * np.exp(-((self.t - 2) / .12) ** 2)
        raw = self.clean + np.array([1., .8])[:, None] * blink
        raw += np.array([220000., -170000.])[:, None]
        original = raw.copy()
        result = self.apply(raw)
        self.assertFalse(result["window"].usable)
        self.assertFalse(result["features"])
        self.assertTrue(result["window"].quality.metrics["live_blink_veto"])
        self.assertTrue(self.row["window"].usable)
        self.assertEqual(self.row["features"], {"sentinel": 1.25})
        np.testing.assert_array_equal(raw, original)

    def test_no_lookahead_outside_model_window(self):
        later = self.clean.copy()
        later[:, self.rate:2*self.rate] += 300.
        raw = np.concatenate([self.clean, later], axis=1)
        self.assertIs(self.apply(raw, end=4*self.rate), self.row)

    def test_model_window_offsets_and_unit_scale_are_respected(self):
        blink = 160 * np.exp(-((self.t - 2) / .12) ** 2)
        raw = self.clean + np.array([1., .8])[:, None] * blink
        profile = {**self.profile, "scale_to_uv": 1e6}
        result = self.apply(raw / 1e6, total=40*self.rate, profile=profile)
        self.assertFalse(result["window"].usable)

    def test_existing_qc_rejections_stay_rejected(self):
        rejected = {**self.row, "window": analyze_window(np.zeros_like(self.clean), self.rate), "features": {}}
        result = veto_live_blinks(rejected, {"samples": self.clean, "total_samples": len(self.t)},
                                  self.profile, len(self.t))
        self.assertIs(result, rejected)


if __name__ == "__main__":
    unittest.main()
