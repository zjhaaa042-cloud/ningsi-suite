import unittest

import numpy as np

from ningsi.models import attlapse_shadow, focus_tcn_recovery
from ningsi.models.logistic import features_from_window
from ningsi.models.research_features import condition_window


class FocusTcnRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.rate = 250
        self.timestamps = np.arange(1500) / self.rate
        t = self.timestamps
        difference = 12 * np.sin(2 * np.pi * 10 * t) + 4 * np.sin(2 * np.pi * 23 * t)
        common = 600 * np.sin(2 * np.pi * 50 * t) + 220 * np.sin(2 * np.pi * 2 * t)
        self.clean = np.stack([.5 * difference, -.5 * difference])
        self.contaminated = self.clean + common

    def analyze(self, values, times=None):
        return focus_tcn_recovery.compute_window(
            values, self.timestamps if times is None else times, self.rate, 1., (0, 1))

    def test_clean_window_matches_frozen_features(self):
        stock = attlapse_shadow.compute_window(self.clean, self.timestamps, self.rate, 1., (0, 1))
        actual = self.analyze(self.clean)
        self.assertFalse(actual["recovered"])
        self.assertEqual(actual["physical_reasons"], [])
        self.assertEqual(actual["features"], stock["features"])

    def test_shared_excess_keeps_original_bipolar_features_and_input_flags(self):
        row = self.analyze(self.contaminated)
        difference = self.contaminated[0:1] - self.contaminated[1:2]
        expected = condition_window(difference, self.timestamps, self.rate, 1.)["window"]
        self.assertTrue(row["recovered"])
        self.assertEqual(row["original_physical_reasons"], ["input_amplitude"])
        self.assertIn("input_amplitude", row["strict_reasons"])
        self.assertEqual(row["physical_reasons"], [])
        self.assertEqual(row["features"], features_from_window(expected))

    def test_held_contact_and_large_unilateral_transient_still_rejected(self):
        plateau = self.contaminated.copy()
        plateau[:, 600:650] = plateau[:, 600:601]
        spike = self.contaminated.copy()
        spike[0, 600:610] += 300
        for values in (plateau, spike):
            row = self.analyze(values)
            self.assertFalse(row["recovered"])
            self.assertTrue(row["physical_reasons"])

    def test_common_low_frequency_without_mains_and_bipolar_eye_proxy_still_rejected(self):
        t = self.timestamps
        no_mains = self.clean + 220 * np.sin(2 * np.pi * 2 * t)
        differential = self.contaminated.copy()
        differential[0] += 20 * np.sin(2 * np.pi * 3 * t)
        self.assertFalse(self.analyze(no_mains)["recovered"])
        row = self.analyze(differential)
        self.assertFalse(row["recovered"])
        self.assertIn("bipolar_eog", row["strict_reasons"])

    def test_invalid_clock_never_uses_signal_recovery(self):
        clock = self.timestamps.copy()
        clock[500] = np.nan
        row = self.analyze(self.contaminated, clock)
        self.assertFalse(row["recovered"])
        self.assertIn("clock_unsynchronized", row["physical_reasons"])
