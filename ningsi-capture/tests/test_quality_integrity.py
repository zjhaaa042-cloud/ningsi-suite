import unittest

import numpy as np

from ningsi_capture.quality_integrity import audit_integrity


class QualityIntegrityTests(unittest.TestCase):
    @staticmethod
    def signal(length, rate=20):
        sample = np.arange(length, dtype=float)
        return np.stack((sample, -sample)), 100. + sample / rate

    def test_large_finite_excursions_remain_unclassified(self):
        values, stamps = self.signal(40)
        values[0, 12] += 10_000.

        audit = audit_integrity(values, stamps, 20)

        self.assertEqual(audit.spans, [])
        self.assertTrue(audit.window_intact(0, 40))
        self.assertFalse(audit.window_intact(-1, 5))
        self.assertFalse(audit.window_intact(10, 10))
        self.assertFalse(audit.window_intact(0, 41))

    def test_bad_raw_values_and_missing_time_report_only_affected_samples(self):
        values, stamps = self.signal(12)
        values[1, 2:4] = [np.inf, np.nan]
        stamps[6:8] = np.nan

        audit = audit_integrity(values, stamps, 20)

        self.assertCountEqual(audit.spans, [
            {"channel": 2, "start_sample_index": 2, "stop_sample_index": 4,
             "start_lsl": stamps[2], "end_lsl": stamps[4], "reason": "raw_nonfinite"},
            {"channel": None, "start_sample_index": 6, "stop_sample_index": 8,
             "start_lsl": None, "end_lsl": stamps[8], "reason": "clock_timestamp_missing"},
        ])
        for first, stop in ((0, 2), (4, 6), (8, 12)):
            self.assertTrue(audit.window_intact(first, stop))
        self.assertFalse(audit.window_intact(1, 3))
        self.assertFalse(audit.window_intact(7, 10))

    def test_large_clock_gap_fails_only_windows_crossing_actual_edge(self):
        values, stamps = self.signal(10, 10)
        stamps[4:] += .2

        audit = audit_integrity(values, stamps, 10)

        self.assertEqual(audit.spans, [
            {"channel": None, "start_sample_index": 4, "stop_sample_index": 4,
             "start_lsl": stamps[3], "end_lsl": stamps[4], "reason": "clock_gap_over_100ms"},
        ])
        self.assertTrue(audit.window_intact(0, 4))
        self.assertTrue(audit.window_intact(4, 10))
        self.assertFalse(audit.window_intact(3, 5))

    def test_duplicate_and_reversed_times_fail_internal_clock_edges(self):
        values, _ = self.signal(5)
        stamps = [100., 100.05, 100.05, 100.01, 100.07]

        audit = audit_integrity(values, stamps, 20)

        self.assertEqual([row["reason"] for row in audit.spans], ["clock_nonincreasing"] * 2)
        self.assertEqual([(row["start_sample_index"], row["stop_sample_index"],
                           row["start_lsl"], row["end_lsl"]) for row in audit.spans],
                         [(2, 2, 100.05, 100.05), (3, 3, 100.05, 100.01)])
        self.assertFalse(audit.window_intact(1, 3))
        self.assertFalse(audit.window_intact(2, 4))
        self.assertTrue(audit.window_intact(0, 2))
        self.assertTrue(audit.window_intact(3, 5))

    def test_constant_signal_marks_actual_channel_including_end_of_recording(self):
        values, stamps = self.signal(50)
        values[0, 2:21] = 89.
        values[1, 30:50] = 89.

        audit = audit_integrity(values, stamps, 20)

        self.assertEqual(audit.spans, [
            {"channel": 2, "start_sample_index": 30, "stop_sample_index": 50,
             "start_lsl": stamps[30], "end_lsl": None, "reason": "constant_signal"},
        ])
        self.assertTrue(audit.window_intact(0, 30))
        self.assertFalse(audit.window_intact(29, 31))
        self.assertFalse(audit.window_intact(30, 50))

    def test_constant_samples_do_not_bridge_an_incomplete_clock(self):
        values, stamps = self.signal(20)
        values[0, :] = 89.
        stamps[10:] += .2

        audit = audit_integrity(values, stamps, 20)

        self.assertEqual([row["reason"] for row in audit.spans], ["clock_gap_over_100ms"])
        self.assertTrue(audit.window_intact(0, 10))
        self.assertTrue(audit.window_intact(10, 20))

    def test_empty_signal_is_supported_without_reporting_a_window(self):
        values, stamps = self.signal(0)

        audit = audit_integrity(values, stamps, 20)

        self.assertEqual(audit.spans, [])
        self.assertFalse(audit.window_intact(0, 0))
