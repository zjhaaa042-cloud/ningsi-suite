import tempfile
import unittest

from .test_support import FakeClock
import numpy as np

from ningsi.acquisition.timing import PACKET_TIMING, analysis_timestamps
from ningsi.models.research_features import condition_window
from ningsi_capture.acquisition import simulation_source, synthetic_samples
from ningsi_capture.export import analysis_windows, recent_window_quality
from ningsi_capture.storage import SessionStore, open_readonly


class PacketTimingTests(unittest.TestCase):
    def context(self):
        indices = np.arange(1500)
        clean = 1000. + indices / 250
        packet_jitter = .017 * np.sin((indices // 10) * 1.7)
        return synthetic_samples(indices).T, clean + packet_jitter

    def test_bounded_packet_jitter_keeps_values_and_produces_uniform_analysis_clock(self):
        values, raw = self.context()
        original = raw.copy()
        strict = condition_window(values, raw, 250, 1.)
        self.assertIn("nonmonotonic_timestamps", strict["window"].quality.reasons)
        times, audit = analysis_timestamps(raw, 250, PACKET_TIMING)
        self.assertTrue(audit["applied"])
        self.assertLessEqual(audit["max_shift_sec"], .05)
        np.testing.assert_allclose(np.diff(times), .004, atol=1e-10)
        np.testing.assert_array_equal(raw, original)
        self.assertTrue(condition_window(values, times, 250, 1.)["window"].usable)

    def test_large_gap_reset_rate_error_and_missing_clock_stay_rejected(self):
        values, raw = self.context()
        gap = raw.copy(); gap[750:] += .3
        reset = raw.copy(); reset[750:] -= 1.
        drift = 1000. + np.arange(1500) / 220
        missing = raw.copy(); missing[500] = np.nan
        constant = np.full(1500, 1000.)
        for times in (gap, reset, drift, missing, constant):
            with self.subTest(times=times[750]):
                estimated, audit = analysis_timestamps(times, 250, PACKET_TIMING)
                self.assertFalse(audit["applied"])
                self.assertFalse(condition_window(values, estimated, 250, 1.)["window"].usable)

    def test_unknown_unit_and_flat_signal_are_never_fixed_by_timing_policy(self):
        values, raw = self.context()
        times, _ = analysis_timestamps(raw, 250, PACKET_TIMING)
        self.assertIn("unit_unconfirmed", condition_window(values, times, 250, None)["window"].quality.reasons)
        self.assertIn("flat_channel", condition_window(np.zeros_like(values), times, 250, 1.)["window"].quality.reasons)

    def test_live_precheck_export_and_raw_storage_agree(self):
        values, stamps = self.context()
        with tempfile.TemporaryDirectory() as directory:
            source = {**simulation_source(), "timestamp_policy": PACKET_TIMING}
            store = SessionStore(directory, participant="PACKET", source=source, protocol={}, clock=FakeClock())
            try:
                store.append(values.T, stamps, offset=0.)
                self.assertEqual(recent_window_quality(store.path), ())
                reader = open_readonly(store.path)
                try:
                    rows = analysis_windows(reader, store.metadata)
                    raw = reader.execute("select source_timestamp_bytes from samples order by sample_index").fetchall()
                    self.assertEqual(b"".join(row[0] for row in raw), stamps.astype('<f8').tobytes())
                    self.assertTrue(rows[-1]["window"].usable)
                    self.assertTrue(rows[-1]["timestamp_adjustment"]["applied"])
                    # Custom candidate analyzers receive exactly the same timing.
                    custom = analysis_windows(reader, store.metadata, analyze=lambda context, source: {
                        "start":context[-1000][1], "end":context[-1][1] + .004})
                    self.assertAlmostEqual(rows[-1]["start"], custom[-1]["start"])
                    self.assertAlmostEqual(rows[-1]["end"], custom[-1]["end"])
                finally:
                    reader.close()
            finally:
                store.close("aborted")

    def test_strict_default_and_unknown_policy(self):
        _, raw = self.context()
        times, audit = analysis_timestamps(raw, 250)
        np.testing.assert_array_equal(times, raw)
        self.assertFalse(audit["applied"])
        with self.assertRaises(ValueError):
            analysis_timestamps(raw, 250, 'unrecognized')
