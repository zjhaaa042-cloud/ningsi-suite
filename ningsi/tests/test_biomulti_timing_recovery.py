import unittest

import numpy as np

from ningsi.acquisition.timing import (
    BIOMULTI_LSL_RECOVERY_TIMING,
    BIOMULTI_MAX_DELAY,
    analysis_timestamps,
    biomulti_packet_clock,
    biomulti_recovery_clock,
)


def _releases(delays):
    """Backdate each available ten-sample packet from its release timestamp."""
    sample_count = len(delays) * 10
    return 1000. + np.arange(sample_count) / 250. + np.repeat(delays, 10)


class BioMultiTimingRecoveryTests(unittest.TestCase):
    def test_recovers_observed_packets_after_delay_catches_up(self):
        delays = np.zeros(200)
        delays[75:98] = .08
        delays[98:100] = [.045, .01]
        raw = _releases(delays)
        saved = raw.copy()
        old = biomulti_packet_clock(raw)
        recovered, audit = analysis_timestamps(raw, 250, BIOMULTI_LSL_RECOVERY_TIMING)

        self.assertFalse(np.isfinite(old[750:1000]).any())
        self.assertTrue(np.isfinite(recovered[750:1000]).all())
        self.assertTrue((np.diff(recovered[:1750]) > 0).all())
        self.assertLessEqual(audit["max_shift_sec"], BIOMULTI_MAX_DELAY)
        self.assertTrue(np.isnan(recovered[-250:]).all())
        np.testing.assert_array_equal(raw, saved)

    def test_keeps_unresolved_and_hard_gaps_inside_packets(self):
        delays = np.resize([0., .002, .003, .004, .085, .052, .020, .001], 260)
        for gap in (.18, .30):
            with self.subTest(gap=gap):
                raw = _releases(delays.copy())[3:]
                # The first delayed packet straddles a one-second boundary.
                raw[1247:] += gap
                recovered = biomulti_recovery_clock(raw)
                self.assertTrue(np.isnan(recovered[1247:1257]).all())
                self.assertTrue(np.isfinite(recovered[1500:2000]).all())
                self.assertTrue((np.diff(recovered[np.isfinite(recovered)]) > 0).all())

    def test_same_axis_for_recording_and_confirmed_prefix(self):
        raw = _releases(np.resize([0., .002, .003, .004, .085, .052, .020, .001], 200))
        full = biomulti_recovery_clock(raw)
        prefix = biomulti_recovery_clock(raw[:1500])
        part = biomulti_recovery_clock(raw[7:1500], first_index=7)

        np.testing.assert_array_equal(full[:1250], prefix[:1250])
        np.testing.assert_allclose(full[500:1250], part[500 - 7 : 1250 - 7], rtol=0, atol=1e-10)
        self.assertTrue(np.isnan(prefix[1250:]).all())

    def test_keeps_clock_reset_and_malformed_packet_unavailable(self):
        raw = _releases(np.resize([0., .002, .003, .004, .085, .052, .020, .001], 300))
        raw[1002] = np.nan
        raw[1752] -= .08
        raw[2400:] -= 5.
        recovered = biomulti_recovery_clock(raw)

        self.assertTrue(np.isnan(recovered[1000:1250]).all())
        self.assertTrue(np.isnan(recovered[1750:2000]).all())
        self.assertTrue(np.isnan(recovered[2400:]).all())
