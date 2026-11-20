import unittest

import numpy as np

from ningsi.signal import conditioning
from ningsi.signal.bands import relative_band_powers
from ningsi.signal.spectrum import welch_psd
from ningsi_capture.acquisition import synthetic_samples
from ningsi_capture.export import _analyze_samples
from ningsi_capture.native_motion import NativeMotion
from ningsi_capture.spectrum_quality import (describe_interval_spectrum, describe_motion_assisted_spectrum,
                                              describe_single_channels, describe_spectrum)


class DescriptiveSpectrumTests(unittest.TestCase):
    def setUp(self):
        self.rate = 250
        self.stamps = np.arange(1500) / self.rate
        self.clean = synthetic_samples(np.arange(1500)).T

    def evaluate(self, signal, stamps=None):
        stamps = self.stamps if stamps is None else stamps
        strict = conditioning.condition(signal, stamps, self.rate, 1., conditioning.POLICY)["window"]
        return strict, describe_spectrum(signal, stamps, self.rate, 1., strict)

    def test_complete_window_reuses_existing_power(self):
        strict, descriptive = self.evaluate(self.clean)
        self.assertTrue(strict.usable)
        self.assertEqual(descriptive["status"], "full")
        self.assertEqual(descriptive["support_sec"], 4.)
        self.assertEqual(descriptive["band_powers"], strict.powers)

    def test_recovers_clean_segment_before_transient(self):
        noisy = self.clean.copy()
        noisy[1, 1350:] += 220
        strict, descriptive = self.evaluate(noisy)
        self.assertFalse(strict.usable)
        self.assertEqual(descriptive["status"], "partial")
        self.assertEqual(descriptive["intervals"], [(2., 4.)])
        self.assertEqual(descriptive["support_sec"], 2.)
        independently_filtered = conditioning.filter_uv(self.clean[:, :1000], self.rate)
        spectra = [welch_psd(channel, self.rate) for channel in independently_filtered]
        self.assertEqual(descriptive["relative_bands"], relative_band_powers(
            spectra[0].freqs, np.mean([spectrum.psd for spectrum in spectra], axis=0)))

    def test_interval_pools_distinct_full_segments_after_reporting_boundary(self):
        indices = np.arange(11 * self.rate)
        raw, stamps = synthetic_samples(indices).T, indices / self.rate
        first = 3 * self.rate
        spectrum = describe_interval_spectrum(raw, stamps, self.rate, 1., first=first)
        self.assertEqual(spectrum["status"], "partial")
        self.assertEqual(spectrum["support_sec"], 8.)
        self.assertEqual(spectrum["segment_lengths_sec"], [2.] * 4)
        np.testing.assert_allclose(spectrum["intervals"], [(3., 5.), (5., 7.), (7., 9.), (9., 11.)])
        wave = conditioning.filter_uv(raw[:, first - 2 * self.rate:], self.rate)
        spectra = [welch_psd(channel, self.rate) for segment in np.split(wave, 4, axis=1)
                   for channel in segment]
        expected = relative_band_powers(spectra[0].freqs, np.mean([row.psd for row in spectra], axis=0))
        for band, value in expected.items():
            self.assertAlmostEqual(spectrum["relative_bands"][band], value, places=12)

    def test_interval_restarts_filter_history_after_clock_gap(self):
        indices = np.arange(14 * self.rate)
        raw, stamps = synthetic_samples(indices).T, indices / self.rate
        stamps[int(7 * self.rate):int(7.5 * self.rate)] = np.nan
        result = describe_interval_spectrum(raw, stamps, self.rate, 1., first=3 * self.rate)
        self.assertEqual(result["support_sec"], 8.)
        np.testing.assert_allclose(result["intervals"], [(3., 5.), (5., 7.), (9.5, 11.5), (11.5, 13.5)])

    def test_interval_uses_nonoverlapping_boundaries_when_clock_runs_slightly_fast(self):
        stamps = self.stamps * 0.999
        result = describe_interval_spectrum(self.clean, stamps, self.rate, 1., first=2 * self.rate)
        first, second = result["intervals"]
        self.assertEqual(result["segments"], 2)
        self.assertEqual(first[1], second[0])

    def test_interval_does_not_replace_two_second_segments_with_short_fragments(self):
        raw = self.clean.copy()
        raw[0, 950:955] += 200.
        _, window = self.evaluate(raw)
        self.assertEqual(window["method"], "short_fragments")
        interval = describe_interval_spectrum(raw, self.stamps, self.rate, 1., first=2 * self.rate)
        self.assertEqual(interval["status"], "unavailable")
        self.assertEqual(interval["intervals"], [])

    def test_short_fragments_exclude_transient_and_its_mask(self):
        noisy = self.clean.copy()
        noisy[0, 950:955] += 200.
        result = conditioning.condition(noisy, self.stamps, self.rate, 1., conditioning.POLICY, traces=True)
        strict = result["window"]
        descriptive = describe_spectrum(noisy, self.stamps, self.rate, 1., strict)
        self.assertFalse(strict.usable)
        self.assertEqual(descriptive["status"], "partial")
        self.assertEqual(descriptive["method"], "short_fragments")
        self.assertEqual(descriptive["segment_lengths_sec"], [1.5, 1.5])
        self.assertEqual(descriptive["support_sec"], 3.)
        intervals = descriptive["intervals"]
        self.assertLess(intervals[0][1], self.stamps[950])
        self.assertGreater(intervals[1][0], self.stamps[954])
        for left, right in intervals:
            start, stop = round((left - 2) * self.rate), round((right - 2) * self.rate)
            self.assertFalse(result["artifact_mask"][start:stop].any())

    def test_requires_local_history_and_does_not_bridge_unknown_time(self):
        stamps = self.stamps.copy()
        stamps[1100:1150] = np.nan
        _, descriptive = self.evaluate(self.clean, stamps)
        self.assertEqual(descriptive["status"], "partial")
        self.assertEqual(descriptive["intervals"], [(2., 4.)])

        stamps[650:700] = np.nan
        _, descriptive = self.evaluate(self.clean, stamps)
        self.assertEqual(descriptive["status"], "unavailable")
        self.assertEqual(descriptive["support_sec"], 0.)
        self.assertFalse(descriptive["band_powers"])

    def test_clock_reset_cannot_create_overlapping_support(self):
        stamps = self.stamps.copy()
        stamps[1300:] -= 2
        _, descriptive = self.evaluate(self.clean, stamps)
        self.assertEqual(descriptive["status"], "unavailable")

    def test_persistent_artifacts_never_generate_power(self):
        motion = self.clean + 220 * np.sign(np.sin(2 * np.pi * 2 * self.stamps))
        emg = self.clean + 90 * np.sin(2 * np.pi * 36 * self.stamps)
        contact = self.clean.copy()
        contact[0] = 4095.
        for signal in (motion, emg, contact):
            with self.subTest(signal=signal[0, :3]):
                _, descriptive = self.evaluate(signal)
                self.assertEqual(descriptive["status"], "unavailable")
                self.assertFalse(descriptive["relative_bands"])

    def evaluate_with_gyro(self, signal, trace):
        strict, normal = self.evaluate(signal)
        return strict, normal, describe_motion_assisted_spectrum(signal, self.stamps, self.rate, 1., strict, normal, trace)

    def gyro_trace(self, *, moving=False, gap=False):
        packets = np.arange(85)
        motion = np.zeros((len(packets), 3))
        motion[:, 0] = .10 + .01 * (packets % 2)
        if moving:
            motion[49, 0] = 2.
        keep = packets != 49 if gap else np.ones(len(packets), dtype=bool)
        return NativeMotion(packets[keep] / 12.5, motion[keep], packets[keep])

    def test_aligned_motion_vetoes_descriptive_bands_even_with_good_eeg(self):
        trace = self.gyro_trace()
        trace.gyro_xyz[25:79:9, 0] = 2.
        strict, _ = self.evaluate(self.clean)
        self.assertTrue(strict.usable)
        descriptive = describe_spectrum(self.clean, self.stamps, self.rate, 1., strict, motion=trace)
        self.assertEqual(descriptive["status"], "unavailable")
        self.assertEqual(descriptive["intervals"], [])

    def test_interval_retains_existing_scale_and_motion_guards(self):
        trace = self.gyro_trace()
        trace.gyro_xyz[25:79:9, 0] = 2.
        for scale, motion in ((None, None), (1., trace)):
            with self.subTest(scale=scale):
                result = describe_interval_spectrum(self.clean, self.stamps, self.rate, scale,
                                                    first=2 * self.rate, motion=motion)
                self.assertEqual(result["status"], "unavailable")
                self.assertFalse(result["band_powers"])

    def test_capture_keeps_independent_lead_without_creating_strict_features(self):
        raw = self.clean.copy()
        raw[0] += 65 * np.sin(2 * np.pi * 18 * self.stamps)
        source = {"channels": 2, "channel_labels": ("left", "right"), "srate": self.rate,
                  "preprocessing_policy": conditioning.POLICY, "scale_to_uv": 1.}
        context = list(zip(raw.T, self.stamps))
        result = _analyze_samples(context, source, include_spectral=True, motion=self.gyro_trace())
        self.assertFalse(result["window"].usable)
        self.assertEqual(result["features"], {})
        self.assertEqual(result["descriptive_spectrum"]["status"], "unavailable")
        right, = result["single_channel_spectra"]
        self.assertEqual((right["channel_index"], right["channel_label"]), (2, "right"))
        self.assertEqual(right["intervals"], [(2., 4.), (4., 6.)])
        expected = conditioning.filter_uv(self.clean[1:2], self.rate)[0]
        spectra = [welch_psd(piece, self.rate) for piece in np.split(expected, 2)]
        self.assertEqual(right["relative_bands"], relative_band_powers(
            spectra[0].freqs, np.mean([item.psd for item in spectra], axis=0)))
        unaligned = _analyze_samples(context, source, include_spectral=True)
        self.assertEqual(unaligned["single_channel_spectra"], [])

    def test_other_lead_eye_muscle_and_contact_veto_single_channel_recovery(self):
        for artifact in (65 * np.sin(2 * np.pi * 2 * self.stamps),
                         90 * np.sin(2 * np.pi * 36 * self.stamps),
                         4095. - self.clean[0]):
            with self.subTest(artifact=artifact[:3]):
                raw = self.clean.copy()
                raw[0] += artifact
                _, dual = self.evaluate(raw)
                kept = describe_single_channels(raw, self.stamps, self.rate, 1., dual, self.gyro_trace(), ("L", "R"))
                self.assertEqual(kept, [])

    def test_quiet_gyro_retains_unmarked_preceding_segment_only(self):
        raw = self.clean.copy()
        raw[0, (self.stamps >= 3.944) & (self.stamps < 5.2)] += 65.
        strict, normal, assisted = self.evaluate_with_gyro(raw, self.gyro_trace())
        self.assertEqual(strict.quality.reasons, ("motion_transient",))
        self.assertEqual(normal["status"], "unavailable")
        self.assertEqual(assisted["decision"], "quiet_imu_near_full_clean_segment")
        self.assertGreaterEqual(assisted["support_sec"], 1.90)
        (left, right), = assisted["intervals"]
        self.assertLess(right, 3.944)
        clean_prefix = self.clean[:, round((left - 2) * self.rate):round(right * self.rate)]
        wave = conditioning.filter_uv(clean_prefix, self.rate)
        spectra = [welch_psd(channel, self.rate, segment_sec=assisted["support_sec"]) for channel in wave]
        reference = relative_band_powers(spectra[0].freqs, np.mean([spectrum.psd for spectrum in spectra], axis=0))
        for band, power in reference.items():
            self.assertAlmostEqual(assisted["relative_bands"][band], power, places=12)

    def test_motion_or_gyro_gap_keeps_motion_rejection(self):
        raw = self.clean.copy()
        raw[0, (self.stamps >= 3.944) & (self.stamps < 5.2)] += 65.
        for trace, decision in ((self.gyro_trace(moving=True), "imu_motion"),
                                (self.gyro_trace(gap=True), "imu_coverage_incomplete")):
            with self.subTest(decision=decision):
                _, _, assisted = self.evaluate_with_gyro(raw, trace)
                self.assertEqual(assisted["status"], "unavailable")
                self.assertEqual(assisted["decision"], decision)

    def test_quiet_gyro_does_not_override_muscle_or_contact_gate(self):
        emg = self.clean + 95 * np.sin(2 * np.pi * 36 * self.stamps)
        plateau = self.clean.copy()
        plateau[0, 550:] = 4095.
        for raw in (emg, plateau):
            _, _, assisted = self.evaluate_with_gyro(raw, self.gyro_trace())
            self.assertEqual(assisted["status"], "unavailable")
            self.assertFalse(assisted["relative_bands"])
