"""Artifact recovery with held-out synthetic EEG; not a human efficacy claim."""

import copy
import unittest

import numpy as np

from ningsi.signal.blinks import BlinkCalibration, calibrate_blinks, process_blinks
from ningsi.signal.window import analyze_window


RATE = 250
LABELS = ("Fp1", "Fp2")
MONTAGE = "fixture:linked-ear"


def brain(seconds, seed):
    random = np.random.default_rng(seed)
    time = np.arange(round(seconds * RATE)) / RATE
    return np.stack([sum(amplitude * np.sin(2*np.pi*freq*time + random.uniform(-np.pi, np.pi))
                         for freq, amplitude in ((6, 5), (10, 8), (20, 3)))
                     + random.normal(0, 1.2, len(time)) for _ in range(2)])


def add_blink(data, peak, amplitude=130., width=.105, ratio=.75):
    time = np.arange(data.shape[1]) / RATE - peak
    for channel, scale in enumerate((1., ratio)):
        data[channel] += amplitude * scale * np.exp(-.5 * (time / width) ** 2)


def profile():
    values = brain(50, 10)
    marks = np.arange(2, 48, 3.2)
    for index, peak in enumerate(marks):
        add_blink(values, peak, 120 + 15*np.sin(index), .105)
    return calibrate_blinks(values, RATE, [round(p*RATE) for p in marks],
                            channel_labels=LABELS, montage_id=MONTAGE)


class DualBlinkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = profile()

    def apply(self, values, calibration=None, **kwargs):
        return process_blinks(values, RATE, self.profile if calibration is None else calibration,
                              channel_labels=LABELS, montage_id=MONTAGE, **kwargs)

    def test_held_out_variable_amplitude_and_width_reduce_error(self):
        for width, amplitude, seed in ((.084, 95, 29), (.105, 150, 37), (.126, 180, 43)):
            with self.subTest(width=width):
                original = brain(4, seed)
                contaminated = original.copy()
                add_blink(contaminated, 1.8, amplitude, width)
                before = contaminated.copy()
                result = self.apply(contaminated)
                self.assertEqual(result.as_dict()["corrected_events"], 1, result.as_dict())
                self.assertFalse(result.mask.any())
                self.assertLess(np.mean((result.cleaned_uv-original)**2),
                                .08*np.mean((contaminated-original)**2))
                np.testing.assert_array_equal(contaminated, before)
                np.testing.assert_array_equal(result.cleaned_uv[:, ~result.corrected_mask],
                                              contaminated[:, ~result.corrected_mask])

    def test_clean_theta_alpha_beta_and_common_activity_are_unchanged(self):
        for seed in range(20):
            original = brain(4, seed)
            original[1] = original[0]  # Common EEG is not an ocular reference.
            result = self.apply(original)
            self.assertFalse(result.mask.any())
            self.assertFalse(result.corrected_mask.any())
            np.testing.assert_array_equal(result.cleaned_uv, original)

    def test_blink_correction_preserves_known_neural_bands(self):
        # Compare to a separately generated clean ground truth, not the corrupt
        # target. A smoother-looking waveform alone is insufficient acceptance.
        for seed in range(100, 120):
            clean = brain(4, seed)
            corrupt = clean.copy()
            add_blink(corrupt, 1.8, 150, .10)
            result = self.apply(corrupt)
            self.assertFalse(result.mask.any())
            expected = analyze_window(clean, RATE).powers
            measured = analyze_window(result.cleaned_uv, RATE).powers
            for band, limit in (("theta", .05), ("alpha", .01), ("beta", .01)):
                with self.subTest(seed=seed, band=band):
                    self.assertLess(abs(measured[band] / expected[band] - 1), limit)

    def test_negative_polarity_and_large_device_offsets(self):
        encoded = self.profile.as_dict()
        encoded["templates"] = [[-v for v in row] for row in encoded["templates"]]
        negative_profile = BlinkCalibration.from_dict(encoded)
        original = brain(4, 69) + np.array([[-230000.], [-215000.]])
        corrupt = original.copy()
        add_blink(corrupt, 2., -130.)
        result = self.apply(corrupt, negative_profile)
        self.assertFalse(result.mask.any())
        self.assertEqual(result.as_dict()["corrected_events"], 1)
        self.assertLess(np.mean((result.cleaned_uv-original)**2),
                        .08*np.mean((corrupt-original)**2))

    def test_partial_and_dense_events_are_masked_without_subtraction(self):
        for peaks in ((.1,), (3.85,), (1.2, 1.75)):
            with self.subTest(peaks=peaks):
                values = brain(4, 31)
                for peak in peaks:
                    add_blink(values, peak)
                result = self.apply(values)
                self.assertTrue(result.mask.any(), result.as_dict())
                self.assertFalse(result.corrected_mask.any())
                np.testing.assert_array_equal(result.cleaned_uv, values)

    def test_one_sided_contact_and_ratio_mismatch_fail_closed(self):
        for ratio in (0., .15, 2.):
            values = brain(4, 31)
            add_blink(values, 2., ratio=ratio)
            result = self.apply(values)
            self.assertTrue(result.mask.any())
            self.assertFalse(result.corrected_mask.any())
            np.testing.assert_array_equal(result.cleaned_uv, values)

    def test_faults_not_hidden_by_template(self):
        values = brain(4, 31)
        add_blink(values, 2.)
        for fault in ("nan", "step", "dropout", "saturation"):
            with self.subTest(fault=fault):
                damaged = values.copy()
                if fault == "nan":
                    damaged[0, 10] = np.nan
                elif fault == "step":
                    damaged[0, 100:] += 120
                elif fault == "dropout":
                    damaged[0, 100:150] = 5
                else:
                    damaged[0, 100:105] = 1500
                result = self.apply(damaged)
                self.assertTrue(result.mask.all())
                self.assertFalse(result.corrected_mask.any())

    def test_no_calibration_is_detection_only(self):
        values = brain(4, 31)
        add_blink(values, 2.)
        result = process_blinks(values, RATE, channel_labels=LABELS, montage_id=MONTAGE)
        self.assertTrue(result.mask.any())
        self.assertIn("blink_uncalibrated", result.reasons)
        np.testing.assert_array_equal(result.cleaned_uv, values)

    def test_scoring_suffix_counts_only_overlapping_events_but_retains_signal_faults(self):
        values = brain(6, 31)
        add_blink(values, .9)
        result = self.apply(values)
        self.assertEqual(result.as_dict()["corrected_events"], 1)
        current = result.as_dict(start_index=2*RATE)
        self.assertEqual(current["state"], "clean")
        self.assertEqual(current["corrected_events"], 0)
        self.assertEqual(current["corrected_fraction"], 0.)
        self.assertEqual(current["events"], [])
        self.assertTrue(result.corrected_mask.any(), "上下文校正本身仍需保留")
        broken = values.copy()
        broken[0, 10] = np.nan
        fault = self.apply(broken).as_dict(start_index=2*RATE)
        self.assertEqual(fault["state"], "rejected")
        self.assertIn("non_finite_signal", fault["reasons"])
        self.assertEqual(fault["masked_fraction"], 1.)

    def test_identity_mismatch_and_swapped_channels_reject_whole_window(self):
        clean = brain(4, 10)
        for labels, montage, rate in ((LABELS[::-1], MONTAGE, RATE), (LABELS, "other-reference", RATE),
                                     (LABELS, MONTAGE, 200)):
            result = process_blinks(clean, rate, self.profile, channel_labels=labels, montage_id=montage)
            self.assertTrue(result.mask.all())
            self.assertIn("blink_calibration_mismatch", result.reasons)

    def test_profile_roundtrip_rejects_malformed_or_unsafe_payload(self):
        encoded = self.profile.as_dict()
        self.assertEqual(BlinkCalibration.from_dict(encoded), self.profile)
        for key, value in (("spec", "unrecognized"), ("rate", float("nan")), ("events", 2),
                           ("channel_labels", ["Fp1", "Fp1"]), ("templates", [[0], [0]]),
                           ("noise_uv", [1, -1]), ("ratio_limits", [2, 1])):
            damaged = copy.deepcopy(encoded)
            damaged[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                BlinkCalibration.from_dict(damaged)
        encoded["templates"][0][10] = 99
        self.assertNotEqual(encoded["templates"][0][10], self.profile.templates[0][10])

    def test_calibration_needs_marked_blinks_not_eye_state_baseline(self):
        clean = brain(40, 23)
        for marks in ([], [500], list(range(500, 750, 10)), list(range(500, 9000, 800))):
            with self.subTest(marks=marks), self.assertRaises(ValueError):
                calibrate_blinks(clean, RATE, marks, montage_id=MONTAGE)


if __name__ == "__main__":
    unittest.main()
