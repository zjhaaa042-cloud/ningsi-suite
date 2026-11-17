"""Live EEG/auxiliary separation, per-window artifact control and clock pairing."""

import json
import os
import unittest
from unittest.mock import patch

import numpy as np

from ningsi.signal.live_artifacts import check_motion_reference, process_live_artifacts, voltage_scales
from ningsi_studio.acquisition.lsl import StreamBuffer, StreamDescriptor
from ningsi_studio.core.live_source import ManagedLslSource
from ningsi_studio.domain.indicators import WindowEngine


class LiveArtifactTests(unittest.TestCase):
    rate = 250

    def setUp(self):
        self.t = np.arange(6*self.rate) / self.rate
        self.alpha = 12 * np.sin(2*np.pi*10*self.t)
        self.eye = 70 * np.sin(2*np.pi*self.t)
        self.size = 4*self.rate

    def process(self, rows, names, units):
        rows = np.asarray(rows)
        return process_live_artifacts(rows, rows, self.rate, self.size, names, units)

    def test_reference_correction_preserves_known_alpha(self):
        out, info = self.process([self.alpha + .8*self.eye, self.eye], ["Fp1", "VEOG"], ["uV"]*2)
        known = self.alpha[-self.size:]
        self.assertEqual(info["ocular_corrected_channels"], [0])
        self.assertGreater(np.corrcoef(known, out[0])[0, 1], .995)
        amplitude = np.sum(out[0]*known) / np.sum(known*known)
        self.assertAlmostEqual(amplitude, 1, delta=.01)
        control, record = self.process([self.alpha, self.eye], ["Fp1", "VEOG"], ["uV"]*2)
        self.assertEqual(record["ocular_corrected_channels"], [])
        np.testing.assert_array_equal(control[0], known)

    def test_unknown_reference_unit_cannot_enable_eye_correction(self):
        out, info = self.process([self.alpha + self.eye, self.eye], ["Fp1", "VEOG"], ["uV", ""])
        self.assertEqual(out.shape, (1, self.size))
        self.assertEqual(info["ocular_reference_axes"], 0)
        self.assertEqual(voltage_scales(["ADC", "mV", "ADC"], 3, 1, [0, 1], .5), [.5, 1000, 1])

    def test_transient_exclusion_and_quiet_control(self):
        artifact = self.alpha.copy()
        artifact[3*self.rate:] += 70
        self.assertIn("electrode_step", self.process([artifact], ["Fp1"], ["uV"])[1]["reasons"])
        blink = 130 * np.exp(-((self.t-3.5)/.16)**2)
        self.assertIn("ocular_transient", self.process([self.alpha+blink], ["Fp1"], ["uV"])[1]["reasons"])
        burst = 30*np.sin(2*np.pi*38*self.t) * ((self.t > 3) & (self.t < 4))
        self.assertIn("muscle_burst", self.process([self.alpha+burst], ["Fp1"], ["uV"])[1]["reasons"])
        record = self.process([self.alpha, 2*burst], ["Fp1", "EMG1"], ["uV"]*2)[1]
        self.assertIn("emg_reference_burst", record["reasons"])
        _, quiet = self.process([self.alpha], ["Fp1"], ["uV"])
        self.assertEqual(quiet["reasons"], [])

    def test_motion_stream_window_and_missing_reference_gate_scoring(self):
        profile = {"eeg": "headset", "motion": "headset-imu", "axes": [0,1,2], "unit": "g"}
        with patch.dict(os.environ, {"NINGSI_MOTION_REFERENCE": json.dumps(profile)}):
            source = ManagedLslSource("headset", device="headset", fill_timeout=0)
        eeg = StreamBuffer(StreamDescriptor("eeg", "headset", "EEG", 1, self.rate, ("Fp1",), declared_units=("uV",)))
        eeg.append_chunk(self.alpha[:, None], self.t)
        source._manager._buffers["eeg"] = eeg
        engine = WindowEngine(source, self.rate, device="headset")
        engine.graded_enabled = True
        missing = engine.step()
        self.assertIn("motion_reference_unavailable", missing.window.quality.reasons)
        mt = np.arange(6*20)/20
        motion = StreamBuffer(StreamDescriptor("motion", "headset-imu", "Motion", 3, 20,
                                              ("Accel X", "Accel Y", "Accel Z"), declared_units=("g",)*3))
        motion.append_chunk(np.stack([.21*(mt > 4), mt*0, mt*0], axis=1), mt)
        source._motion_manager._buffers["motion"] = motion
        step = engine.step()
        self.assertIn("motion_reference_excursion", step.window.quality.reasons)
        self.assertEqual(step.graded["grade"], "C")
        self.assertFalse(step.scores)
        self.assertEqual(step.payload()["quality"]["metrics"]["artifact_control"]["motion_reference"]["unit"], "g")
        snap = source.model_snapshot()
        np.testing.assert_array_equal(snap["samples"][0], self.alpha)

    def test_active_capture_model_baseline_uses_the_training_filter(self):
        from ningsi.signal import conditioning
        source = ManagedLslSource("headset", device="headset", fill_timeout=0, research_timing=True)
        profile = {"srate": self.rate, "scale_to_uv": 1., "preprocessing_policy": conditioning.POLICY}
        source.research_profile = profile
        eeg = StreamBuffer(StreamDescriptor("eeg", "headset", "EEG", 1, self.rate, ("Fp1",), declared_units=("uV",)))
        raw = self.alpha + 700*np.sin(2*np.pi*50*self.t)
        eeg.append_chunk(raw[:, None], self.t)
        source._manager._buffers["eeg"] = eeg
        wave, evidence = source.analysis_window()
        expected = conditioning.condition(raw[None, :], self.t, self.rate, 1., conditioning.POLICY, traces=True)
        self.assertEqual(evidence["spec"], conditioning.POLICY)
        self.assertEqual(evidence["reasons"], list(expected["window"].quality.reasons))
        np.testing.assert_allclose(wave, expected["filtered_uv"], atol=1e-10)

    def test_stale_alignment_or_conflicting_motion_units_are_unavailable(self):
        stamps = self.t[::10]
        snap = {"age": .01, "timestamps": stamps, "samples": np.zeros((3,len(stamps))),
                "descriptor": {"declared_units": ["mg"]*3}}
        self.assertEqual(check_motion_reference(snap, self.t, [0,1,2], "g")["detail"],
                         "acceleration_unit_conflict")
        self.assertEqual(check_motion_reference(snap, self.t+10, [0,1,2], "mg")["detail"],
                         "motion_clock_coverage")

    def test_frozen_model_blink_gate_uses_only_the_scored_window(self):
        from ningsi.signal import conditioning
        source = ManagedLslSource('headset', device='headset', fill_timeout=0, research_timing=True)
        source.research_profile = {'srate': self.rate, 'scale_to_uv': 1., 'preprocessing_policy': conditioning.POLICY}
        stamps = np.arange(7.5*self.rate) / self.rate
        for peak, veto in ((1., False), (4., True), (7., False)):
            with self.subTest(peak=peak):
                buffer = StreamBuffer(StreamDescriptor('eeg','headset','EEG',2,self.rate,('Fp1','Fp2'),declared_units=('uV','uV')))
                blink = 145*np.exp(-((stamps-peak)/.14)**2)
                channels = np.stack([8*np.sin(2*np.pi*10*stamps)+blink, 7*np.sin(2*np.pi*10*stamps+.25)+.8*blink])
                buffer.append_chunk(channels.T, stamps)
                source._manager._buffers['eeg'] = buffer
                _, evidence = source.analysis_window()
                self.assertEqual(evidence['sample_end_index'], 1499)  # [2,6)s is scored at 7.5s.
                self.assertEqual(bool(evidence['blink_control']['reasons']), veto, evidence)
