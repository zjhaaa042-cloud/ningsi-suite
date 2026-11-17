"""Personal calibration through raw buffers, persistent binding and live quality gates."""

import json
from pathlib import Path
import tempfile
import threading
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ningsi_studio.acquisition.lsl import StreamBuffer, StreamDescriptor
from ningsi_studio.core.blink_control import BlinkCalibrationService, verify_binding
from ningsi_studio.core.live_source import ManagedLslSource
from ningsi_studio.db import repository, sqlite_store
from ningsi.signal.artifacts import check_channels
from ningsi.signal.live_artifacts import with_artifact_evidence


class BlinkCalibrationTests(unittest.TestCase):
    rate = 250

    def setUp(self):
        self.source = ManagedLslSource("headset", device="headset", fill_timeout=0)
        self.buffer = StreamBuffer(StreamDescriptor("eeg", "headset", "EEG", 2, self.rate,
            ("Fp1", "Fp2"), source_id="personal-headset", declared_units=("uV", "uV")))
        self.source._manager._buffers["eeg"] = self.buffer
        self.preview = SimpleNamespace(_source=SimpleNamespace(engine=self.source), _key="lsl:headset",
                                      _lock=threading.RLock(), start=lambda device: None)
        self.service = BlinkCalibrationService()
        self.t = np.arange(56*self.rate)/self.rate
        self.raw = np.stack([8*np.sin(2*np.pi*10*self.t), 7*np.sin(2*np.pi*10*self.t+.25)])
        # Realistic large DC and continuous drift: no artificial segment joins.
        self.raw += np.array([[-230000.], [-229000.]]) + .7*self.t
        self.peaks = np.arange(4., 49., 3.)
        for peak in self.peaks:
            eye = 145*np.exp(-((self.t-peak)/.14)**2)
            self.raw += np.stack([eye, .8*eye])
        self.end = 0

    def append_to(self, seconds):
        end = round(seconds*self.rate)
        self.buffer.append_chunk(self.raw[:, self.end:end].T, self.t[self.end:end])
        self.end = end

    def act(self, action, token=None, **changes):
        return self.service.action({"action": action, "participant": "p01", "device": "lsl:headset",
                                    "token": token, **changes}, self.preview)

    def calibrate(self):
        self.append_to(3)
        token = self.act("start")["token"]
        for peak in self.peaks:
            self.append_to(peak+.75)
            self.act("mark", token)
        result = self.act("finish", token)
        self.assertEqual(result["state"], "ready")
        self.assertEqual(result["count"], 15)
        return token, self.service.binding(token, "p01", "lsl:headset")

    def test_contiguous_personal_calibration_keeps_raw_untouched_and_binds_identity(self):
        token, binding = self.calibrate()
        self.assertGreaterEqual(binding["profile"]["events"], 8)
        verify_binding(binding, self.source)
        self.assertEqual(self.service._drafts[token]["chunks"], [])
        np.testing.assert_array_equal(self.buffer.samples_count(self.end), self.raw[:, :self.end])
        with self.assertRaises(ValueError):
            self.service.binding(token, "other", "lsl:headset")
        changed = ManagedLslSource("other", device="other")
        changed._manager._buffers["eeg"] = StreamBuffer(StreamDescriptor("eeg", "other", "EEG", 2,
            self.rate, ("Fp1", "Fp2"), declared_units=("uV", "uV")))
        with self.assertRaises(ValueError):
            verify_binding(binding, changed)

    def test_unfinished_duplicate_and_reconnected_marks_cannot_be_used(self):
        self.append_to(3)
        token = self.act("start")["token"]
        with self.assertRaises(ValueError):
            self.act("finish", token)
        self.append_to(4.75)
        self.act("mark", token)
        with self.assertRaises(ValueError):
            self.act("mark", token)
        self.source._manager._buffers["eeg"] = StreamBuffer(self.buffer.descriptor)
        with self.assertRaises(ValueError):
            self.act("status", token)

    def test_calibrated_correction_precedes_filter_and_unknown_blink_is_vetoed(self):
        _, binding = self.calibrate()
        t = np.arange(6*self.rate)/self.rate
        brain = np.stack([8*np.sin(2*np.pi*10*t), 7*np.sin(2*np.pi*10*t+.25)])
        eye = 145*np.exp(-((t-3.25)/.14)**2)
        raw = brain+np.stack([eye, .8*eye])+np.array([[-230000.], [-229000.]])
        before = raw.copy()
        _, unknown = self.source._condition(raw, 4*self.rate, return_artifacts=True)
        self.assertIn("blink_uncalibrated", unknown["reasons"])
        self.source.blink_binding = binding
        wave, evidence = self.source._condition(raw, 4*self.rate, return_artifacts=True)
        self.assertEqual(evidence["blink_control"]["corrected_events"], 1)
        self.assertFalse(evidence["blink_control"]["reasons"])
        self.assertLess(np.max(np.abs(wave)), 20)
        np.testing.assert_array_equal(before, raw)

    def test_blink_in_filter_context_does_not_reject_later_clean_scoring_window(self):
        t = np.arange(6*self.rate) / self.rate
        clean = np.stack([8*np.sin(2*np.pi*10*t), 7*np.sin(2*np.pi*10*t+.2)])
        for peak, accepted in ((.5, True), (1.8, False), (2.5, False), (5.9, False)):
            with self.subTest(peak=peak):
                eye = 145*np.exp(-((t-peak)/.14)**2)
                raw = clean + np.stack([eye, .8*eye])
                before = raw.copy()
                wave, evidence = self.source._condition(raw, 4*self.rate, return_artifacts=True)
                quality = with_artifact_evidence(check_channels(wave, self.rate), evidence)
                self.assertEqual(quality.ok, accepted, evidence)
                self.assertEqual(bool(evidence["blink_control"]["reasons"]), not accepted)
                self.assertEqual(wave.shape[1], 4*self.rate)
                if accepted:
                    self.assertEqual(evidence["blink_control"]["state"], "clean")
                    self.assertEqual(evidence["blink_control"]["masked_fraction"], 0.)
                    self.assertEqual(evidence["blink_control"]["events"], [])
                np.testing.assert_array_equal(raw, before)

    def test_binding_is_persisted_separately_from_voltage_calibration(self):
        _, binding = self.calibrate()
        with tempfile.TemporaryDirectory() as temporary:
            path = sqlite_store.initialize(Path(temporary)/"studio.sqlite")
            with sqlite_store.connect(path) as connection:
                subject_id = repository.create_subject(connection, "p01")
                row = repository.create_session(connection, subject_id, device="lsl:headset", blink_calibration=binding)
                self.assertIsNone(row["eeg_calibration"])
                self.assertEqual(json.loads(row["blink_calibration"]), binding)
            sqlite_store.initialize(path)  # migration remains idempotent

    def test_unit_change_invalidates_ready_profile_and_live_scores(self):
        token, binding = self.calibrate()
        self.source.blink_binding = binding
        self.buffer.descriptor = replace(self.buffer.descriptor, declared_units=("mV", "mV"))
        with self.assertRaises(ValueError):
            self.service.binding(token, "p01", "lsl:headset", self.preview)
        _, evidence = self.source._condition(self.raw[:, -1500:], 1000, return_artifacts=True)
        self.assertIn("blink_calibration_mismatch", evidence["reasons"])

    def test_invalid_tokens_and_repeated_finish_are_safe(self):
        token, _ = self.calibrate()
        self.assertEqual(self.act("finish", token)["state"], "ready")
        for token_value in ([], {}, None, 12):
            with self.subTest(token=token_value), self.assertRaises(ValueError):
                self.act("status", token_value)

    def test_same_buffer_worker_reconnect_invalidates_draft_ready_profile_and_running_gate(self):
        token, binding = self.calibrate()
        self.source.blink_binding = binding
        self.source._condition(self.raw[:, -1500:], 1000, return_artifacts=True)
        self.buffer.begin_connection()  # Same buffer and descriptor, new inlet/worker.
        with self.assertRaises(ValueError):
            self.act("status", token)
        with self.assertRaises(ValueError):
            self.service.binding(token, "p01", "lsl:headset", self.preview)
        _, evidence = self.source._condition(self.raw[:, -1500:], 1000, return_artifacts=True)
        self.assertIn("blink_calibration_mismatch", evidence["reasons"])


from tests.helpers import StudioTestCase


class BlinkCalibrationApiTests(StudioTestCase):
    port_base = 19186

    def test_api_validates_personal_tokens_and_incomplete_calibration(self):
        subject = self.create_subject()
        personal = subject["public_id"]
        source = ManagedLslSource("api-headset", device="api-headset", fill_timeout=0)
        buffer = StreamBuffer(StreamDescriptor("eeg", "api-headset", "EEG", 2, 250,
            ("Fp1", "Fp2"), declared_units=("uV", "uV")))
        t = np.arange(750)/250
        buffer.append_chunk(np.stack([8*np.sin(2*np.pi*10*t), 7*np.sin(2*np.pi*10*t+.2)], axis=1), t)
        source._manager._buffers["eeg"] = buffer
        preview = SimpleNamespace(_source=SimpleNamespace(engine=source), _key="lsl:api-headset",
                                  _lock=threading.RLock(), start=lambda device: None)
        service = BlinkCalibrationService()
        base = {"participant": personal, "device": "lsl:api-headset"}
        with patch("ningsi_studio.core.live_source.preview_stream", return_value=preview), patch(
                "ningsi_studio.core.blink_control.SERVICE", service):
            response = self.post("/api/devices/blink-calibration", {**base, "action": "start"})
            token = self.json_body(response, 200, "开始个人眨眼校准")["token"]
            for token_value in ([], {}, None, 12, "unknown"):
                response = self.post("/api/devices/blink-calibration", {**base, "action": "mark", "token": token_value})
                self.assertEqual(response.status, 422)
            self.assertEqual(self.post("/api/devices/blink-calibration", {**base, "action": "finish", "token": token}).status, 422)
            self.assertEqual(self.post("/api/devices/blink-calibration", {**base, "action": "clear", "token": token}).status, 200)


if __name__ == "__main__":
    unittest.main()
