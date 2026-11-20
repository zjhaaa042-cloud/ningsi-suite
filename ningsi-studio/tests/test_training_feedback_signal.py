from dataclasses import dataclass, replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from ningsi.signal.artifacts import Verdict
from ningsi.signal.conditioning import EYE, MOVEMENT
from ningsi_studio.domain.indicators import WindowStep
from ningsi_studio.domain.training_feedback import TrainingFeedbackSignal, local_eeg_feedback


@dataclass
class TestWindow:
    quality: Verdict

    @property
    def usable(self):
        return self.quality.ok


def window_step(index=1, *, reasons=(), mask=None, timing=None):
    t = np.arange(1000) / 250
    good = 15 * np.sin(2 * np.pi * 10 * t) + 10 * np.sin(2 * np.pi * 6 * t)
    return WindowStep(index, index * 2., "focused",
                      window=TestWindow(Verdict(not reasons, reasons, {})),
                      raw=np.array([good, good]), sample_timing=timing,
                      fragment_artifact_mask=mask)


def timing(end, *, generation=1, received=100.):
    return {"buffer_id": 1, "connection_generation": generation,
            "sample_end": end, "latest_sample_end": end, "received_at": received}


class TrainingFeedbackSignalTests(unittest.TestCase):
    def test_current_personal_index_requires_baseline_finite_grade_a_and_fresh_signal(self):
        engine = Mock()
        engine.source.assessment_timing_anchor.return_value = timing(1000)
        stamps = [timing(2000), timing(2000), timing(2500), timing(3000), timing(3500), timing(4000)]
        steps = [window_step(i, timing=s) for i,s in enumerate(stamps)]
        for step in steps:
            step.graded = {"grade": "A"}
            step.scores = {"focus": .75}
        steps[3].graded = {"grade": "B"}
        steps[4].scores = {"focus": float("nan")}
        engine.step.side_effect = steps
        signal = TrainingFeedbackSignal(SimpleNamespace(engine=engine, source=SimpleNamespace(kind="lsl", srate=250.), scale=1))
        with patch("ningsi_studio.domain.training_feedback.time.monotonic", return_value=100.5), \
             patch("ningsi_studio.domain.training_feedback.time.time", return_value=1000.):
            for i, baseline in enumerate((True, True, False, True, True)):
                payload = signal.presentation(signal.read("focused"), "guided", None, 2*i, baseline_valid=baseline)
                if i == 0:
                    self.assertEqual(payload['presentation_mode'], 'measured_focus')
                    self.assertEqual(payload['measured_focus_index'], {'value':.75,'source':'lsl','valid_until_unix_ms':1001500})
                else:
                    self.assertIsNone(payload['measured_focus_index'])
            delayed = signal.read("focused")
        with patch("ningsi_studio.domain.training_feedback.time.monotonic", return_value=102.):
            self.assertIsNone(signal.presentation(delayed, 'guided', None, 10, baseline_valid=True)['measured_focus_index'])
        self.assertEqual(signal.measured_focus_updates, 1)
        self.assertEqual(signal.measured_focus_only_updates, 1)

    def test_localized_rejection_recovers_only_untouched_half(self):
        mask = np.zeros(1000, dtype=np.uint8)
        mask[120:375] = MOVEMENT
        step = window_step(reasons=("motion_transient",), mask=mask)
        local = local_eeg_feedback(step, srate=250.)
        self.assertEqual([item["intervals"] for item in local["channels"]], [[[-2., 0.]], [[-2., 0.]]])
        self.assertTrue(all(item["relative"]["alpha"] > .6 for item in local["channels"]))
        self.assertFalse(step.window.quality.ok)
        self.assertIsNone(local_eeg_feedback(replace(step, fragment_artifact_mask=None), srate=250.))
        mask[:] = 0
        self.assertIsNone(local_eeg_feedback(step, srate=250.), "A matching mask is required to localize a veto")

    def test_unlocalized_blink_overlay_and_other_hard_fault_still_veto(self):
        mask = np.zeros(1000, dtype=np.uint8)
        mask[100:250] = EYE
        step = window_step(reasons=("blink_uncalibrated",), mask=mask)
        self.assertIsNone(local_eeg_feedback(step, srate=250.))
        step.fragment_localized_reasons = ("blink_uncalibrated",)
        self.assertTrue(local_eeg_feedback(step, srate=250.))
        step.window.quality = Verdict(False, ("blink_signal_fault",), {})
        step.fragment_localized_reasons = ("blink_signal_fault",)
        self.assertIsNone(local_eeg_feedback(step, srate=250.))

    def test_finite_other_channel_survives_flat_or_amplitude_gate_but_clock_veto_cannot_be_bypassed(self):
        step = window_step(reasons=("flat_channel", "amplitude"))
        step.raw[0, 500:] = 0.
        step.raw[0, :500] *= 10.
        local = local_eeg_feedback(step, srate=250.)
        self.assertEqual([item["channel"] for item in local["channels"]], [1])
        step.window.quality = Verdict(False, ("timestamp_gap_over_100ms",), {})
        self.assertIsNone(local_eeg_feedback(step, srate=250.))

    def test_real_feedback_rejects_repeats_old_input_and_reconnect_history(self):
        source = Mock()
        source.assessment_timing_anchor.return_value = timing(1000)
        engine = Mock(source=source)
        engine.step.side_effect = [window_step(i, timing=stamp) for i, stamp in enumerate((
            timing(1500), timing(2000), timing(2000), timing(2500),
            timing(2000, generation=2), timing(2500, generation=2),
            timing(3000, generation=2), timing(3500, generation=2, received=98.),
        ))]
        runtime = SimpleNamespace(engine=engine, source=SimpleNamespace(kind="lsl", srate=250.), scale=1.)
        signal = TrainingFeedbackSignal(runtime)
        with patch("ningsi_studio.domain.training_feedback.time.monotonic", return_value=100.5), \
             patch("ningsi_studio.domain.training_feedback.time.time", return_value=1_000.):
            live = [signal.capture(signal.read("focused"), None, i * 2.) for i in range(8)]
        self.assertEqual([i for i, payload in enumerate(live) if payload], [1, 3, 6])
        self.assertEqual(live[1]["valid_until_unix_ms"], 1_001_500)
        summary = signal.summary()
        self.assertEqual(summary["observations_count"], 10)
        self.assertEqual(summary["support_seconds"], 10.)
        self.assertEqual(summary["support_seconds_by_channel"], {"0": 10., "1": 10.})
        self.assertTrue(all("valid_until_unix_ms" not in row for row in summary["observations"]))


if __name__ == "__main__":
    unittest.main()
