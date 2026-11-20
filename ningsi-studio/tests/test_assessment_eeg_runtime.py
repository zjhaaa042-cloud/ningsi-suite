"""Assessment collection regressions using the normal window and scoring engines."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from ningsi.acquisition.simulate import SyntheticEEG
from ningsi.scales.scoring import score_scale
from ningsi_studio.core.assessment_eeg import AssessmentSampleClock
from ningsi_studio.core.runtime import SessionRuntime
from ningsi_studio.domain.indicators import WindowEngine


class ScheduledSignal:
    def __init__(self):
        self.signal = SyntheticEEG(srate=250, channels=2, seed=15)
        self.usable_at = lambda _: True
        self.index = 0

    def window(self, state, *, artifact=False):
        self.index += 1
        if self.usable_at(self.index):
            return self.signal.window(state, artifact=artifact)
        return np.zeros((2, 1000))

    def schedule(self, usable_at):
        self.index = 0
        self.usable_at = usable_at


class AssessmentRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.source = ScheduledSignal()
        self.runtime = SessionRuntime("eeg-test", bus=Mock())
        self.runtime.source = SimpleNamespace(kind="sim", device="test", srate=250)
        self.runtime.engine = WindowEngine(self.source, 250, device="test")
        self.runtime._sleep = Mock()
        self.runtime._progress = Mock()
        self.runtime._publish_window = Mock()
        self.runtime._persist_summary_metrics = Mock()

    def test_baseline_resumes_after_poor_startup_and_dropout(self):
        self.source.schedule(lambda index: index > 10 and not 24 <= index <= 27)
        baseline, steps = self.runtime._collect_assessment_baseline()
        state = self.runtime.assessment_eeg_collection["baseline_open"]
        self.assertTrue(baseline.valid)
        self.assertTrue(state["completed"])
        self.assertGreater(state["elapsed_seconds"], state["target_seconds"])
        self.assertLess(state["elapsed_seconds"], state["max_seconds"])
        self.assertGreaterEqual(state["valid_seconds"], state["target_seconds"])
        self.assertEqual(sum(not step.usable for step in steps), 4)
        self.assertEqual(state["attempted_windows"], len(steps))
        self.assertGreater(state["observed_windows"], len(steps))
        self.assertEqual(baseline.quality["usable_seconds"], state["valid_seconds"])

    def test_monitor_cap_with_some_usable_windows_retains_behavior_and_scales(self):
        baseline, _ = self.runtime._collect_assessment_baseline()
        self.assertTrue(baseline.valid)
        self.source.schedule(lambda index: index > 53)
        monitor = self.runtime._run_monitor()
        state = self.runtime.assessment_eeg_collection["monitor"]
        self.assertEqual(state["elapsed_seconds"], 120)
        self.assertEqual(state["status_reason"], "insufficient_effective_time")
        self.assertGreater(state["valid_seconds"], 0)
        self.assertGreater(state["shortfall_seconds"], 0)
        self.assertEqual(monitor["quality"]["valid_ratio"], 1)
        self.assertEqual(monitor["summary"], {})
        scale = score_scale("SAS", [2] * 20)
        assessment = self.runtime._run_assessment(
            baseline, {"SAS": scale}, {"pvt": {"valid": True, "lapse_rate": 0}},
            monitor["summary"], monitor["quality"],
        )
        evidence = {item.code: item for item in assessment.evidence}
        self.assertFalse(evidence["eeg_unavailable"].available)
        self.assertIn("目标 42 秒", evidence["eeg_unavailable"].summary)
        self.assertTrue(evidence["sas_standard_score"].available)
        self.assertTrue(evidence["pvt_alertness"].available)

    def test_monitor_continues_to_limit_when_elapsed_good_time_is_high_but_ratio_is_low(self):
        self.runtime._collect_assessment_baseline()
        self.source.schedule(lambda index: index % 5 in (2, 3))
        monitor = self.runtime._run_monitor()
        state = self.runtime.assessment_eeg_collection["monitor"]
        self.assertTrue(state["target_reached"])
        self.assertFalse(state["completed"])
        self.assertEqual(state["elapsed_seconds"], 120)
        self.assertEqual(state["shortfall_seconds"], 0)
        self.assertEqual(state["status_reason"], "low_valid_ratio")
        self.assertEqual(monitor["summary"], {})

    def test_live_scheduler_rejects_repeated_samples_and_publishes_both_confirmed_startup_windows(self):
        elapsed = [0.0]
        signal = self.source

        class BufferedSignal:
            def assessment_timing_anchor(self):
                return self.timing(0)

            def analysis_window(self, state, *, artifact, assessment_timing):
                sample_time = 6 if 6 <= elapsed[0] <= 10 else elapsed[0]
                return signal.window(state, artifact=artifact), {
                    "sample_timing": self.timing(sample_time),
                }

            @staticmethod
            def timing(second):
                end = round(second * 250)
                return {"buffer_id": 1, "connection_generation": 1, "sample_end": end,
                        "received_at": second, "latest_sample_end": end}

        live_source = BufferedSignal()
        self.runtime.source = SimpleNamespace(kind="lsl", device="test", srate=250, engine=live_source)
        self.runtime.engine = WindowEngine(live_source, 250, device="test")
        self.runtime._sleep_real = lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds)
        with patch("ningsi_studio.core.runtime.time.monotonic", new=lambda: elapsed[0]):
            baseline, recorded = self.runtime._collect_assessment_baseline()
        state = self.runtime.assessment_eeg_collection["baseline_open"]
        self.assertTrue(baseline.valid)
        self.assertEqual(state["repeated_windows"], 2)
        self.assertEqual(state["valid_seconds"], 60)
        self.assertEqual(state["elapsed_seconds"], 62)
        displayed = {call.args[0].index: call.args[0].usable
                     for call in self.runtime._publish_window.call_args_list}
        self.assertTrue(displayed[2] and displayed[3])
        self.assertFalse(displayed[4] or displayed[5])
        self.assertEqual(sum(displayed.values()), sum(step.usable for step in recorded))

    def test_disconnected_closed_baseline_reads_without_waiting_for_each_buffer(self):
        source = Mock()
        source.analysis_window.return_value = np.zeros((2, 0)), {}
        self.runtime.source = SimpleNamespace(kind="lsl", device="test", srate=250)
        self.runtime.engine = WindowEngine(source, 250, device="test")
        elapsed = [0.0]

        def advance(seconds):
            elapsed[0] += seconds

        self.runtime._sleep_real = advance
        with patch("ningsi_studio.core.runtime.time.monotonic", new=lambda: elapsed[0]):
            baseline, _ = self.runtime._collect_baseline("eyes_closed", 120, "baseline_closed")
        self.assertFalse(baseline.valid)
        self.assertEqual(elapsed[0], 120)
        self.assertEqual(source.analysis_window.call_count, 60)
        self.assertTrue(all(call.kwargs["wait"] is False for call in source.analysis_window.call_args_list))


class AssessmentSampleClockTests(unittest.TestCase):
    @staticmethod
    def timing(sample, received, *, latest=None, generation=1):
        return {"buffer_id": 1, "connection_generation": generation, "sample_end": sample,
                "received_at": received, "latest_sample_end": sample if latest is None else latest}

    def test_clock_credits_only_new_samples_and_waits_for_new_connection(self):
        clock = AssessmentSampleClock(250, 100, self.timing(1000, 99.99))
        self.assertEqual(clock.end_of(self.timing(1500, 102)), 2)
        self.assertEqual(clock.end_of(self.timing(1500, 102)), 2)
        self.assertEqual(clock.end_of(self.timing(2000, 106)), 4)
        self.assertIsNone(clock.end_of(self.timing(500, 108, generation=2)))
        self.assertEqual(clock.end_of(self.timing(1500, 112, generation=2)), 12)

    def test_aligned_research_window_uses_its_older_ending_index(self):
        clock = AssessmentSampleClock(250, 100, self.timing(1000, 100))
        self.assertEqual(clock.end_of(self.timing(2000, 106, latest=2500)), 4)
        self.assertEqual(clock.end_of(self.timing(2500, 108, latest=3000)), 6)
