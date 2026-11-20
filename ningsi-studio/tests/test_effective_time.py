"""Virtual-clock acceptance for bounded EEG replacement time (no long sleeps)."""
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
from ningsi import config
from ningsi.signal.baseline import build_baseline
from ningsi.signal.window import analyze_window
from ningsi_studio.core import phases, ten_min
from ningsi_studio.core.effective_time import DisplayClock, collect_baseline, run, training_round
from ningsi_studio.core.runtime import SessionCancelled, SessionRuntime
from ningsi_studio.domain.indicators import WindowStep
from ningsi_studio.domain.training import StudioTraining
from tests.helpers import StudioTestCase


@contextmanager
def patched_clock(clock):
    with patch("ningsi_studio.core.effective_time.time.monotonic", clock.monotonic), \
         patch("ningsi_studio.core.effective_time.time.perf_counter", clock.monotonic):
        yield


class VirtualClock:
    def __init__(self):
        self.now = 0.0
    def monotonic(self):
        return self.now
    def sleep(self, seconds):
        self.now += max(0, seconds)


class Bus:
    def __init__(self):
        self.events = []
    def publish(self, name, payload):
        self.events.append((name, payload))


class FakeRuntime:
    latest_analysis_window = None

    def __init__(self, clock, good, bad, pattern=None, cancel_at=None, latency=0):
        self.clock, self.scale = clock, 1
        self.paused_seconds = 0.0
        self.source = SimpleNamespace(kind="sim", device="fixture", srate=250)
        self.subject = {"public_id": "p01"}
        self.bus, self.payloads, self.persisted = Bus(), {}, []
        self.phase, self.pattern = None, pattern or (lambda phase, index: True)
        self.cancel_at, self.latency, self.index = cancel_at, latency, 0
        self.good, self.bad, self.diagnostic_windows = good, bad, []
        self.engine = SimpleNamespace(step=self.step, t=0, baseline=None, summarize=lambda _: {})
        self.artifacts, self.summary = {}, {}
        self.effective_time_result = {"phase_results": {}, "timings": {}, "segments": []}
    def protocol_now(self):
        return time.monotonic() - self.paused_seconds
    def precise_now(self):
        return time.perf_counter() - self.paused_seconds
    def step(self, state):
        self.clock.sleep(self.latency)
        self.index += 1
        self.engine.t += 2
        usable = self.pattern(self.phase, self.index)
        window = replace(self.good if usable else self.bad, t_end=self.engine.t)
        return WindowStep(self.index, self.engine.t, state, window=window,
                          scores={"focus": .75} if usable else {},
                          graded={"grade": "A" if usable else "C", "channels": []})
    def _sleep_real(self, seconds):
        if self.cancel_at is not None and self.clock.now+seconds >= self.cancel_at:
            self.clock.now = self.cancel_at
            raise SessionCancelled()
        self.clock.sleep(seconds)
    _sleep = _sleep_real
    def _check_cancel(self):
        if self.cancel_at is not None and self.clock.now >= self.cancel_at:
            raise SessionCancelled()
    @contextmanager
    def _phase(self, key):
        self.phase = key
        yield {"run_id": key}
    def _publish_window(self, *args, **kwargs):
        pass
    def _progress(self, *args, **kwargs):
        pass
    def _merge_run_payload(self, run_id, payload):
        self.payloads.setdefault(run_id, {}).update(payload)
    def _record_diagnostic(self, *args):
        pass
    def _record_heatmap(self, *args):
        return {}
    def current_affect(self):
        return None
    def _persist_training_segment(self, segment):
        self.persisted.append(segment)
    def _run_pvt(self, auto, *, practice=False):
        self.pvt_calls = getattr(self, "pvt_calls", []) + [(practice, self.clock.now)]
        return {"trials": 5 if practice else 30}
    def _run_assessment(self, *args):
        return SimpleNamespace(as_dict=lambda: {})
    def _write_report(self, *args):
        self.training_report = args[-2]
        return {}
    def _finish(self, status, error, summary):
        self.status, self.summary = status, summary


class EffectiveTimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        t = np.arange(1000)/250
        raw = 12*np.sin(2*np.pi*6*t)+10*np.sin(2*np.pi*10*t)+7*np.sin(2*np.pi*20*t)
        cls.good = analyze_window(raw[None, :], 250, t_end=4)
        cls.bad = analyze_window(np.zeros((1, 1000)), 250, t_end=4)
        cls.base = build_baseline([replace(cls.good, t_end=4+i*2) for i in range(30)], device="fixture")
        assert cls.good.usable and cls.base.valid

    def runtime(self, clock, **kwargs):
        return FakeRuntime(clock, self.good, self.bad, **kwargs)

    def test_display_does_not_credit_future_or_refresh_expiry_on_poll(self):
        timer = DisplayClock()
        timer.display(0, usable=True, mode="eeg")
        self.assertEqual(timer.snapshot(0)["valid_seconds"], 0)
        for at in (1, 2, 3, 4, 20):
            timer.settle(at)
        self.assertEqual(timer.valid, 2)
        timer.display(20, usable=False, mode="guided")
        timer.settle(22)
        self.assertEqual(timer.valid, 2)
        self.assertEqual(timer.mode_seconds, {"eeg": 2, "guided": 2})

    def test_baseline_overlap_union_and_original_minimum_are_separate(self):
        clock = VirtualClock()
        r = self.runtime(clock)
        with patched_clock(clock):
            steps, base = collect_baseline(r, "baseline_open", "b")
        state = r.payloads["b"]["effective_time"]
        self.assertTrue(base.valid)
        self.assertEqual(state["valid_seconds"], 60)
        self.assertEqual(state["elapsed_seconds"], 60)
        self.assertEqual(len(steps), 29)
        self.assertEqual(state["minimum_baseline_seconds"], 30)
        self.assertTrue(state["completed"])

    def test_bad_baseline_windows_remain_in_denominator_even_after_goal(self):
        clock = VirtualClock()
        r = self.runtime(clock, pattern=lambda phase, index: index % 3 == 0)
        with patched_clock(clock):
            _, base = collect_baseline(r, "baseline_open", "b")
        state = r.payloads["b"]["effective_time"]
        self.assertGreaterEqual(state["valid_seconds"], 60)
        self.assertLess(base.quality["valid_ratio"], .6)
        self.assertFalse(base.valid)
        self.assertFalse(state["completed"])
        self.assertEqual(state["completion_reason"], "max_duration")
        self.assertEqual(state["elapsed_seconds"], 180)

    def test_all_good_training_waits_last_interval_and_preserves_gate(self):
        clock = VirtualClock()
        r = self.runtime(clock)
        trainer = StudioTraining("p01", "01", "001", self.base, "ten_min")
        with patched_clock(clock):
            segment = training_round(r, trainer, self.base, 0, "t")
        self.assertEqual(segment["duration_sec"], 54)
        self.assertEqual(segment["stats"]["valid_seconds"], 50)
        self.assertTrue(segment["effective_time"]["completed"])
        feedback = [p for name,p in r.bus.events if name == "feedback"]
        self.assertEqual([p["feedback_mode"] for p in feedback[:3]], ["guided", "guided", "eeg"])
        self.assertEqual([p["presentation_mode"] for p in feedback[:3]], ["measured_focus", "measured_focus", "eeg"])
        self.assertEqual([p["measured_focus_index"]["value"] for p in feedback[:3]], [.75]*3)
        self.assertTrue(all(p["score"] is None and p["target"] is None and not p["usable"] for p in feedback[:2]))
        self.assertEqual(segment["measured_focus_updates"], len(feedback))
        self.assertEqual(segment["measured_focus_only_updates"], 2)
        self.assertEqual(feedback[2]["effective_time"]["valid_seconds"], 0)
        self.assertEqual(feedback[-1]["effective_time"]["valid_seconds"], 48)
        self.assertEqual(segment["stats"]["on_target_valid_ratio"], 1)
        self.assertAlmostEqual(segment["stats"]["valid_time_ratio"], 50/54, places=4)

    def test_interruption_pauses_progress_breaks_hold_and_recovers_after_three(self):
        clock = VirtualClock()
        r = self.runtime(clock, pattern=lambda phase, index: index not in (6, 7))
        original = r.engine.step
        def changed_score_after_bad_windows(state):
            step = original(state)
            if step.index >= 8 and step.usable:
                step.scores["focus"] = .35
            return step
        r.engine.step = changed_score_after_bad_windows
        trainer = StudioTraining("p01", "01", "001", self.base, "ten_min")
        with patched_clock(clock):
            segment = training_round(r, trainer, self.base, 0, "t")
        feedback = [p for name,p in r.bus.events if name == "feedback"]
        self.assertEqual(segment["stats"]["valid_seconds"], 50)
        self.assertEqual(segment["duration_sec"], 62)
        self.assertEqual(feedback[5]["effective_time"]["valid_seconds"], feedback[8]["effective_time"]["valid_seconds"])
        self.assertEqual([p["feedback_mode"] for p in feedback[5:10]], ["guided", "guided", "guided", "guided", "eeg"])
        self.assertIsNone(feedback[5]["score"])
        self.assertEqual(feedback[5]["current_hold_sec"], 0)
        self.assertEqual(feedback[9]["score"], .35, "Bad windows must reset smoothing before feedback resumes")
        self.assertLess(segment["stats"]["on_target_ratio"], segment["stats"]["on_target_valid_ratio"])

    def test_descriptive_bands_persist_without_focus_rewards_or_effective_time(self):
        clock = VirtualClock()
        r = self.runtime(clock, pattern=lambda *_: False)
        original = r.engine.step
        ts = np.arange(1000) / 250
        lead = 15 * np.sin(2 * np.pi * 10 * ts) + 8 * np.sin(2 * np.pi * 6 * ts)

        def partially_clean(state):
            step = original(state)
            step.raw = np.stack((np.zeros_like(lead), lead))
            return step

        r.engine.step = partially_clean
        trainer = StudioTraining("p01", "01", "001", self.base, "ten_min")
        with patched_clock(clock):
            segment = training_round(r, trainer, self.base, 0, "t")
        feedback = [p for name, p in r.bus.events if name == "feedback"]
        self.assertTrue(all(p["presentation_mode"] == "local_eeg" for p in feedback))
        self.assertTrue(all(p["score"] is None and p["target"] is None and not p["on_target"] for p in feedback))
        self.assertEqual(segment["stats"]["valid_seconds"], 0.)
        self.assertEqual(segment["target_after"], segment["target"])
        self.assertFalse(segment["effective_time"]["completed"])
        self.assertGreater(segment["local_eeg_feedback"]["observations_count"], 0)
        self.assertEqual(set(segment["local_eeg_feedback"]["support_seconds_by_channel"]), {"1"})

    def test_all_bad_finishes_bounded_insufficient_and_pvt_stays_fixed(self):
        clock = VirtualClock()
        r = self.runtime(clock, pattern=lambda phase, index: False)
        with patched_clock(clock):
            run(r, True)
        result = r.summary["effective_time"]
        self.assertEqual(clock.now, 840)
        self.assertEqual(r.status, "done")
        self.assertEqual(result["collection_status"], "insufficient")
        self.assertFalse(result["acquisition_complete"])
        self.assertEqual(result["timings"]["practice"], 30)
        self.assertEqual(result["timings"]["pvt"], 180)
        self.assertEqual(len(r.persisted), 4)
        for segment in r.persisted:
            self.assertEqual(segment["duration_sec"], 50)
            self.assertEqual(segment["stats"]["valid_seconds"], 0)
            self.assertIsNone(segment["mean_score"])
            self.assertIsNone(segment["on_target_valid_ratio"])
            self.assertFalse(segment["effective_time"]["completed"])
            self.assertEqual(segment["effective_time"]["completion_reason"], "baseline_invalid")

    def test_normal_run_and_previous_protocol_definitions_do_not_change(self):
        clock = VirtualClock()
        r = self.runtime(clock)
        with patched_clock(clock):
            run(r, True)
        self.assertEqual(r.summary["effective_time"]["collection_status"], "complete")
        self.assertEqual(clock.now, 616)
        self.assertEqual(sum(phases.TEN_DURATIONS.values()), 600)
        self.assertEqual(sum(p["duration_sec"] for p in phases.as_list("ten_min")), 600)
        self.assertEqual(phases.training_plan("ten_min"), phases.training_plan("effective_time"))
        self.assertEqual(phases.profile_dict("short")["pvt_duration_sec"], 120)
        self.assertEqual(phases.profile_dict("effective_time")["pvt_duration_sec"], 180)
        self.assertEqual(r.training_report["effective_time"]["valid_seconds"], 200)

    def test_demo_pauses_do_not_consume_phase_budgets_or_credit_feedback(self):
        for protocol in ("ten_min", "effective_time"):
            with self.subTest(protocol=protocol):
                clock = VirtualClock()
                r = self.runtime(clock)
                r.scale = .5
                original_sleep = r._sleep_real
                paused_phases = set()
                def sleep_with_presentation_pause(seconds):
                    if r.phase not in paused_phases and seconds > 0:
                        # An extended presenter pause spans more than a whole round.
                        clock.sleep(120)
                        r.paused_seconds += 120
                        paused_phases.add(r.phase)
                    original_sleep(seconds)
                r._sleep_real = sleep_with_presentation_pause
                r._sleep = lambda seconds: r._sleep_real(seconds * r.scale)
                with patched_clock(clock):
                    if protocol == "ten_min":
                        ten_min.run(r, True)
                        result = r.ten_min_result
                        self.assertEqual(result["timings"], {k: v for k, v in phases.TEN_DURATIONS.items() if v})
                        for segment in result["segments"]:
                            self.assertEqual(segment["mode_seconds"], {"guided": 4, "eeg": 46})
                    else:
                        run(r, True)
                        result = r.summary["effective_time"]
                        self.assertTrue(result["acquisition_complete"])
                        for phase in ("baseline_open", "baseline_after", "training"):
                            state = result["phase_results"][phase]
                            self.assertEqual(state["actual_elapsed_seconds"], state["elapsed_seconds"] * r.scale + 120)
                        for segment in result["segments"]:
                            self.assertEqual(segment["stats"]["valid_seconds"], 50)
                            self.assertEqual(segment["duration_sec"], 54)
                        self.assertEqual(result["actual_elapsed_seconds"], clock.now)
                self.assertEqual(paused_phases, {"qc", "baseline_open", "practice", "pvt", "training", "baseline_after"})
                self.assertEqual(result["elapsed_sec"], clock.now)
                expected_protocol = 600 if protocol == "ten_min" else 616
                self.assertEqual(result["protocol_elapsed_seconds"], expected_protocol)
                self.assertEqual(clock.now, expected_protocol * r.scale + r.paused_seconds)

    def test_cancel_keeps_partial_training_and_never_completes_target(self):
        clock = VirtualClock()
        r = self.runtime(clock, cancel_at=13)
        trainer = StudioTraining("p01", "01", "001", self.base, "ten_min")
        with patched_clock(clock):
            with self.assertRaises(SessionCancelled):
                training_round(r, trainer, self.base, 0, "t")
        self.assertEqual(len(r.persisted), 1)
        partial = r.persisted[0]
        self.assertEqual(partial["duration_sec"], 13)
        self.assertEqual(partial["stats"]["valid_seconds"], 9)
        self.assertEqual(partial["effective_time"]["completion_reason"], "cancelled")
        self.assertFalse(partial["effective_time"]["completed"])

    def test_valid_baseline_but_all_bad_training_expires_at_full_cap(self):
        clock = VirtualClock()
        r = self.runtime(clock, pattern=lambda phase, index: False)
        trainer = StudioTraining("p01", "01", "001", self.base, "ten_min")
        with patched_clock(clock):
            segment = training_round(r, trainer, self.base, 0, "t")
        self.assertEqual(segment["duration_sec"], 100)
        self.assertEqual(segment["effective_time"]["completion_reason"], "max_duration")
        self.assertEqual(segment["stats"]["valid_seconds"], 0)

    def test_cancelled_baseline_retains_partial_union_and_reason(self):
        clock = VirtualClock()
        r = self.runtime(clock, cancel_at=17)
        with patched_clock(clock):
            with self.assertRaises(SessionCancelled):
                collect_baseline(r, "baseline_open", "b")
        state = r.payloads["b"]["effective_time"]
        self.assertEqual(state["valid_seconds"], 16)
        self.assertEqual(state["elapsed_seconds"], 17)
        self.assertEqual(state["completion_reason"], "cancelled")
        self.assertFalse(state["completed"])

    def test_simulation_protocol_clock_is_separate_from_actual_wall(self):
        clock = VirtualClock()
        r = self.runtime(clock)
        r.scale = .05
        trainer = StudioTraining("p01", "01", "001", self.base, "ten_min")
        with patched_clock(clock):
            segment = training_round(r, trainer, self.base, 0, "t")
        state = segment["effective_time"]
        self.assertEqual(state["time_basis"], "protocol_seconds")
        self.assertEqual(state["time_scale"], .05)
        self.assertAlmostEqual(state["elapsed_seconds"], 54)
        self.assertAlmostEqual(state["actual_elapsed_seconds"], 2.7)
        self.assertAlmostEqual(state["valid_seconds"], 50)

    def test_analysis_gap_is_not_interpolated_and_wall_cap_remains_visible(self):
        clock = VirtualClock()
        r = self.runtime(clock, latency=5)
        trainer = StudioTraining("p01", "01", "001", self.base, "ten_min")
        with patched_clock(clock):
            segment = training_round(r, trainer, self.base, 0, "t")
        self.assertLess(segment["stats"]["valid_seconds"], 50)
        self.assertEqual(segment["effective_time"]["completion_reason"], "max_duration")
        self.assertGreater(segment["unobserved_seconds"], 0)
        self.assertGreaterEqual(segment["duration_sec"], 100)

    def test_analysis_is_prepared_before_expiry_without_inventing_support_time(self):
        clock = VirtualClock()
        r = self.runtime(clock, latency=.5)
        trainer = StudioTraining("p01", "01", "001", self.base, "ten_min")
        target_before = trainer.target
        with patched_clock(clock):
            segment = training_round(r, trainer, self.base, 0, "t")
        self.assertAlmostEqual(segment["duration_sec"], 54.4)
        self.assertEqual(segment["stats"]["valid_seconds"], 50)
        self.assertEqual(segment["effective_time"]["valid_seconds"], 50)
        self.assertTrue(segment["effective_time"]["timing_consistent"])
        self.assertTrue(segment["effective_time"]["completed"])
        self.assertGreaterEqual(segment["stats"]["max_hold_sec"], trainer.inner.hold_sec)
        self.assertTrue(segment["achieved"])
        self.assertGreater(segment["target_after"], target_before)
        self.assertEqual(segment["excluded_windows"], 2, "Only the two initial gate windows were unusable")
        self.assertEqual(segment["support_gap_count"], 0)
        self.assertEqual(segment["support_gap_interruptions"], [])
        self.assertEqual(segment["feedback_schedule"]["support_cap_seconds"], config.STEP_SEC)
        self.assertAlmostEqual(segment["feedback_schedule"]["feedback_interval_seconds_mean"], 1.95)
        self.assertAlmostEqual(segment["feedback_schedule"]["actual_scheduling_margin_seconds"], .05)
        feedback = [p for name,p in r.bus.events if name == "feedback"]
        self.assertTrue(any(p["current_hold_sec"] >= 6 for p in feedback))

    def test_late_wakeup_still_breaks_hold_without_adding_a_rejected_eeg_window(self):
        clock = VirtualClock()
        r = self.runtime(clock, latency=.5)
        late = False
        def delayed_sleep(seconds):
            nonlocal late
            if r.index == 6 and not late:
                late = True
                clock.sleep(.5)
            clock.sleep(seconds)
        r._sleep_real = delayed_sleep
        trainer = StudioTraining("p01", "01", "001", self.base, "ten_min")
        with patched_clock(clock):
            segment = training_round(r, trainer, self.base, 0, "t")
        feedback = [p for name,p in r.bus.events if name == "feedback"]
        self.assertEqual(segment["support_gap_count"], 1)
        self.assertEqual(segment["excluded_windows"], 2)
        self.assertEqual(feedback[6]["current_hold_sec"], 0)
        self.assertAlmostEqual(segment["support_gap_interruptions"][0],
                               feedback[5]["effective_time"]["elapsed_seconds"]+config.STEP_SEC)
        self.assertEqual(segment["stats"]["valid_seconds"], 50)
        self.assertTrue(segment["effective_time"]["timing_consistent"])
        self.assertGreater(segment["unobserved_seconds"], .4)

    def test_sudden_long_analysis_still_expires_the_previous_score(self):
        clock = VirtualClock()
        r = self.runtime(clock, latency=.5)
        original = r.engine.step
        def slow_once(state):
            if r.index == 6:
                clock.sleep(3)
            return original(state)
        r.engine.step = slow_once
        trainer = StudioTraining("p01", "01", "001", self.base, "ten_min")
        with patched_clock(clock):
            segment = training_round(r, trainer, self.base, 0, "t")
        feedback = [p for name,p in r.bus.events if name == "feedback"]
        self.assertEqual(segment["support_gap_count"], 1)
        self.assertEqual(segment["excluded_windows"], 2)
        self.assertEqual(feedback[6]["current_hold_sec"], 0)
        self.assertGreater(segment["unobserved_seconds"], 2.9)
        self.assertEqual(segment["stats"]["valid_seconds"], 50)
        self.assertTrue(segment["effective_time"]["timing_consistent"])

    def test_feedback_uses_high_resolution_clock_when_tick_clock_is_coarse(self):
        clock = VirtualClock()
        r = self.runtime(clock, latency=.001)
        r.scale = .01
        trainer = StudioTraining("p01", "01", "001", self.base, "ten_min")
        # Quantized GetTickCount64 would give the initial guided and accepted
        # updates identical times. The entire feedback ledger uses QPC instead.
        with patch("ningsi_studio.core.effective_time.time.monotonic", lambda: int(clock.now/.015625)*.015625), \
             patch("ningsi_studio.core.effective_time.time.perf_counter", clock.monotonic):
            segment = training_round(r, trainer, self.base, 0, "t")
        self.assertTrue(segment["effective_time"]["timing_consistent"])
        self.assertAlmostEqual(segment["effective_time"]["valid_seconds"], segment["stats"]["valid_seconds"])
        self.assertTrue(segment["effective_time"]["completed"])
        self.assertGreaterEqual(segment["stats"]["max_hold_sec"], 6)
        self.assertEqual(segment["feedback_schedule"]["clock"], "perf_counter")

    def test_cancellation_during_expired_feedback_tail_retains_a_gap_but_not_a_bad_window(self):
        clock = VirtualClock()
        r = self.runtime(clock, latency=.5)
        original = r.engine.step
        def cancelled_analysis(state):
            if r.index == 3:
                clock.sleep(3)
                raise SessionCancelled()
            return original(state)
        r.engine.step = cancelled_analysis
        trainer = StudioTraining("p01", "01", "001", self.base, "ten_min")
        with patched_clock(clock):
            with self.assertRaises(SessionCancelled):
                training_round(r, trainer, self.base, 0, "t")
        partial = r.persisted[0]
        self.assertAlmostEqual(partial["duration_sec"], 8.85)
        self.assertEqual(partial["stats"]["valid_seconds"], 2)
        self.assertEqual(partial["support_gap_interruptions"], [6.4])
        self.assertEqual(partial["excluded_windows"], 2)
        self.assertEqual(partial["stats"]["current_hold_sec"], 0)
        self.assertTrue(partial["effective_time"]["timing_consistent"])
        self.assertEqual(partial["effective_time"]["completion_reason"], "cancelled")

    def test_final_score_before_cap_cannot_support_past_cap(self):
        clock = VirtualClock()
        r = self.runtime(clock, pattern=lambda phase, index: index >= 26)
        original = r.engine.step
        def first_offset(state):
            if r.index == 0:
                clock.sleep(1)
            return original(state)
        r.engine.step = first_offset
        def late_wakeup(seconds):
            clock.sleep(seconds + 2 if clock.now > 99 else seconds)
        r._sleep_real = late_wakeup
        trainer = StudioTraining("p01", "01", "001", self.base, "ten_min")
        with patched_clock(clock):
            segment = training_round(r, trainer, self.base, 0, "t")
        self.assertGreater(segment["samples"][-1][0], 99)
        self.assertEqual(segment["duration_sec"], 102)
        self.assertIn(100, segment["interruptions"])
        self.assertAlmostEqual(segment["stats"]["valid_seconds"], 100-segment["samples"][0][0])
        self.assertEqual(segment["effective_time"]["valid_seconds"], segment["stats"]["valid_seconds"])
        self.assertFalse(segment["effective_time"]["completed"])
        self.assertEqual(segment["excluded_windows"], 27)
        self.assertEqual(segment["support_gap_interruptions"], [100])
        self.assertTrue(segment["effective_time"]["timing_consistent"])

    def test_new_pvt_uses_same_phase_matching_and_fixed_deadline(self):
        # Exercise real _run_pvt with a virtual clock, real onset generator, and
        # answers arriving inside the response window (no 210-second wait).
        for protocol in ("ten_min", "effective_time"):
            for practice, duration in ((True, 30), (False, 180)):
                clock = VirtualClock()
                runtime = SessionRuntime.__new__(SessionRuntime)
                runtime.uuid, runtime.protocol = "test-clock", protocol
                runtime.profile = phases.profile_dict(protocol)
                runtime.session = {"time_scale": 1}
                runtime.bus, runtime.cancel_event = Bus(), threading.Event()
                runtime._sleep_real = clock.sleep
                runtime._persist_behavior = lambda *args: None
                def response(name, payload, *, auto, fallback, window):
                    self.assertIn(payload["phase"], ("practice", "main"))
                    self.assertLessEqual(clock.now+window, duration+1e-6)
                    clock.sleep(min(.2, window))
                    return {"responded": True, "rt": .2}
                runtime._wait_trial = response
                with patch("ningsi_studio.core.runtime.time.monotonic", clock.monotonic):
                    result = runtime._run_pvt(False, practice=practice)
                self.assertLessEqual(clock.now, duration)
                self.assertGreater(result["trials"], 0)

    def test_new_pvt_rejects_late_practice_answer_with_same_index(self):
        runtime = SessionRuntime("a"*32)
        runtime.protocol = "effective_time"
        def accept_check(name, *, timeout, accept):
            self.assertFalse(accept({"phase": "practice", "index": 0}))
            self.assertFalse(accept({"phase": "main", "index": 1}))
            self.assertTrue(accept({"phase": "main", "index": 0}))
            return {"phase": "main", "index": 0, "responded": True, "rt": .3}
        runtime.wait_for_input = accept_check
        answer = runtime._wait_trial("pvt", {"task": "pvt", "phase": "main", "index": 0},
                                     auto=False, fallback={}, window=1)
        self.assertEqual(answer["phase"], "main")


class EffectiveTimeApiTests(StudioTestCase):
    port_base = 19024

    def test_protocol_config_live_snapshot_and_persisted_report(self):
        cfg = self.json_body(self.get("/api/config"), 200)
        profiles = cfg.get("protocols") or cfg.get("profiles")
        self.assertIsNotNone(profiles)
        profile = next(row for row in profiles if row["key"] == "effective_time")
        self.assertEqual(profile["max_total_sec"], 1040)
        created = self.json_body(self.post("/api/sessions", {
            "participant": "ef01", "device": "sim-bsense", "protocol": "effective_time",
            "time_scale": .01, "create_subject": True}), 201)
        uuid = created["session"]["uuid"]
        self.assertEqual(created["protocol"], "effective_time")
        saw_snapshot = False
        deadline = time.monotonic()+30
        while time.monotonic() < deadline:
            detail = self.json_body(self.get("/api/sessions/"+uuid), 200)
            feedback = detail.get("runtime", {}).get("latest_feedback")
            if feedback:
                live = self.json_body(self.get("/api/sessions/"+uuid+"/training"), 200)
                self.assertIn("segment_stats", live["current_feedback"])
                self.assertIn("effective_time", live["current_feedback"])
                saw_snapshot = True
            if detail["status"] in ("done", "failed", "cancelled"):
                break
            time.sleep(.03)
        self.assertEqual(detail["status"], "done", detail.get("error"))
        self.assertTrue(saw_snapshot)
        report = self.json_body(self.get("/api/sessions/"+uuid+"/report"), 200)
        effective = report["extras"]["effective_time"]
        self.assertIn(effective["collection_status"], ("complete", "insufficient"))
        self.assertEqual(effective["max_acquisition_seconds"], 1040)
        self.assertEqual(set(effective["phase_results"]), {"baseline_open", "training", "baseline_after"})
        self.assertEqual(report["extras"]["protocol"], "effective_time")
        self.assertEqual(report["indicators"]["summary"], {})
        self.assertIn("result_overview", detail)
        training = self.json_body(self.get("/api/sessions/"+uuid+"/training"), 200)
        self.assertEqual(training["summary"]["mode"], "effective_time")
        self.assertEqual(len(training["segments"]), 4)
        for segment in training["segments"]:
            self.assertEqual(segment["effective_time"]["time_scale"], .01)
            stats = segment["stats"]
            self.assertAlmostEqual(stats["valid_time_ratio"], stats["valid_seconds"]/segment["duration_sec"], places=4)
            self.assertEqual(segment["effective_time"]["valid_seconds"], stats["valid_seconds"])


if __name__ == "__main__":
    unittest.main()
