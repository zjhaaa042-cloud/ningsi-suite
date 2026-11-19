from collections import Counter
from dataclasses import replace
import unittest
from unittest.mock import Mock

from .test_support import MemoryStore
from ningsi_capture.protocol import ARTIFACT_CALIBRATION_ACTIONS, ProtocolConfig, TaskRunner
from ningsi.models.capture_protocols import SHORT_PROTOCOL


class ProtocolTests(unittest.TestCase):
    def runner(self, config=None):
        store = MemoryStore()
        runner = TaskRunner(store, lambda view: store.clock.now(), config or ProtocolConfig(), store.clock)
        runner.start()
        return store, runner

    def reach_sart(self):
        store, runner = self.runner()
        for _ in range(2):
            store.clock.seconds = runner.deadline
            runner.tick()
        self.assertEqual(runner.state, "sart")
        return store, runner

    def test_full_protocol_trial_counts_probes_and_no_feedback(self):
        store, runner = self.runner()
        iterations = 0
        while runner.running:
            iterations += 1
            self.assertLess(iterations, 100000)
            if runner.state == "sart" and not runner.trial["responded"] and not runner.trial["nogo"]:
                store.clock.advance(.25)
                runner.key("space")
            elif runner.state == "pvt_stimulus":
                store.clock.advance(.3)
                runner.key("space")
            elif runner.state == "probe":
                store.clock.advance(.3)
                runner.key("3")
            else:
                store.clock.advance(.1)
                runner.tick()
        probes = [e for e in store.events if e["kind"] == "probe_onset" and not e["payload"]["practice"]]
        sart = [e for e in store.events if e["kind"] == "trial_result" and e["payload"].get("task") == "sart"]
        self.assertTrue(runner.finished)
        self.assertEqual(len(probes), 32)
        self.assertEqual(len(sart), 4 * 180 + 12)
        self.assertEqual(sum(e["payload"]["nogo"] for e in sart if not e["payload"]["practice"]), 80)
        self.assertFalse(any("score" in e["payload"] for e in store.events))

    def test_quick_demo_runs_all_phases_and_records_formal_probes(self):
        store, runner = self.runner(ProtocolConfig.demo())
        iterations = 0
        while runner.running:
            iterations += 1
            self.assertLess(iterations, 5000)
            if runner.state == "probe":
                store.clock.advance(.2)
                runner.key("3")
            else:
                store.clock.advance(.1)
                runner.tick()
        phases = [e["payload"]["phase"] for e in store.events if e["kind"] == "phase_start"]
        self.assertIn("sart_practice", phases)
        self.assertIn("pvt_post", phases)
        self.assertTrue(runner.finished)
        self.assertEqual(sum(e["kind"] == "probe_onset" and not e["payload"]["practice"] for e in store.events), 2)

    def test_short_protocol_runs_three_complete_blocks_with_the_planned_timing(self):
        config = ProtocolConfig.short()
        store, runner = self.runner(config)
        start = store.clock.seconds
        while runner.running:
            if runner.state == "probe":
                store.clock.advance(7 if runner.phase.get("practice") else 3)
                runner.key("3")
            else:
                store.clock.seconds = runner.deadline
                runner.tick()
        self.assertTrue(runner.finished)
        self.assertEqual(config.name, SHORT_PROTOCOL)
        self.assertEqual((config.calibration_sec, config.holdout_sec, config.rest_sec), (120, 45, 45))
        self.assertEqual((config.sart_soa, config.probe_timeout, config.recovery_sec), (2.2, 8, 1))
        phases = [e["payload"]["phase"] for e in store.events if e["kind"] == "phase_start"]
        self.assertEqual(phases, ["baseline_calibration", "baseline_holdout", "sart_practice", "probe_practice",
                                  "sart_1", "rest_1", "sart_2", "rest_2", "sart_3"])
        trials = [e["payload"] for e in store.events if e["kind"] == "trial_result"]
        self.assertEqual(Counter(t["block"] for t in trials), {0: 24, 1: 115, 2: 115, 3: 115})
        self.assertEqual(Counter(t["block"] for t in trials if not t["practice"] and t["nogo"]), {1: 13, 2: 13, 3: 13})
        probes = [e["payload"] for e in store.events if e["kind"] == "probe_onset"]
        self.assertEqual(Counter(p["block"] for p in probes), {0: 1, 1: 5, 2: 5, 3: 5})
        self.assertEqual(len({p["probe_id"] for p in probes}), 16)
        self.assertAlmostEqual(store.clock.seconds - start, 1133.8)

    def test_session_calibration_presents_five_separate_cues_per_action_before_research(self):
        config = replace(ProtocolConfig.short(), personal_artifact_calibration=True)
        store = MemoryStore()
        precheck = Mock(side_effect=["幅度过大", None])
        runner = TaskRunner(store, lambda view: store.clock.now(), config, store.clock, research_precheck=precheck)
        runner.start()
        first = store.clock.seconds
        while runner.state != "artifact_calibration_wait":
            store.clock.seconds = runner.deadline
            runner.tick()
        self.assertEqual(store.clock.seconds - first, 300)
        cues = [e for e in store.events if e["kind"] == "artifact_calibration_cue"]
        self.assertEqual(len(cues), 30)
        offsets = [e["stamp"].perf - first for e in cues]
        self.assertEqual(offsets, list(range(60, 240, 6)))
        for index, event in enumerate(cues, 1):
            self.assertEqual(event["payload"]["cue_index"], index)
            self.assertIs(event["payload"]["observed_action"], False)
            self.assertEqual(event["payload"]["onset_basis"], "tk_idle_estimate")
        for repetition in range(1, 6):
            self.assertEqual({e["payload"]["action"] for e in cues if e["payload"]["repetition"] == repetition},
                             {action for action, _ in ARTIFACT_CALIBRATION_ACTIONS})
        names = [e["kind"] for e in store.events]
        self.assertEqual(names.count("artifact_calibration_start"), 1)
        self.assertEqual(names.count("artifact_calibration_complete"), 1)
        self.assertNotIn("protocol_start", names)
        store.clock.advance(60)
        runner.tick()
        self.assertEqual(precheck.call_count, 1)
        runner.key("return")
        original_start = next(e for e in store.events if e["kind"] == "protocol_start")
        self.assertEqual(original_start["stamp"].perf, first + 360)
        self.assertEqual(runner.phase["name"], "baseline_calibration")
        self.assertEqual(precheck.call_count, 2)

    def test_late_calibration_tick_preserves_recovery_and_abort_leaves_no_research_start(self):
        config = replace(ProtocolConfig.demo(), personal_artifact_calibration=True)
        store = MemoryStore()
        painted = []

        def render(view):
            store.clock.advance(.25)
            painted.append(store.clock.now())
            return painted[-1]

        runner = TaskRunner(store, render, config, store.clock)
        runner.start()
        store.clock.seconds = runner.deadline
        runner.tick()
        cue = next(e for e in store.events if e["kind"] == "artifact_calibration_cue")
        self.assertEqual(cue["stamp"], painted[-1])
        self.assertEqual(cue["stamp"].perf - cue["payload"]["requested_perf"], .25)
        store.clock.advance(15)
        runner.tick()
        self.assertEqual(runner.state, "artifact_reset")
        self.assertEqual(runner.deadline - painted[-1].perf, 4.)
        runner.abort("operator_stop")
        names = [e["kind"] for e in store.events]
        self.assertEqual(names.count("artifact_calibration_cue"), 1)
        self.assertIn("artifact_calibration_abort", names)
        self.assertNotIn("artifact_calibration_complete", names)
        self.assertNotIn("protocol_start", names)

    def test_invalid_practice_configuration_is_rejected_before_recording(self):
        with self.assertRaisesRegex(ValueError, "至少需要5"):
            replace(ProtocolConfig(), sart_practice_trials=4)

    def test_phase_remaining_uses_block_and_probe_clocks(self):
        config = replace(ProtocolConfig.demo(), sart_soa=1, probe_timeout=8, recovery_sec=2)
        store, runner = self.runner(config)
        while runner.phase["name"] != "sart_1":
            if runner.phase["kind"] == "practice_probe":
                self.assertIsNone(runner.phase_remaining(store.clock.seconds))
                runner.key("1")
            else:
                deadline = runner.deadline
                if runner.phase["kind"] == "pvt":
                    left = runner.phase_deadline - store.clock.seconds
                    self.assertEqual(runner.phase_remaining(store.clock.seconds), (left, left))
                    deadline = min(deadline, runner.phase_deadline)
                store.clock.seconds = deadline
                runner.tick()
        start = store.clock.seconds
        self.assertEqual(runner.phase_remaining(start), (38, 46))  # 36 trials, one 0–8s prompt, 2s recovery
        while runner.state == "sart":
            store.clock.seconds = runner.deadline
            runner.tick()
        self.assertEqual(runner.state, "probe")
        store.clock.advance(3)
        digits_left = len(runner.digits) - runner.trial_index
        self.assertEqual(runner.phase_remaining(store.clock.seconds), (digits_left + 2, digits_left + 7))
        runner.key("1")
        self.assertEqual(runner.state, "recovery")
        store.clock.advance(1)
        self.assertEqual(runner.phase_remaining(store.clock.seconds), (digits_left + 1, digits_left + 1))
        self.assertEqual(runner.phase_started.perf, start)

    def test_practice_probe_allows_reading_before_answer_and_is_marked_practice(self):
        store, runner = self.runner(ProtocolConfig.demo())
        while runner.phase.get("kind") != "practice_probe":
            store.clock.advance(.1)
            runner.tick()
        store.clock.advance(60)
        runner.tick()
        self.assertEqual(runner.state, "probe")
        runner.key("3")
        response = [e for e in store.events if e["kind"] == "probe_response"][-1]
        self.assertTrue(response["payload"]["practice"])
        self.assertEqual(response["payload"]["choice"], 3)

    def test_sart_response_does_not_advance_stimulus_early(self):
        store, runner = self.reach_sart()
        old = runner.trial["trial_id"]
        onset = runner.trial["onset_perf"]
        store.clock.advance(.2)
        runner.key("space")
        self.assertEqual(runner.trial["trial_id"], old)
        store.clock.seconds = onset + runner.config.sart_soa
        runner.tick()
        self.assertNotEqual(runner.trial["trial_id"], old)

    def test_late_key_does_not_answer_next_trial(self):
        store, runner = self.reach_sart()
        store.clock.seconds = runner.deadline + .1
        runner.key("space")
        self.assertFalse(runner.trial["responded"])
        self.assertTrue(store.events[-1]["payload"]["ignored_after_transition"])

    def test_probe_timeout_is_missing_not_guessed_label(self):
        store, runner = self.reach_sart()
        runner.show_probe()
        store.clock.seconds = runner.deadline
        runner.tick()
        response = [e for e in store.events if e["kind"] == "probe_response"][-1]
        self.assertIsNone(response["payload"]["choice"])
        self.assertTrue(response["payload"]["timeout"])

    def test_abort_invalidates_active_trial_and_preserves_events(self):
        store, runner = self.reach_sart()
        runner.abort("eeg_stream_stale")
        result = [e for e in store.events if e["kind"] == "trial_result"][-1]
        self.assertFalse(result["payload"]["valid"])
        self.assertEqual(result["payload"]["invalid_reason"], "eeg_stream_stale")
        self.assertFalse(runner.finished)
        self.assertEqual(store.events[-1]["kind"], "protocol_abort")
        self.assertEqual(runner.abort_reason, "eeg_stream_stale")
        with self.assertRaisesRegex(RuntimeError, "新建采集"):
            runner.start()

    def test_probe_choice_mapping_preserves_external_and_uncertain(self):
        store, runner = self.reach_sart()
        runner.show_probe()
        store.clock.advance(.5)
        runner.key("7")
        response = [e for e in store.events if e["kind"] == "probe_response"][-1]
        self.assertEqual(response["payload"]["choice"], 7)
        self.assertEqual(runner.state, "recovery")


if __name__ == "__main__":
    unittest.main()
