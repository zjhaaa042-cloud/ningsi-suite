import unittest

import numpy as np

from ningsi_capture.artifact_calibration import POLICY, build_artifact_report
from ningsi_capture.native_motion import NativeMotion


RATE = 250


def _event(kind, time, **payload):
    return {"kind": kind, "time_lsl": float(time), "payload": payload}


def _record(shapes):
    cues = [(20 + 6 * n, action, rep) for n, (rep, action) in enumerate(
        (rep, action) for rep in range(1, 6) for action in shapes)]
    after = cues[-1][0] + 8
    end = after + 14
    stamps = np.arange((end + 20) * RATE) / RATE
    base = 12 * np.sin(2 * np.pi * 10 * stamps) + 7 * np.sin(2 * np.pi * 6 * stamps)
    raw = np.stack((base, .9 * base + 4 * np.cos(2 * np.pi * 11 * stamps)))
    events = [_event("artifact_calibration_start", 1),
              _event("phase_start", 2, phase="artifact_calibration_rest_before"),
              _event("phase_end", 16, phase="artifact_calibration_rest_before")]
    for cue, action, rep in cues:
        events.append(_event("artifact_calibration_cue", cue, action=action, repetition=rep))
    events += [_event("phase_start", after, phase="artifact_calibration_rest_after"),
               _event("phase_end", end, phase="artifact_calibration_rest_after"),
               _event("artifact_calibration_complete", end)]
    review = [(end + 3, next(iter(shapes)))]
    for cue, action in [(cue, action) for cue, action, _ in cues] + review:
        t = stamps - cue - .55
        envelope = np.exp(-.5 * (t / .10) ** 2)
        artifact = 240 * envelope if shapes[action] == "blink" else 93 * envelope * np.sin(2 * np.pi * 39 * t)
        raw += np.outer([1., .90], artifact)
    return raw, stamps, events, cues, review


class ArtifactCalibrationTests(unittest.TestCase):
    def test_two_repeatable_patterns_and_quiet_post_prefix_windows(self):
        raw, stamps, events, _, review = _record({"blink": "blink", "jaw": "muscle"})
        original = raw.copy()
        report = build_artifact_report(raw, stamps, RATE, events)
        self.assertTrue(np.array_equal(raw, original))
        self.assertEqual(report["policy"], POLICY)
        self.assertEqual(report["status"], "partial")
        for action in ("blink", "jaw"):
            self.assertEqual(report["actions"][action]["status"], "available")
            self.assertEqual(report["actions"][action]["train_count"], 3)
            self.assertEqual(report["actions"][action]["validation_hits"], 2)
        self.assertEqual(report["actions"]["eyes"]["status"], "unavailable")
        self.assertEqual(report["windows"][0]["actions"], ["blink"])
        self.assertEqual(report["windows"][-1]["status"], "no_signature")
        self.assertGreaterEqual(report["windows"][0]["start_lsl"], review[0][0] - 3)

    def test_overlapping_action_templates_abstain(self):
        raw, stamps, events, _, _ = _record({"blink": "blink", "eyes": "blink"})
        report = build_artifact_report(raw, stamps, RATE, events)
        self.assertEqual(report["status"], "unavailable")
        self.assertEqual(report["actions"]["blink"]["reason"], "held_out_inconsistent_or_overlapping")
        self.assertEqual(report["actions"]["eyes"]["reason"], "held_out_inconsistent_or_overlapping")
        self.assertEqual(report["windows"][0]["actions"], [])
        self.assertEqual(report["windows"][0]["status"], "uncertain")

    def test_repeatable_but_frequent_background_signature_abstains(self):
        raw, stamps, events, _, _ = _record({"blink": "blink"})
        rest = next(event["time_lsl"] for event in events if event["kind"] == "phase_start"
                    and event["payload"]["phase"] == "artifact_calibration_rest_after")
        for cue in np.arange(rest + 2., rest + 13., 2.):
            transient = 240 * np.exp(-.5 * ((stamps - cue - .55) / .10) ** 2)
            raw += np.outer([1., .90], transient)
        report = build_artifact_report(raw, stamps, RATE, events)
        action = report["actions"]["blink"]
        self.assertEqual(action["validation_hits"], 2)
        self.assertEqual(action["reason"], "frequent_during_rest_check")
        self.assertGreater(action["rest_check_matches"], 1)

    def test_head_match_requires_same_span_motion_evidence(self):
        raw, stamps, events, cues, review = _record({"head": "muscle"})
        unverified = build_artifact_report(raw, stamps, RATE, events)
        self.assertEqual(unverified["actions"]["head"]["reason"], "verified_gyro_required")
        gyro_times = np.arange(0., stamps[-1] + .1, .08)
        gyro = np.full((len(gyro_times), 3), .01)
        for cue, _, _ in cues:
            gyro[(gyro_times > cue + .2) & (gyro_times < cue + 1.), 0] = 1.6
        motion = NativeMotion(gyro_times, gyro, np.arange(len(gyro_times)))
        report = build_artifact_report(raw, stamps, RATE, events, motion=motion)
        self.assertEqual(report["actions"]["head"]["status"], "available")
        self.assertEqual(report["windows"][0]["status"], "uncertain")
        self.assertEqual(report["windows"][0]["actions"], [])
        cue, _ = review[0]
        gyro[(gyro_times > cue + .2) & (gyro_times < cue + 1.), 0] = 1.6
        report = build_artifact_report(raw, stamps, RATE, events, motion=motion)
        self.assertEqual(report["windows"][0]["actions"], ["head"])

    def test_incomplete_calibration_and_clock_gap_are_explicit(self):
        raw, stamps, events, cues, _ = _record({"blink": "blink"})
        incomplete = build_artifact_report(raw, stamps, RATE, events[:-1])
        self.assertEqual(incomplete["reason"], "incomplete_calibration_or_rest")
        held_out = cues[-1][0]
        stamps[(stamps >= held_out) & (stamps <= held_out + 1.)] = np.nan
        complete = events[-1]["time_lsl"]
        stamps[(stamps >= complete + 1.) & (stamps <= complete + 1.2)] = np.nan
        report = build_artifact_report(raw, stamps, RATE, events)
        self.assertEqual(report["actions"]["blink"]["validation_count"], 1)
        self.assertEqual(report["actions"]["blink"]["status"], "unavailable")
        self.assertEqual(report["windows"][0]["status"], "unavailable")
        self.assertEqual(report["windows"][0]["reason"], "incomplete_eeg_or_clock")

    def test_unverifiable_elapsed_span_does_not_schedule_empty_intervals(self):
        raw, stamps, events, _, _ = _record({"blink": "blink"})
        invalid_clock = stamps.copy()
        invalid_clock[-1] += 1e9
        displaced_rest_end = [dict(event, time_lsl=event["time_lsl"] + 1e9)
                              if event["kind"] == "artifact_calibration_complete" or (
                                  event["kind"] == "phase_end"
                                  and event["payload"].get("phase") == "artifact_calibration_rest_after")
                              else event for event in events]
        for trial_stamps, trial_events in ((invalid_clock, events), (stamps, displaced_rest_end)):
            with self.subTest(clock=trial_stamps is invalid_clock):
                report = build_artifact_report(raw, trial_stamps, RATE, trial_events)
                self.assertEqual(report["status"], "unavailable")
                self.assertEqual(report["reason"], "clock_span_exceeds_observed_samples")
                self.assertEqual(report["windows"], [])
