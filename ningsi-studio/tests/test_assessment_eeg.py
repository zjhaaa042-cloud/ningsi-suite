import unittest
from types import SimpleNamespace

from ningsi_studio.core.assessment_eeg import (
    ASSESSMENT_BASELINE_MAX_SEC,
    ASSESSMENT_BASELINE_TARGET_SEC,
    ASSESSMENT_MONITOR_MAX_SEC,
    ASSESSMENT_MONITOR_TARGET_SEC,
    AssessmentEEGCollection,
)


def step(*, usable=True):
    return SimpleNamespace(window=SimpleNamespace(usable=usable, t_end=999.))


class AssessmentEEGCollectionTests(unittest.TestCase):
    def test_requires_two_distinct_consecutive_usable_windows_after_warmup(self):
        capture = AssessmentEEGCollection(60., 180., phase_start=100.)
        self.assertFalse(capture.add(step(), 3., sample_end=103.))
        first = step()
        self.assertFalse(capture.add(first, 4., sample_end=104.))
        self.assertFalse(capture.add(first, 5., sample_end=104.))
        recent = step()
        self.assertFalse(capture.add(recent, 6., sample_end=106.))
        self.assertFalse(capture.started)
        self.assertEqual(capture.steps, [])
        self.assertEqual(capture.usable_seconds, 0.)

        following = step()
        self.assertTrue(capture.add(following, 8., sample_end=108.))
        self.assertTrue(capture.started)
        self.assertIs(capture.steps[0], recent)
        self.assertIs(capture.steps[1], following)
        self.assertEqual(capture.sample_ends, [106., 108.])
        self.assertEqual(capture.usable_seconds, 6.)
        self.assertEqual(capture.repeated_windows, 1)

    def test_retains_rejected_attempts_without_changing_or_crediting_them(self):
        capture = AssessmentEEGCollection(42., 120.)
        capture.add(step(), 4., sample_end=4.)
        accepted = step()
        capture.add(accepted, 6., sample_end=6.)
        missing = step()
        bad = step(usable=False)
        self.assertFalse(capture.add(missing, 8., sample_end=None))
        self.assertFalse(capture.add(accepted, 9., sample_end=6.))
        self.assertFalse(capture.add(bad, 10., sample_end=10.))
        self.assertFalse(capture.add(step(), 11., sample_end=9.))
        self.assertTrue(capture.add(step(), 12., sample_end=12.))

        self.assertIs(capture.steps[1], accepted)
        self.assertIs(capture.steps[2], missing)
        self.assertIs(capture.steps[3], accepted)
        self.assertIs(capture.steps[4], bad)
        self.assertEqual(accepted.window.t_end, 999.)
        self.assertTrue(accepted.window.usable)
        self.assertEqual(capture.sample_ends, [4., 6., None, None, 10., None, 12.])
        self.assertEqual(capture.usable_mask, [True, True, False, False, False, False, True])
        self.assertEqual(capture.state(12.)["valid_ratio"], 3 / 7)
        self.assertEqual(capture.usable_seconds, 10.)
        self.assertEqual(capture.repeated_windows, 2)

    def test_rejects_windows_straddling_the_phase_start(self):
        capture = AssessmentEEGCollection(42., 120., phase_start=100.)
        self.assertFalse(capture.add(step(), 4., sample_end=103.5))
        self.assertFalse(capture.add(step(), 6., sample_end=106.))
        self.assertTrue(capture.add(step(), 8., sample_end=108.))
        self.assertEqual(capture.sample_ends, [106., 108.])
        self.assertEqual(capture.usable_seconds, 6.)

    def test_collects_until_hard_limit_even_after_reaching_target(self):
        capture = AssessmentEEGCollection(6., 10.)
        capture.add(step(), 4., sample_end=4.)
        capture.add(step(), 6., sample_end=6.)
        self.assertTrue(capture.target_reached)
        self.assertFalse(capture.state(6.)["limit_reached"])
        capture.add(step(usable=False), 8., sample_end=8.)
        self.assertTrue(capture.add(step(), 10., sample_end=10.))
        self.assertFalse(capture.add(step(), 11., sample_end=11.))
        result = capture.state(11.)
        self.assertTrue(result["limit_reached"])
        self.assertEqual(result["candidate_windows"], 4)
        self.assertEqual(result["usable_windows"], 3)
        self.assertEqual(result["usable_seconds"], 10.)

    def test_policy_limits(self):
        self.assertEqual((ASSESSMENT_BASELINE_TARGET_SEC, ASSESSMENT_BASELINE_MAX_SEC), (60., 180.))
        self.assertEqual((ASSESSMENT_MONITOR_TARGET_SEC, ASSESSMENT_MONITOR_MAX_SEC), (42., 120.))
