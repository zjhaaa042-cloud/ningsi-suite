"""量表计分与行为任务的复现性测试。"""

from __future__ import annotations

import unittest

from tests.helpers import temp_root
from ningsi.behavior.pvt import build_schedule, simulate_result
from ningsi.behavior.sart import NOGO_DIGIT, SartResult, build_sequence
from ningsi.scales.instruments import SAS, SDS
from ningsi.scales.scoring import level_of, score_scale
from ningsi.scales.store import read_records, save_scale_run


class ScaleTest(unittest.TestCase):
    def test_reverse_items_match_published_rule(self):
        self.assertEqual(SAS.reverse_indices(), (5, 9, 13, 17, 19))
        self.assertEqual(len(SDS.reverse_indices()), 10)

    def test_scoring_bounds_and_levels(self):
        self.assertEqual(score_scale("SAS", [1] * 20).standard_score, 44)
        self.assertEqual(score_scale("SAS", [4] * 20).standard_score, 81)
        self.assertEqual(score_scale("SAS", [4] * 20).level, "重度")
        self.assertEqual(level_of(49), "正常范围")
        self.assertEqual(level_of(50), "轻度")
        self.assertEqual(level_of(70), "重度")

    def test_reverse_scoring_changes_result(self):
        forward = score_scale("SAS", [4] * 20).raw_score
        self.assertEqual(forward, 80 - 5 * 3)

    def test_missing_answers_are_recorded(self):
        result = score_scale("SDS", {1: 2})
        self.assertEqual(len(result.missing_items), 19)
        self.assertEqual(result.answered, 1)

    def test_invalid_length_rejected(self):
        with self.assertRaises(ValueError):
            score_scale("SAS", [1] * 19)

    def test_records_roundtrip(self):
        root = temp_root("scales")
        result = score_scale("SAS", [2] * 20)
        path = save_scale_run(root, "p01", "01", "001", result, [2] * 20)
        records = read_records(path)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["result"]["standard_score"], result.standard_score)


class BehaviorTest(unittest.TestCase):
    def test_sequence_is_reproducible_and_balanced(self):
        first = build_sequence("p01", "01", "001")
        second = build_sequence("p01", "01", "001")
        other = build_sequence("p02", "01", "001")
        self.assertEqual(first.digits, second.digits)
        self.assertNotEqual(first.digits, other.digits)
        self.assertEqual(len(first.nogo_positions), 20)
        self.assertTrue(all(first.digits[i] == NOGO_DIGIT for i in first.nogo_positions))
        self.assertEqual(len(set(first.nogo_positions)), 20)

    def test_commission_and_omission_counts(self):
        sequence = build_sequence("p01", "01", "001")
        responded = tuple(True for _ in range(sequence.trials))
        rts = tuple(0.3 for _ in range(sequence.trials))
        score = SartResult("p01", "01", "001", sequence, responded, rts).score()
        self.assertEqual(score["commission_errors"], 20)
        self.assertEqual(score["omission_errors"], 0)

        responded = tuple(False for _ in range(sequence.trials))
        rts = tuple(None for _ in range(sequence.trials))
        score = SartResult("p01", "01", "001", sequence, responded, rts).score()
        self.assertEqual(score["omission_errors"], sequence.trials - 20)
        self.assertEqual(score["commission_errors"], 0)

    def test_pvt_schedule_and_lapses(self):
        self.assertEqual(build_schedule(3), build_schedule(3))
        result = simulate_result(3, base_rt=0.3, lapse_probability=0.5)
        self.assertGreater(result.score()["lapses"], 0)
        self.assertTrue(result.score()["valid"])


if __name__ == "__main__":
    unittest.main()
