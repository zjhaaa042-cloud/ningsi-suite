"""Measurement integrity and wording regressions for the six workflow fixes."""
import unittest

from ningsi.assessment.joint import assess
from ningsi.assessment.report import SessionReport
from ningsi.behavior.pvt import PvtResult, PvtTrial
from ningsi.training.neurofeedback import NeurofeedbackSession, Segment


class TrainingTimingTests(unittest.TestCase):
    def test_sparse_success_is_not_full_duration_success_or_adaptation(self):
        session = NeurofeedbackSession('p01', '01', '001', target=.5)
        segment = session.add_segment([(4, .8), (6, .8)], 50, excluded_windows=23,
                                      interruptions=list(range(8, 51, 2)))
        stats = segment.stats()
        self.assertEqual(stats['valid_seconds'], 4)
        self.assertEqual(stats['valid_time_ratio'], .08)
        self.assertEqual(stats['on_target_ratio'], .08)
        self.assertEqual(stats['on_target_valid_ratio'], 1)
        self.assertEqual(stats['on_target_sample_ratio'], 1)
        self.assertFalse(stats['coverage_sufficient'])
        self.assertFalse(stats['achieved'])
        self.assertEqual((session.target, session.hold_sec), (.5, 6))

    def test_irregular_spacing_is_time_weighted_and_gaps_are_not_filled(self):
        stats = Segment(1, 8, .5, 6, [(0, .8), (1, .8), (4, .2), (6, .8)]).stats()
        self.assertEqual(stats['valid_seconds'], 7)
        self.assertEqual(stats['on_target_seconds'], 5)
        self.assertEqual(stats['on_target_ratio'], .625)
        self.assertEqual(stats['on_target_valid_ratio'], .7143)

    def test_duplicate_and_outside_times_do_not_inflate_coverage(self):
        stats = Segment(1, 50, .5, 6, [(0, .8), (0, .8), (100, .8)]).stats()
        self.assertEqual(stats['valid_seconds'], 2)
        self.assertEqual(stats['valid_time_ratio'], .04)
        self.assertFalse(stats['coverage_sufficient'])

    def test_explicit_invalid_sample_breaks_hold_even_when_gap_is_short(self):
        stats = Segment(1, 50, .5, 6, [(0, .8), (2, .8), (4, .8), (6, .8)],
                        interruptions=[3]).stats()
        self.assertFalse(stats['achieved'])
        self.assertEqual(stats['max_hold_sec'], 2)
        self.assertEqual(stats['current_hold_sec'], 2)

    def test_latest_invalid_sample_clears_current_hold_without_erasing_history(self):
        stats = Segment(1, 50, .5, 6, [(0, .8), (2, .8), (4, .8), (6, .8)],
                        interruptions=[7]).stats()
        self.assertEqual(stats['current_hold_sec'], 0)
        self.assertEqual(stats['max_hold_sec'], 6)
        self.assertTrue(stats['achieved'])

    def test_adequate_coverage_without_continuous_hold_cannot_raise_target(self):
        session = NeurofeedbackSession('p01', '01', '001', target=.5)
        samples = [(i * 2, .8) for i in range(25)]
        segment = session.add_segment(samples, 50, interruptions=list(range(5, 50, 6)))
        self.assertTrue(segment.stats()['coverage_sufficient'])
        self.assertFalse(segment.stats()['achieved'])
        self.assertEqual(session.target, .5)


class PvtValidityTests(unittest.TestCase):
    def result(self, good, false=0, missed=0):
        trials = [PvtTrial(i, True, .2) for i in range(good)]
        trials += [PvtTrial(i, False, None, True) for i in range(false)]
        trials += [PvtTrial(i) for i in range(missed)]
        return PvtResult(trials).score()

    def test_seventy_false_starts_and_five_responses_are_not_normal_evidence(self):
        result = self.result(5, false=70)
        self.assertFalse(result['valid'])
        self.assertIn('excessive_false_starts', result['unavailable_reasons'])
        self.assertIn('insufficient_response_coverage', result['unavailable_reasons'])
        assessment = assess({}, behavior={'pvt': result})
        self.assertEqual(assessment.dimension_states['attention']['state'], '证据不足')

    def test_response_coverage_boundary_preserves_high_omission_evidence(self):
        self.assertTrue(self.result(60, missed=40)['valid'])
        self.assertFalse(self.result(59, missed=41)['valid'])
        assessment = assess({}, behavior={'pvt': self.result(60, missed=40)})
        self.assertIn('pvt_alertness', assessment.dimension_states['attention']['flagged'])

    def test_false_start_ratio_boundary(self):
        self.assertTrue(self.result(75, false=25)['valid'])
        self.assertFalse(self.result(74, false=26)['valid'])


class EvidenceWordingTests(unittest.TestCase):
    def test_uncollected_tasks_and_monitor_are_not_printed_as_zero_scores(self):
        report = SessionReport('p01', '01', '001', behavior={'sart': {}, 'pvt': {}},
                               extras={'protocol':'ten_min'})
        markdown = report.to_markdown()
        self.assertIn('SART：未采集', markdown)
        self.assertIn('PVT：未采集', markdown)
        self.assertIn('未采集独立监测窗口', markdown)
        self.assertNotIn('正确率 0%', markdown)
        self.assertNotIn('0/0 可用', markdown)
    def test_single_pvt_cannot_claim_multi_source_agreement_or_ability_level(self):
        result = assess({}, behavior={'pvt': {'valid': True, 'lapse_rate': .2}})
        state = result.dimension_states['attention']
        self.assertEqual(state['state'], '单一来源提示偏离')
        self.assertEqual(state['source_count'], 1)
        self.assertIn('仅PVT行为', result.conclusion)
        self.assertIn('缺少其他来源交叉验证', result.conclusion)
        self.assertNotIn('专注力相关指标偏低', result.conclusion)
        self.assertIn('无法比较', result.consistency['eeg_vs_behavior'])

    def test_below_threshold_is_not_called_normal(self):
        result = assess({}, behavior={'pvt': {'valid': True, 'lapse_rate': 0}})
        self.assertEqual(result.dimension_states['attention']['state'], '未触发工程阈值')
        self.assertNotIn('正常范围', result.conclusion)

    def test_two_eeg_indices_are_one_source_and_eeg_plus_behavior_are_two(self):
        eeg = {'available': True, 'valid_ratio': 1, 'focus': .2, 'load': .8, 'relax': .2}
        result = assess(eeg)
        self.assertEqual(result.dimension_states['stress']['state'], '单一来源提示偏离')
        result = assess(eeg, behavior={'pvt': {'valid': True, 'lapse_rate': .2}})
        self.assertEqual(result.dimension_states['attention']['state'], '一致提示偏离')
        self.assertEqual(result.dimension_states['attention']['source_count'], 2)
