import time
import unittest
from unittest.mock import patch
import numpy as np

from tests.helpers import StudioTestCase
from ningsi.signal.graded import describe_fragments, FeedbackGate, robust_reference, union_seconds


class GradedTests(unittest.TestCase):
    def test_overlapping_fragments_do_not_double_count_coverage(self):
        self.assertEqual(union_seconds([(0,2),(0,2),(1,3),(5,7)]),5)

    def test_flat_and_nan_never_create_feedback(self):
        for data in (np.zeros((2,1000)), np.full((2,1000), np.nan)):
            self.assertEqual(describe_fragments(data,250)["grade"], "C")

    def test_good_channel_survives_bad_channel_without_strict_upgrade(self):
        t = np.arange(1000)/250
        good = 15*np.sin(2*np.pi*10*t)+10*np.sin(2*np.pi*6*t)
        result = describe_fragments(np.array([good, np.zeros(1000)]),250)
        self.assertEqual(result["grade"], "B")
        self.assertEqual([x['channel'] for x in result['channels']], [0])
        self.assertEqual(set(robust_reference([result])), {'0'})

    def test_feedback_downgrades_immediately_and_upgrades_slowly(self):
        gate = FeedbackGate()
        self.assertEqual([gate.update('A',True) for _ in range(3)], ['guided','guided','eeg'])
        self.assertEqual(gate.update('B',True), 'quality')
        self.assertEqual(gate.update('C',True), 'guided')
        self.assertNotEqual(gate.update('A',False), 'eeg')


class TenMinuteSessionTests(StudioTestCase):
    port_base = 18961

    def test_flat_signal_still_reports_behavior_without_eeg_rewards(self):
        from ningsi_studio.domain.indicators import WindowEngine
        original = WindowEngine.step
        def flat(engine, *args, **kwargs):
            with patch.object(engine.source, 'window', return_value=np.zeros((1,1000))):
                return original(engine, *args, **kwargs)
        with patch.object(WindowEngine, 'step', flat):
            created = self.json_body(self.post('/api/sessions', {
                'participant':'fl10', 'device':'sim-bsense', 'protocol':'ten_min',
                'time_scale':.05, 'create_subject':True}),201)
            uuid = created['session']['uuid']
            deadline = time.monotonic()+90
            while time.monotonic()<deadline:
                detail = self.json_body(self.get('/api/sessions/'+uuid),200)
                if detail['status'] in ('done','failed'): break
                time.sleep(.5)
        self.assertEqual(detail['status'],'done',detail.get('error'))
        report = self.json_body(self.get('/api/sessions/'+uuid+'/report'),200)
        self.assertGreater(report['behavior']['pvt']['trials'],0)
        self.assertFalse(report['extras']['baseline_eyes_open']['valid'])
        self.assertIsNone(report['extras']['training']['mean_focus'])
        self.assertFalse(report['extras']['ten_min']['changes'])
        for segment in report['extras']['ten_min']['segments']:
            self.assertFalse(segment['achieved'])
            self.assertNotIn('eeg',segment['mode_seconds'])
            self.assertIsNone(segment['mean_score'])
            self.assertAlmostEqual(sum(segment['mode_seconds'].values())+segment['unobserved_seconds'],50,places=2)

    def test_complete_profile_report_and_practice_exclusion(self):
        created = self.json_body(self.post('/api/sessions', {
            'participant':'ten01', 'device':'sim-bsense', 'protocol':'ten_min',
            'time_scale':.05, 'create_subject':True}),201)
        uuid = created['session']['uuid']
        deadline = time.monotonic()+100
        while time.monotonic()<deadline:
            detail = self.json_body(self.get('/api/sessions/'+uuid),200)
            if detail['status'] in ('done','failed','cancelled'):
                break
            time.sleep(1)
        self.assertEqual(detail['status'],'done',detail.get('error'))
        phases = [x['phase'] for x in detail['runs']]
        self.assertEqual(phases,['qc','baseline_open','practice','pvt','training','baseline_after','assessment','report'])
        report = self.json_body(self.get('/api/sessions/'+uuid+'/report'),200)
        ten = report['extras']['ten_min']
        self.assertEqual(len(ten['segments']),4)
        self.assertEqual(ten['budget_sec'],600)
        for ref in (*ten['baseline_before'].values(), *ten['baseline_after'].values()):
            self.assertLessEqual(ref['usable_seconds'],60)
        self.assertLessEqual(report['extras']['baseline_eyes_open']['quality']['usable_seconds'],60)
        self.assertEqual(report['behavior']['sart'],{})
        self.assertEqual(report['scales'],{})
        practice = next(x['payload'] for x in detail['runs'] if x['phase']=='practice')
        self.assertLess(practice['trials'],report['behavior']['pvt']['trials'])
        self.assertEqual(sum(report['extras']['signal_diagnostics']['grades'].values()),
                         report['extras']['signal_diagnostics']['total'])
        self.assertIn('十分钟分级反馈',report['report_markdown'])


if __name__ == '__main__':
    unittest.main()
