"""Regressions using the real collectors, training adapter and flow timing code."""
import json
from pathlib import Path
import re
import shutil
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ningsi.acquisition.simulate import SyntheticEEG
from ningsi_studio.core import phases
from ningsi_studio.core.runtime import SessionRuntime
from ningsi_studio.domain.behavior import PvtTask
from ningsi_studio.domain.indicators import WindowEngine
from ningsi_studio.domain.training import StudioTraining
from ningsi_studio.domain.assessment import eeg_summary, assess
from ningsi_studio.db import repository as repo, sqlite_store as store
from .helpers import StudioTestCase

ROOT = Path(__file__).resolve().parents[1]


class TrainingApiShapeTests(StudioTestCase):
    port_base = 0

    def test_missing_summary_values_remain_null_and_ten_plan_ignores_hidden_mode(self):
        with store.connect(self.db_path) as conn:
            subject = repo.create_subject(conn, repo.next_public_id(conn))
            row = repo.create_session(conn, subject, protocol='ten_min', training_mode='full')
            run = repo.start_run(conn, row['id'], 'training')
            repo.finish_run(conn, run)
            repo.update_run_payload(conn, run, {'training': {'spec': 'neurofeedback-v3',
                                                          'mode': 'ten_min', 'mean_focus': None}})
            repo.update_session(conn, row['uuid'], status='done', phase='done')
        summary = self.json_body(self.get('/api/sessions/' + row['uuid'] + '/training'))['summary']
        for key in ('mean_focus', 'on_target_ratio', 'on_target_valid_ratio', 'volatility', 'first_to_last_change'):
            self.assertIn(key, summary)
            self.assertIsNone(summary[key])
        detail = self.session_detail(row['uuid'])
        self.assertEqual(next(p for p in detail['phases'] if p['key'] == 'training')['duration_sec'], 240)


class BaselineBoundaryTests(unittest.TestCase):
    def collect(self, seconds, kind='lsl'):
        runtime = SessionRuntime.__new__(SessionRuntime)
        runtime.session = {'time_scale': 1}
        runtime.source = SimpleNamespace(kind=kind, device='test', srate=250)
        runtime.engine = WindowEngine(SyntheticEEG(seed=42), 250, device='test')
        runtime._check_cancel = Mock()
        runtime._publish_window = Mock()
        runtime._progress = Mock()
        clock = [100.]
        def advance(duration):
            clock[0] += duration
        runtime._sleep_real = advance
        runtime._sleep = advance
        with patch('ningsi_studio.core.runtime.time.monotonic', side_effect=lambda: clock[0]):
            baseline, steps = runtime._collect_baseline('rest', seconds, 'training', progress=False)
        return baseline, steps, runtime, clock[0]

    def test_thirty_second_phase_never_counts_prior_training_window(self):
        baseline, steps, runtime, ended = self.collect(30)
        self.assertTrue(baseline.valid)
        self.assertEqual(baseline.quality['usable_seconds'], 30)
        self.assertEqual(ended, 130)
        self.assertEqual(len(steps), 14)
        self.assertTrue(all(step.window.t_end - 4 >= 100 for step in steps))
        self.assertTrue(all(step.window.t_end <= ended for step in steps))
        self.assertEqual(runtime._publish_window.call_count, 15, 'crossing window remains diagnostic evidence')

    def test_post_training_buffer_and_simulated_baselines_share_boundary_rule(self):
        baseline, _, _, _ = self.collect(32)
        self.assertGreaterEqual(baseline.quality['usable_seconds'], 30)
        baseline, steps, _, _ = self.collect(30, kind='sim')
        self.assertEqual(baseline.quality['usable_seconds'], 30)
        self.assertTrue(all(step.window.t_end >= 4 for step in steps))


class StudioTrainingIntegrityTests(unittest.TestCase):
    def test_no_monitor_is_distinct_from_invalid_baseline_in_training_advice(self):
        for baseline_valid, expected in ((True, 'monitor_not_collected'), (False, 'baseline_invalid')):
            eeg = eeg_summary(baseline=SimpleNamespace(valid=baseline_valid), quality={'windows_total':0}, indicator_summary={})
            self.assertEqual(eeg['unavailable_code'], expected)
            result = assess(eeg, behavior={'pvt': {'valid':True,'lapse_rate':0}})
            advice = ' '.join(result.advice)
            if baseline_valid:
                self.assertIn('训练指数和训练目标单独查看', advice)
                self.assertNotIn('不支持个体', advice)
                self.assertNotIn('重新采集基线', advice)
            else:
                self.assertIn('不支持个体脑电评分或脑电训练目标', advice)
                self.assertIn('重新采集基线', advice)

    def test_invalid_feedback_break_is_saved_and_resets_immediately(self):
        trainer = StudioTraining('p01', '01', '001', None, mode='ten_min')
        for t in (0, 2):
            trainer.feed(.8, t)
        trainer.feed(None, 3, usable=False)
        self.assertEqual(trainer.feedback_status()['current_hold_sec'], 0)
        for t in (4, 6):
            trainer.feed(.8, t)
        self.assertFalse(trainer.feedback_status()['achieved'])
        segment = trainer.close_segment()
        self.assertEqual(segment['interruptions'], [3])
        self.assertFalse(segment['stats']['achieved'])
        self.assertEqual(segment['target_after'], .5)
        self.assertIn('覆盖不足', segment['adaptation_reason'])
        self.assertEqual(trainer.interruptions, [])

    def test_completed_high_false_start_task_fails_validity(self):
        task = PvtTask(onsets=list(range(75)))
        for index in range(75):
            task.submit(index=index, responded=index < 5, rt=.2 if index < 5 else None,
                        false_start=index >= 5)
        result = task.result()
        self.assertTrue(result['completed'])
        self.assertFalse(result['valid'])
        self.assertIn('excessive_false_starts', result['unavailable_reasons'])


class ProtocolConsistencyTests(unittest.TestCase):
    def test_training_duration_comes_from_the_executed_plan(self):
        for mode in ('quick', 'full'):
            row = next(p for p in phases.as_list('ten_min', mode) if p['key'] == 'training')
            trainer = StudioTraining('p01', '01', '001', None, mode='ten_min')
            self.assertEqual(row['duration_sec'], 240)
            self.assertEqual(row['training_plan']['segments'], trainer.describe()['segments'])
            self.assertEqual(row['training_plan']['segment_sec'], trainer.describe()['segment_sec'])
            self.assertEqual(row['training_plan']['rest_sec'], trainer.describe()['rest_sec'])
        for mode, seconds in (('quick', 272), ('full', 512)):
            row = next(p for p in phases.as_list('full', mode) if p['key'] == 'training')
            self.assertEqual(row['duration_sec'], seconds)

    def test_profile_totals_and_saved_duration_are_consistent(self):
        profiles = {p['key']: p for p in phases.profiles_as_list()}
        for key, profile in profiles.items():
            self.assertEqual(profile['total_sec'], round(sum(p['duration_sec'] for p in phases.as_list(key)), 1))
            self.assertAlmostEqual(profile['saved_sec'], profiles['full']['total_sec'] - profile['total_sec'])
        self.assertEqual(profiles['ten_min']['total_sec'], 600)
        self.assertEqual(profiles['full']['total_sec_by_training_mode']['full'] - profiles['full']['total_sec'], 240)

    def test_short_protocol_instructions_match_trials_and_pvt_duration(self):
        rows = {p['key']: p for p in phases.as_list('short')}
        sart = json.dumps(rows['sart'], ensure_ascii=False)
        self.assertIn('90', sart)
        self.assertNotIn('180', sart)
        self.assertNotIn('12 个练习', sart)
        self.assertNotIn('3 分钟', json.dumps(rows['pvt'], ensure_ascii=False))
        self.assertIn('120秒', json.dumps(rows['pvt'], ensure_ascii=False))

    @unittest.skipUnless(shutil.which('node'), 'Node required for actual flow timing regression')
    def test_frontend_ten_minute_estimate_ignores_hidden_regular_training_mode(self):
        source = (ROOT / 'web/js/views/flow.js').read_text(encoding='utf-8')
        function = re.search(r'const describeTiming = \(phase, state\) => \{.*?\n  \};', source, re.S).group()
        for mode in ('quick', 'full'):
            phase = next(p for p in phases.as_list('ten_min', mode) if p['key'] == 'training')
            program = f"const {{pick,fmtSeconds}} = await import({json.dumps((ROOT/'web/js/util.js').as_uri())});\n"
            program += "const local={phaseStartedAt:new Map(),now:0}; const ctx={store:{state:{currentSession:{time_scale:1,training_mode:" + json.dumps(mode) + "}}}};\n"
            program += function + '\nconsole.log(JSON.stringify(describeTiming(' + json.dumps(phase) + ',"pending")));'
            result = subprocess.run(['node', '--input-type=module', '-e', program],
                                    capture_output=True, text=True, encoding='utf-8', timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout.strip()), '预计 240s')
