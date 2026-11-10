import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ningsi.acquisition.simulate import SyntheticEEG
from ningsi.assessment.report import SessionReport
from ningsi_studio.core.runtime import SessionRuntime
from ningsi_studio.domain.indicators import WindowEngine
from ningsi_studio.domain.behavior import PvtTask


class MeasurementValidityTests(unittest.TestCase):
    def test_partial_pvt_never_enters_assessment_as_valid(self):
        task = PvtTask(onsets=list(range(10)))
        for index in range(5):
            task.submit(index=index, responded=True, rt=.2)
        result = task.result()
        self.assertFalse(result['valid'])
        self.assertIn('task_incomplete', result['unavailable_reasons'])
        for index in range(5, 10):
            task.submit(index=index, responded=True, rt=.355)
        result = task.result()
        self.assertTrue(result['valid'])
        self.assertEqual(result['lapses'], 0)
        self.assertEqual(result['lapses_355ms'], 5)

    def test_accelerated_real_baseline_cannot_claim_virtual_duration(self):
        runtime = SessionRuntime.__new__(SessionRuntime)
        runtime.source = SimpleNamespace(kind='lsl', device='test', srate=250)
        runtime.engine = WindowEngine(SyntheticEEG(seed=42), 250, device='test')
        runtime._check_cancel = Mock()
        runtime._publish_window = Mock()
        runtime._sleep = Mock()
        with patch('ningsi_studio.core.runtime.time.monotonic', side_effect=[100+i*.1 for i in range(15)]):
            baseline, _ = runtime._collect_baseline('rest', 30, 'baseline_open', progress=False)
        self.assertFalse(baseline.valid)
        self.assertAlmostEqual(baseline.quality['usable_seconds'], 5.4)

    def test_report_preserves_validation_boundaries(self):
        report = SessionReport('p', 's', 'r')
        self.assertEqual(report.as_dict()['threshold_status'], 'pending_validation')
        self.assertIn('不是专注概率', report.to_markdown())
        self.assertIn('待验证', report.to_markdown())
