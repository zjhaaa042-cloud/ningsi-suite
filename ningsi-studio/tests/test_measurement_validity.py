import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ningsi.acquisition.simulate import SyntheticEEG
from ningsi.assessment.report import SessionReport
from ningsi_studio.core.runtime import SessionRuntime
from ningsi_studio.domain.indicators import WindowEngine
from ningsi_studio.domain.behavior import PvtTask


class MeasurementValidityTests(unittest.TestCase):
    def test_diagnostic_record_does_not_publish_or_persist_training_scores(self):
        runtime = SessionRuntime.__new__(SessionRuntime)
        runtime.source = SimpleNamespace(kind='lsl')
        runtime.diagnostic_started = 0
        runtime.diagnostic_windows = []
        runtime.bus = Mock()
        step = WindowEngine(SyntheticEEG(seed=42), 250, device='test').step('focused')
        with patch('ningsi_studio.core.runtime.store.connect') as connect:
            runtime._record_diagnostic(step, 'training')
        connect.assert_not_called()
        runtime.bus.publish.assert_not_called()
        self.assertEqual(runtime.diagnostic_windows[0]['state'], 'unknown')
        self.assertNotIn('signal', runtime.diagnostic_windows[0])

    def test_live_buffer_waits_past_half_window(self):
        import numpy as np
        from ningsi_studio.core.live_source import ManagedLslSource
        source = ManagedLslSource('test', device='test', channels=2)
        manager = Mock()
        manager.descriptor.return_value = None
        manager.status.return_value = {'streams': {'eeg': {'seconds_since_last': 0}}}
        manager.buffer.return_value.samples_count.side_effect = [np.ones((2, 530)), np.ones((2, 1000))]
        source._manager = manager
        source._condition = Mock(side_effect=lambda data, expected, rate: data)
        with patch('ningsi_studio.core.live_source.time.sleep'):
            result = source.window()
        self.assertEqual(result.shape, (2, 1000))
        self.assertEqual(manager.buffer.return_value.samples_count.call_count, 2)

    def test_real_qc_waits_for_new_windows_without_synthetic_artifact(self):
        engine = WindowEngine(SyntheticEEG(seed=42), 250, device='test')
        wait = Mock()
        result = engine.device_qc(artifact=False, wait_between=wait)
        self.assertEqual(result['windows'], 4)
        self.assertEqual(wait.call_count, 3)
        self.assertTrue(all(call.args == (2.0,) for call in wait.call_args_list))

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
