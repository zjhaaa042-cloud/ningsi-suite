"""Documented device parameters reach the API and actual preview conditioning."""
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from ningsi_studio.acquisition.lsl import StreamDescriptor
from ningsi_studio.core.live_source import ManagedLslSource, build_source
from ningsi_studio.db import repository as repo, sqlite_store as store
from tests.helpers import StudioTestCase


NAME = 'BioMulti Lite EEG-ManualFixture'
DEVICE = 'lsl:' + NAME
ROW = {'key': DEVICE, 'kind': 'lsl', 'stream_kind': 'eeg', 'stream_type': 'EEG',
       'device': NAME, 'channels': 2, 'srate': 250., 'nominal_srate': 250.,
       'is_eeg': True, 'real': True, 'simulated': False, 'declared_units': [],
       'channel_labels': ['Ch1', 'Ch2'], 'source_id': 'manual-fixture'}


class BsenseIntegrationTests(StudioTestCase):
    port_base = 0

    def test_session_persists_manual_defaults_without_device_forms(self):
        runtime = SimpleNamespace(alive=False, source=SimpleNamespace(
            key=DEVICE, kind='lsl', srate=250., channels=2))
        with patch('ningsi_studio.api.routes.live_source.list_available', return_value=[ROW]), \
             patch.object(self.app.manager, 'start_with_rollback', return_value=runtime):
            created = self.json_body(self.post('/api/sessions', {
                'participant': 'm01', 'device': DEVICE, 'protocol': 'ten_min', 'time_scale': 1}), 201)
        with store.read_only(self.db_path) as conn:
            saved = repo.get_session(conn, created['session']['uuid'])
        self.assertEqual(json.loads(saved['eeg_calibration']), {
            'source_unit': 'uV', 'scale_to_uv': 1., 'origin': 'manufacturer_manual'})
        self.assertEqual((saved['srate'], saved['channels']), (250., 2))

    def test_mismatched_rate_is_rejected_before_creating_a_session(self):
        before = self.json_body(self.get('/api/sessions'))['total']
        with patch('ningsi_studio.api.routes.live_source.list_available',
                   return_value=[{**ROW, 'nominal_srate': 200., 'srate': 200.}]), \
             patch.object(self.app.manager, 'start_with_rollback') as start:
            response = self.post('/api/sessions', {
                'participant': 'm02', 'device': DEVICE, 'protocol': 'ten_min', 'time_scale': 1})
        self.assert_error(response, 422, 'validation_failed')
        self.assertIn('250 Hz', self.json_body(response, 422)['error']['message'])
        start.assert_not_called()
        self.assertEqual(self.json_body(self.get('/api/sessions'))['total'], before)

    def test_preview_conditioning_uses_manual_microvolts_and_preserves_labels(self):
        source = ManagedLslSource(NAME, device=NAME, channels=2, srate=250.)
        descriptor = StreamDescriptor(kind='eeg', name=NAME, stream_type='EEG',
            channel_count=2, nominal_srate=250., channel_labels=('Ch1', 'Ch2'),
            source_id='manual-fixture', declared_units=('', ''))
        source._manager = Mock()
        source._manager.descriptor.return_value = descriptor
        t = np.arange(1500) / 250.
        raw = np.vstack([10*np.sin(2*np.pi*10*t), 8*np.sin(2*np.pi*12*t)])
        cleaned, record = source._condition(raw, 1000, 250., return_artifacts=True)
        self.assertEqual(source.operator_eeg_scale_to_uv, 1.)
        self.assertEqual(source.channel_labels, ['Ch1', 'Ch2'])
        self.assertEqual(cleaned.shape, (2, 1000))
        self.assertTrue(np.isfinite(cleaned).all())
        declared = ManagedLslSource('External EEG', device='External EEG', channels=2, srate=250.)
        declared._manager = Mock()
        declared._manager.descriptor.return_value = replace(descriptor, name='External EEG', declared_units=('uV', 'uV'))
        expected, _ = declared._condition(raw, 1000, 250., return_artifacts=True)
        np.testing.assert_allclose(cleaned, expected)

    def test_preview_does_not_use_single_channel_placeholder_for_documented_device(self):
        with patch.object(ManagedLslSource, 'start', return_value=False):
            source = build_source(DEVICE, wait_ready=False)
        self.assertEqual(source.channels, 2)
        self.assertEqual(source.srate, 250.)

    def test_only_auxiliary_streams_cannot_auto_start_an_eeg_session(self):
        rows = [{**ROW, 'key': 'lsl:Heart Rate', 'stream_kind': 'heart_rate', 'is_eeg': False}]
        with patch('ningsi_studio.api.routes.live_source.list_available', return_value=rows), \
             patch.object(self.app.manager, 'start_with_rollback') as start:
            response = self.post('/api/sessions', {'participant': 'aux01', 'device': 'auto'})
        self.assert_error(response, 409, 'conflict')
        start.assert_not_called()
        self.assertEqual(self.json_body(self.get('/api/subjects'))['total'], 0)

    def test_preview_rejects_empty_lsl_name_before_opening_any_stream(self):
        with patch.object(ManagedLslSource, 'start') as start:
            for name in ('lsl:', 'lsl:  '):
                self.assert_error(self.post('/api/devices/preview', {'source': name}), 422)
        start.assert_not_called()

    def test_failed_lsl_start_is_cleaned_up_and_preview_remains_inactive(self):
        from ningsi_studio.core.live_source import PreviewStream
        preview = PreviewStream()
        with patch('ningsi_studio.api.routes.live_source.preview_stream', return_value=preview), \
             patch.object(ManagedLslSource, 'start', side_effect=RuntimeError('receiver failed')), \
             patch.object(ManagedLslSource, 'stop') as stop:
            error = self.assert_error(self.post('/api/devices/preview', {'source': DEVICE}), 409)
            self.assertIn('receiver failed', error['message'])
            self.assertFalse(self.json_body(self.get('/api/devices/preview'))['active'])
        stop.assert_called_once()

    def test_failed_preview_stop_reports_failure_and_allows_retry(self):
        from ningsi_studio.core.live_source import PreviewStream, SourceInfo
        preview = PreviewStream()
        engine = SimpleNamespace(stop=Mock(side_effect=[RuntimeError('cannot stop inlet'), None]))
        source = SourceInfo(DEVICE, 'lsl', 250., 2, NAME, '', engine)
        with patch('ningsi_studio.api.routes.live_source.preview_stream', return_value=preview), \
             patch('ningsi_studio.core.live_source.build_source', return_value=source):
            self.json_body(self.post('/api/devices/preview', {'source': DEVICE}))
            error = self.assert_error(self.delete('/api/devices/preview'), 409)
            self.assertIn('cannot stop inlet', error['message'])
            self.assertTrue(self.json_body(self.get('/api/devices/preview'))['active'])
            self.assertTrue(self.json_body(self.delete('/api/devices/preview'))['stopped'])
            self.assertFalse(self.json_body(self.get('/api/devices/preview'))['active'])


if __name__ == '__main__':
    import unittest
    unittest.main()
