"""Competition routes run the real processing and preserve source semantics."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ningsi_studio.api.routes import build_router
from ningsi_studio.db import repository as repo, sqlite_store as store
from ningsi_studio.http.router import Request, ValidationError
from ningsi_studio.settings import Settings


class ShowcaseApiTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.settings = Settings(data_dir=Path(temporary.name))
        store.initialize(self.settings.db_path)
        self.router = build_router(self.settings)
        self.addCleanup(self.router.human_models.close)

    def call(self, method, suffix, payload=None):
        request = Request(method, '/api/showcase/' + suffix, {}, {},
                          json.dumps(payload).encode() if payload is not None else b'')
        response = self.router.dispatch(request)
        return response.status, json.loads(response.body)

    def test_analysis_routes_compute_quality_and_keep_invalid_scores_missing(self):
        _, meta = self.call('GET', 'metadata')
        self.assertEqual(meta['bands_hz']['theta'], [4.0, 8.0])
        _, clean = self.call('POST', 'analyze', {'mode': 'demo', 'artifact': 'clean'})
        self.assertTrue(clean['source']['simulated'])
        self.assertGreater(clean['quality']['windows_usable'], 0)
        self.assertTrue(clean['baseline']['valid'])
        self.assertEqual(set(clean['latest']['scores']), {'focus', 'relax', 'load'})
        _, contaminated = self.call('POST', 'analyze', {'mode': 'demo', 'artifact': 'mixed'})
        self.assertFalse(contaminated['latest']['usable'])
        self.assertEqual(contaminated['latest']['scores'], {})
        self.assertTrue(any(point is None for point in contaminated['traces']['channels'][0]['cleaned_uv']))
        json.dumps(contaminated, allow_nan=False)

    def test_invalid_analysis_and_profile_are_validation_errors(self):
        for body in ([], {}, {'mode': 'raw', 'samples': [[1]], 'rate': 250, 'scale_to_uv': 1}):
            with self.subTest(body=body), self.assertRaises(ValidationError):
                self.call('POST', 'analyze', body)
        with self.assertRaises(ValidationError):
            self.call('POST', 'device-profile', {'srate': 'not-a-number'})

    def test_saved_device_declaration_does_not_become_a_compliance_certificate(self):
        _, result = self.call('POST', 'device-profile', {'manufacturer': '演示厂家', 'model': 'DEMO-2CH',
                              'electrodes': 'Fp1,Fp2', 'reference': '耳参考', 'srate': 250,
                              'source_unit': 'uV', 'scale_to_uv': 1,
                              'unit_basis': '演示声明，未作硬件标定', 'document_reference': '待补厂商资料'})
        self.assertEqual(result['compliance_status'], 'not_verified')
        _, loaded = self.call('GET', 'device-profile')
        self.assertEqual(loaded['profile'], result['profile'])
        self.assertEqual(loaded['declaration_status'], 'operator_declaration')

    def test_demo_session_is_explicit_simulation_and_uses_normal_persistence(self):
        runtime = SimpleNamespace(alive=True, source=SimpleNamespace(key='sim-bsense'))
        with patch.object(self.router.manager, 'start_with_rollback', return_value=runtime):
            status, result = self.call('POST', 'demo-session', {})
        self.assertEqual(status, 201)
        self.assertEqual(result['source_kind'], 'sim')
        session = result['session']
        self.assertEqual(session['device'], 'sim-bsense')
        self.assertEqual(session['protocol'], 'full')
        self.assertEqual(session['time_scale'], .01)
        with store.read_only(self.settings.db_path) as conn:
            saved = repo.get_session(conn, session['uuid'])
            self.assertEqual(saved['source'], 'sim-bsense')
        with self.assertRaises(ValidationError):
            self.call('POST', 'demo-session', {'device': 'lsl:unverified'})


if __name__ == '__main__':
    unittest.main()
