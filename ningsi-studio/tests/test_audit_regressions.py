"""深度审计回归：真实 HTTP、SQLite、报告和前端模块共同验证异常路径。"""
import http.client
import json
import shutil
import socket
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tests.helpers import StudioTestCase
from tests.test_frontend_dom import NODE_STUB
from ningsi_studio.app import create_app
from ningsi_studio.core import paired_ledger
from ningsi_studio.core.runtime import SessionRuntime
from ningsi_studio.db import repository as repo, sqlite_store as store
from ningsi_studio.settings import Settings

ROOT = Path(__file__).resolve().parents[1]


class AuditRegressionTests(StudioTestCase):
    port_base = 0

    @classmethod
    def make_settings(cls, data_dir):
        return Settings(data_dir=Path(data_dir), port=0, sse_heartbeat_sec=0.05)

    def seed_session(self, *, status='done', scores=None, phase='monitor', device='sim-bsense'):
        with store.connect(self.db_path) as conn:
            public_id = repo.next_public_id(conn)
            subject_id = repo.create_subject(conn, public_id)
            row = repo.create_session(conn, subject_id, device=device)
            for name, value in (scores or {}).items():
                repo.add_metric(conn, row['id'], 'indicator_summary', name,
                                mean=value, n=35, quality={'phase': phase})
            repo.update_session(conn, row['uuid'], status=status, phase=status)
            return dict(repo.get_session(conn, row['uuid']))

    def ledger(self, session, score):
        paired_ledger.append_history_record(self.app.settings.runs_root / session['uuid'], {
            'session_uuid': session['uuid'], 'participant': session['public_id'],
            'recorded_at': session['created_at'], 'device': session['device'],
            'srate': session['srate'], 'channels': session['channels'],
            'indicators': {'focus': score},
        })

    def test_monitor_disconnect_keeps_report_detail_trend_and_export_unavailable(self):
        original = SessionRuntime._run_monitor

        class MissingSource:
            def window(self, *args, **kwargs):
                return np.zeros((1, 0))

        def disconnected(runtime):
            source = runtime.engine.source
            runtime.engine.source = MissingSource()
            try:
                return original(runtime)
            finally:
                runtime.engine.source = source

        subject = self.create_subject()
        with patch.object(SessionRuntime, '_run_monitor', disconnected), \
             patch.object(SessionRuntime, '_sleep_real', lambda runtime, seconds: runtime._check_cancel()):
            session = self.create_session(subject['public_id'], protocol='short', time_scale=0.01)
            uuid = session['uuid']
            self.app.manager.get(uuid).join(timeout=10)
            self.assertFalse(self.app.manager.get(uuid).alive, '断流会话应及时完成并标明数据缺失')
        report = self.json_body(self.get(f'/api/sessions/{uuid}/report'))
        self.assertEqual(report['quality']['windows_usable'], 0)
        self.assertEqual(report['indicators']['summary'], {})
        detail = self.session_detail(uuid)
        self.assertEqual(detail['status'], 'done')
        self.assertEqual(detail['indicator_summary'], {})
        self.assertEqual(self.json_body(self.get(f'/api/sessions/{uuid}/live'))['indicator_summary'], {})
        self.assertEqual(self.json_body(self.get(f'/api/sessions/{uuid}/trend'))['points'], [])
        trend = self.json_body(self.get('/api/reports/trend?participant=' + subject['public_id']))
        self.assertEqual((trend['points'], trend['ledger_points']), ([], []))
        with store.read_only(self.db_path) as conn:
            stored = repo.get_session(conn, uuid)
            baseline_rows = [row for row in repo.list_metrics(conn, stored['id'], kind='indicator_summary')
                             if json.loads(row['quality']).get('phase') == 'baseline_closed']
            self.assertTrue(baseline_rows, '应确实存在容易被误用的闭眼基线评分')
            exported = next(row for row in repo.all_sessions_for_ledger(conn) if row['session_uuid'] == uuid)
        self.assertEqual(exported['indicators'], {})
        self.assertEqual(exported['quality'], {'valid_ratio': 0.0, 'passed': False})

    def test_summary_and_trend_ignore_baseline_and_unknown_phase(self):
        session = self.seed_session(scores={'focus': 0.2, 'relax': 0.3, 'load': 0.4})
        with store.connect(self.db_path) as conn:
            for phase in ('baseline_open', 'baseline_closed', 'training', None):
                repo.add_metric(conn, session['id'], 'indicator_summary', 'focus',
                                mean=0.9, n=60, quality={'phase': phase})
            # 损坏的历史 quality 字段也不能破坏查询或成为可用评分。
            conn.execute("UPDATE metrics SET quality='not-json' WHERE id=(SELECT MAX(id) FROM metrics)")
            points = repo.session_points(conn, public_id=session['public_id'], limit=1)
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0]['indicators'], {'focus': 0.2, 'relax': 0.3, 'load': 0.4},
                         'limit 应按会话计数，且其他阶段不可覆盖监测评分')
        self.assertEqual(self.session_detail(session['uuid'])['indicator_summary']['focus']['mean'], 0.2)

    def test_filtered_ledger_matches_participant_and_device_reference(self):
        first = self.seed_session(scores={'focus': 0.2})
        other = self.seed_session(scores={'focus': 0.8})
        self.ledger(first, 0.2)
        self.ledger(other, 0.8)
        # 同一被试后来换设备，两个来源都必须采用这次最新设备作为参照。
        with store.connect(self.db_path) as conn:
            latest = repo.create_session(conn, first['subject_id'], device='lsl:new-device')
            repo.add_metric(conn, latest['id'], 'indicator_summary', 'focus', mean=0.6,
                            n=35, quality={'phase': 'monitor'})
            repo.update_session(conn, latest['uuid'], status='done', phase='done')
            latest = dict(repo.get_session(conn, latest['uuid']))
        self.ledger(latest, 0.6)
        result = self.json_body(self.get('/api/reports/trend?participant=sub-' + first['public_id']))
        self.assertEqual(result['points'][0]['mean'], 0.6)
        self.assertEqual(result['ledger_points'][0]['mean'], 0.6)
        self.assertEqual(result['ledger_points'][0]['n'], 1)
        self.assertEqual(result['ledger_points'][0]['rejected'], 1)
        absent = self.json_body(self.get('/api/reports/trend?participant=z9999'))
        self.assertEqual((absent['points'], absent['ledger_points']), ([], []))

    def test_restart_recovers_pending_sessions_and_preserves_completed_results(self):
        interrupted = [self.seed_session(status=status) for status in ('running', 'queued')]
        terminal = [self.seed_session(status=status) for status in ('done', 'failed', 'cancelled')]
        with store.connect(self.db_path) as conn:
            completed_run = repo.start_run(conn, interrupted[0]['id'], 'baseline_open')
            repo.finish_run(conn, completed_run, payload={'baseline': {'valid': True}})
            unfinished_run = repo.start_run(conn, interrupted[0]['id'], 'scales')
            repo.update_run_payload(conn, unfinished_run, {'answers_saved': 20})
        self.restart_service()
        for session in interrupted:
            detail = self.session_detail(session['uuid'])
            self.assertEqual((detail['status'], detail['phase']), ('failed', 'failed'))
            self.assertIsNotNone(detail['ended_at'])
            self.assertTrue(detail['error'])
            self.assertFalse(detail['runtime']['alive'])
        runs = self.session_detail(interrupted[0]['uuid'])['runs']
        self.assertEqual(runs[0]['status'], 'done')
        self.assertTrue(runs[0]['payload']['baseline']['valid'])
        self.assertEqual(runs[1]['status'], 'failed')
        self.assertEqual(runs[1]['payload']['answers_saved'], 20)
        stamp = runs[1]['ended_at']
        self.restart_service()
        self.assertEqual(self.session_detail(interrupted[0]['uuid'])['runs'][1]['ended_at'], stamp)
        for session in terminal:
            self.assertEqual(self.session_detail(session['uuid'])['status'], session['status'])

    def test_second_service_cannot_recover_a_live_services_records(self):
        session = self.seed_session(status='running')
        with self.assertRaisesRegex(RuntimeError, '数据目录已被'):
            create_app(self.make_settings(self.data_dir))
        self.assertEqual(self.session_detail(session['uuid'])['status'], 'running')

    def test_cross_origin_reads_writes_and_preflights_are_rejected_without_side_effects(self):
        before = self.json_body(self.get('/api/subjects'))['total']
        origins = ('http://untrusted.invalid', 'null', 'http://localhost:1',
                   self.base_url + '.evil.invalid', self.base_url + '/path')
        for origin in origins:
            for method in ('GET', 'POST', 'OPTIONS'):
                with self.subTest(origin=origin, method=method):
                    response = self.request(method, '/api/subjects', raw_body=b'{"auto_id":true}',
                                            content_type='text/plain', headers={'Origin': origin})
                    self.assert_error(response, 403, 'forbidden')
                    self.assertNotIn('Access-Control-Allow-Origin', response.headers)
        self.assertEqual(self.json_body(self.get('/api/subjects'))['total'], before)

    def test_same_origin_and_api_clients_work_but_untrusted_hosts_do_not(self):
        self.assert_status(self.get('/api/subjects', headers={'Origin': self.base_url}), 200)
        self.assert_status(self.request('OPTIONS', '/api/subjects', headers={'Origin': self.base_url}), 204)
        self.assert_status(self.post('/api/subjects', {'auto_id': True}, headers={'Origin': self.base_url}), 201)
        for headers in ({'Host': f'untrusted.invalid:{self.port}'},
                        {'Host': f'untrusted.invalid:{self.port}', 'Origin': f'http://untrusted.invalid:{self.port}'},
                        {'Sec-Fetch-Site': 'cross-site'}):
            self.assert_error(self.get('/api/subjects', headers=headers), 403, 'forbidden')

    def test_session_events_release_head_and_disconnected_get_subscribers(self):
        subject = self.create_subject()
        session = self.create_session(subject['public_id'], time_scale=1)
        uuid = session['uuid']
        bus = self.app.manager.get(uuid).bus
        for _ in range(4):
            self.assert_status(self.request('HEAD', f'/api/sessions/{uuid}/events'), 200)
            self.assertEqual(bus.subscriber_count(), 0)
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=3)
        connection.request('GET', f'/api/sessions/{uuid}/events')
        client_socket = connection.sock
        response = connection.getresponse()
        try:
            self.assertEqual(response.status, 200)
            response.readline()
            self.assertEqual(bus.subscriber_count(), 1)
            client_socket.shutdown(socket.SHUT_RDWR)
        finally:
            response.close()
            connection.close()
        self.wait_for(bus.subscriber_count, lambda count: count == 0, timeout=3, interval=0.05,
                      message='断开 SSE 后应释放订阅')

    def test_finished_event_stream_reaches_eof(self):
        session = self.seed_session()
        connection, response = self.open_event_stream(session['uuid'], timeout=2)
        try:
            data = response.read().decode('utf-8')
            self.assertIn('event: closed', data)
            self.assertIn('event: finished', data)
        finally:
            response.close()
            connection.close()

    @unittest.skipUnless(shutil.which('node'), '需要 Node 运行前端模块')
    def test_rendered_download_links_work_with_url_and_meta_tokens(self):
        session = self.seed_session()
        uuid = session['uuid']
        artifact = self.data_dir / 'report.json'
        artifact.write_text(json.dumps({'participant': session['public_id'], 'extras': {'source_kind': 'sim'}}),
                            encoding='utf-8')
        with store.connect(self.db_path) as conn:
            repo.add_artifact(conn, session['id'], 'report_json', str(artifact))
        token = 'audit token+&?=/#'
        original_token = self.app.settings.api_token
        try:
            for token_source in ('url', 'meta', 'none'):
                self.app.settings.api_token = '' if token_source == 'none' else token
                program = NODE_STUB + '\n' + """
                    StubNode.prototype.append = function (...nodes) { this.children.push(...nodes); };
                    document.getElementById = () => null;
                    const token = TOKEN, source = SOURCE, base = BASE, uuid = UUID;
                    document.querySelector = () => source === 'meta' ? {getAttribute: () => token} : null;
                    globalThis.window = {location: {search: source === 'url' ? '?token=' + encodeURIComponent(token) : ''},
                                         setTimeout, clearTimeout};
                    const nativeFetch = globalThis.fetch;
                    globalThis.fetch = (path, init) => nativeFetch(new URL(path, base), init);
                    const {render} = await import(REPORT_URI);
                    const root = new StubNode('main');
                    await render(root, {params: new URLSearchParams({session: uuid}),
                        signal: new AbortController().signal, navigate() {},
                        store: {state: {}, setState() {}}});
                    const links = [];
                    const walk = node => {
                        if (node.tagName === 'A') links.push(node.getAttribute('href'));
                        for (const child of node.children || []) walk(child);
                    };
                    walk(root);
                    console.log(JSON.stringify(links));
                """.replace('REPORT_URI', json.dumps((ROOT / 'web/js/views/report.js').as_uri())) \
                    .replace('TOKEN', json.dumps(token)).replace('SOURCE', json.dumps(token_source)) \
                    .replace('BASE', json.dumps(self.base_url)).replace('UUID', json.dumps(uuid))
                completed = subprocess.run(['node', '--input-type=module', '-e', program],
                                           capture_output=True, text=True, timeout=20, cwd=ROOT)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                links = json.loads(completed.stdout.strip())
                self.assertEqual(len(links), 2, '应覆盖 ZIP 和文件列表中实际渲染的下载链接')
                for link in links:
                    with self.subTest(source=token_source, link=link):
                        self.assert_status(self.get(link), 200)
                        self.assertEqual('token=' in link, token_source != 'none')
            self.app.settings.api_token = token
            self.assert_error(self.get(f'/api/sessions/{uuid}/export.zip'), 401, 'unauthorized')
        finally:
            self.app.settings.api_token = original_token
