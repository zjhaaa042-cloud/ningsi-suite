import http.client
import io
import socket
import tempfile
import threading
import unittest
import zipfile
from concurrent.futures import ThreadPoolExecutor
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ningsi_studio.api.routes import _build_zip_response
from ningsi_studio.db import repository as repo, sqlite_store as store
from ningsi_studio.http.router import ApiError, Response, Router
from ningsi_studio.http.server import AppServer, StudioRequestHandler, MAX_BODY_BYTES
from ningsi_studio.http.sse import EventBus
from ningsi_studio.settings import Settings
from ningsi_studio.domain import export


class BodyValidationTests(unittest.TestCase):
    def handler(self, lengths, data=b'', transfer=None):
        h = object.__new__(StudioRequestHandler)
        h.headers = Message()
        for value in lengths:
            h.headers['Content-Length'] = value
        if transfer:
            h.headers['Transfer-Encoding'] = transfer
        h.connection = Mock()
        h.connection.gettimeout.return_value = None
        h.rfile = io.BytesIO(data)
        h.close_connection = False
        return h

    def test_framing_limits_and_short_reads(self):
        for lengths, data, transfer, status in [
            (['-1'], b'', None, 400), (['abc'], b'', None, 400),
            (['1', '1'], b'x', None, 400),
            ([str(MAX_BODY_BYTES + 1)], b'', None, 413),
            (['4'], b'ab', None, 400), ([], b'', 'chunked', 400),
        ]:
            with self.subTest(lengths=lengths, transfer=transfer):
                h = self.handler(lengths, data, transfer)
                with self.assertRaises(ApiError) as caught:
                    h._read_body()
                self.assertEqual(caught.exception.status, status)
                self.assertTrue(h.close_connection)
        self.assertEqual(self.handler(['3'], b'abc')._read_body(), b'abc')
        self.assertEqual(self.handler(['0'])._read_body(), b'')

    def test_total_deadline_and_socket_timeout(self):
        h = self.handler(['2'])
        h.rfile = Mock()
        h.rfile.read1.return_value = b'x'
        with patch('ningsi_studio.http.server.time.monotonic', side_effect=[0, 0, 11]):
            with self.assertRaises(ApiError) as caught:
                h._read_body()
        self.assertEqual(caught.exception.status, 408)
        h.connection.settimeout.assert_called_with(None)
        h.rfile.read1.side_effect = TimeoutError
        with self.assertRaises(ApiError) as caught:
            h._read_body()
        self.assertEqual(caught.exception.status, 408)

    def test_unauthorized_request_is_rejected_without_waiting_for_body(self):
        router = Router()
        called = Mock(return_value=Response.json({}))
        router.post('/api/test')(called)
        server = AppServer(router, Settings(port=0, api_token='secret'))
        server.start_background()
        try:
            with socket.create_connection(('127.0.0.1', server.port), timeout=2) as sock:
                sock.sendall((f'POST /api/test HTTP/1.1\r\nHost: 127.0.0.1:{server.port}'
                              '\r\nContent-Length: 100\r\n\r\n').encode())
                response = http.client.HTTPResponse(sock)
                response.begin()
                self.assertEqual(response.status, 401)
                response.read()
            called.assert_not_called()
        finally:
            server.stop()


class BacklogTests(unittest.TestCase):
    def test_zero_capacity_still_delivers_live_events(self):
        bus = EventBus('test', backlog=0)
        subscriber, _ = bus.subscribe()
        for i in range(100):
            bus.publish('window', i)
        self.assertEqual(bus.snapshot(), [])
        self.assertEqual(subscriber.events.qsize(), 100)

    def test_capacity_and_replay_order(self):
        with self.assertRaises(ValueError):
            EventBus('test', backlog=-1)
        bus = EventBus('test', backlog=1)
        for i in range(3):
            bus.publish('window', i)
        subscriber, count = bus.subscribe(0)
        self.assertEqual(count, 1)
        self.assertEqual(subscriber.events.get_nowait()['id'], 3)


class ZipConcurrencyTests(unittest.TestCase):
    def test_concurrent_downloads_use_independent_files_and_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(data_dir=Path(tmp))
            settings.ensure_dirs()
            store.initialize(settings.db_path)
            source = Path(tmp) / 'report.txt'
            source.write_text('report', encoding='utf-8')
            with store.connect(settings.db_path) as conn:
                subject = repo.create_subject(conn, 'p01')
                session = repo.create_session(conn, subject)
                repo.add_artifact(conn, session['id'], 'report', str(source))
            uuid = session['uuid']
            manager = SimpleNamespace(get=lambda _: None)
            barrier = threading.Barrier(2)
            original = export.build_zip
            targets = []

            def build(target, files, meta):
                targets.append(target)
                original(target, files, meta)
                barrier.wait(timeout=5)
                return target

            def download():
                return _build_zip_response(uuid, settings.db_path, settings, manager)

            with patch.object(export, 'build_zip', build), ThreadPoolExecutor(2) as pool:
                responses = list(pool.map(lambda _: download(), range(2)))
            self.assertNotEqual(targets[0], targets[1])
            for response in responses:
                with zipfile.ZipFile(io.BytesIO(response.body)) as archive:
                    self.assertIsNone(archive.testzip())
                    self.assertEqual(archive.read('report/report.txt'), b'report')
            self.assertFalse(list(settings.runs_root.glob('*/export-*')))
            with patch.object(export, 'build_zip', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    download()
            self.assertFalse(list(settings.runs_root.glob('*/export-*')))
