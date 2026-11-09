"""启动恢复的独占性，以及失败/无 HTTP 循环时的锁释放。"""
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from types import SimpleNamespace

from ningsi_studio.app import create_app
from ningsi_studio.core.runtime import SessionManager, SessionRuntime, SessionCancelled
from ningsi_studio.http.sse import EventBus
from ningsi_studio.db.ownership import DataDirectoryLock
from ningsi_studio.http.router import Conflict
from ningsi_studio.http.server import AppServer
from ningsi_studio.settings import Settings


class ApplicationLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='ningsi-lifecycle-')
        self.addCleanup(self.temp.cleanup)
        self.settings = Settings(data_dir=Path(self.temp.name), port=0)

    def test_stop_without_serve_forever_releases_socket_and_directory(self):
        app = create_app(self.settings)
        thread = threading.Thread(target=app.stop, daemon=True)
        thread.start()
        thread.join(2)
        self.assertFalse(thread.is_alive(), '未启动 HTTP 循环的应用也必须能关闭（CLI demo 使用此路径）')
        with DataDirectoryLock(self.settings.data_dir):
            pass

    def test_failed_bind_releases_directory(self):
        with patch.object(AppServer, 'bind', side_effect=RuntimeError('bind failed')):
            with self.assertRaisesRegex(RuntimeError, 'bind failed'):
                create_app(self.settings)
        app = create_app(self.settings)
        app.stop()

    def test_directory_lock_is_exclusive_across_processes_and_released_on_exit(self):
        source = Path(__file__).resolve().parents[1] / 'src'
        program = ('import sys; sys.path.insert(0, ' + repr(str(source)) + '); '
                   'from ningsi_studio.db.ownership import DataDirectoryLock; '
                   'lock = DataDirectoryLock(' + repr(str(self.settings.data_dir)) + ').acquire()')
        with DataDirectoryLock(self.settings.data_dir):
            result = subprocess.run([sys.executable, '-c', program], capture_output=True, timeout=10)
            self.assertNotEqual(result.returncode, 0, '第二进程不得占用同一目录')
        result = subprocess.run([sys.executable, '-c', program], capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        # 子进程没有显式 release；操作系统应在退出后自动释放锁。
        with DataDirectoryLock(self.settings.data_dir):
            pass

    def test_shutdown_rejects_requests_that_arrive_after_stop(self):
        manager = SessionManager(self.settings)
        manager.shutdown()
        self.assertTrue(manager.idle())
        with self.assertRaises(Conflict):
            manager.ensure_capacity()
        with self.assertRaises(Conflict):
            manager.start('0' * 32)

    def test_start_failure_releases_loaded_source_and_closes_bus(self):
        for failure in ('cancel', 'load', 'thread'):
            with self.subTest(failure=failure):
                engine = SimpleNamespace(stop=Mock())
                bus = EventBus('startup-test')
                subscriber, _ = bus.subscribe()
                runtime = SessionRuntime('startup-test', settings=self.settings, bus=bus)
                error = RuntimeError('startup failed')

                def load():
                    runtime.source = SimpleNamespace(engine=engine)
                    if failure == 'cancel':
                        runtime.cancel()
                    elif failure == 'load':
                        raise error

                with patch.object(runtime, '_load', side_effect=load), \
                     patch.object(threading.Thread, 'start', side_effect=error):
                    with self.assertRaises(SessionCancelled if failure == 'cancel' else RuntimeError) as caught:
                        runtime.start()
                if failure != 'cancel':
                    self.assertIs(caught.exception, error)
                engine.stop.assert_called_once_with()
                self.assertFalse(runtime.alive)
                self.assertTrue(bus.closed)
                self.assertEqual(bus.subscriber_count(), 0)
