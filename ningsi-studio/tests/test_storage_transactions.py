"""用真实 SQLite 锁验证事务边界重试，业务代码不可被重放。"""
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ningsi_studio.db import sqlite_store as store


class TransactionRetryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='ningsi-transaction-')
        self.addCleanup(self.temp.cleanup)
        self.path = store.initialize(Path(self.temp.name) / 'studio.sqlite3')
        with store.connect(self.path) as conn:
            conn.execute('CREATE TABLE audit_values (value INTEGER)')
        original = store._connect

        def short_timeout(path):
            conn = original(path)
            conn.execute('PRAGMA busy_timeout=0')
            return conn

        patcher = patch.object(store, '_connect', side_effect=short_timeout)
        patcher.start()
        self.addCleanup(patcher.stop)

    def values(self):
        with store.read_only(self.path) as conn:
            return [row[0] for row in conn.execute('SELECT value FROM audit_values')]

    def test_begin_lock_retries_then_executes_body_once(self):
        blocker = store._connect(self.path)
        self.addCleanup(blocker.close)
        blocker.execute('BEGIN IMMEDIATE')
        calls = []
        with patch.object(store.time, 'sleep', side_effect=lambda _: blocker.rollback()) as slept:
            with store.connect(self.path) as conn:
                calls.append('body')
                conn.execute('INSERT INTO audit_values VALUES (1)')
        self.assertEqual(slept.call_count, 1)
        self.assertEqual(calls, ['body'])
        self.assertEqual(self.values(), [1])

    def test_begin_lock_exhaustion_never_enters_body(self):
        blocker = store._connect(self.path)
        self.addCleanup(blocker.close)
        blocker.execute('BEGIN IMMEDIATE')
        with patch.object(store.time, 'sleep') as slept:
            with self.assertRaises(sqlite3.OperationalError):
                with store.connect(self.path, retries=3):
                    self.fail('拿到写锁之前不可执行事务体')
        self.assertEqual(slept.call_count, 2)
        blocker.rollback()
        self.assertEqual(self.values(), [])

    def commit_blocker(self):
        # rollback journal 模式下，读事务允许 BEGIN IMMEDIATE，但会阻塞 COMMIT。
        reader = store._connect(self.path)
        self.addCleanup(reader.close)
        reader.execute('PRAGMA journal_mode=DELETE')
        reader.execute('BEGIN')
        reader.execute('SELECT * FROM audit_values').fetchall()
        return reader

    def test_commit_lock_retries_without_replaying_body(self):
        reader = self.commit_blocker()
        calls = []
        with patch.object(store.time, 'sleep', side_effect=lambda _: reader.rollback()) as slept:
            with store.connect(self.path) as conn:
                calls.append('body')
                conn.execute('INSERT INTO audit_values VALUES (2)')
        self.assertEqual(slept.call_count, 1)
        self.assertEqual(calls, ['body'])
        self.assertEqual(self.values(), [2])

    def test_commit_lock_exhaustion_rolls_back(self):
        reader = self.commit_blocker()
        with patch.object(store.time, 'sleep') as slept:
            with self.assertRaises(sqlite3.OperationalError):
                with store.connect(self.path, retries=2) as conn:
                    conn.execute('INSERT INTO audit_values VALUES (3)')
        self.assertEqual(slept.call_count, 1)
        reader.rollback()
        self.assertEqual(self.values(), [])

    def test_body_lock_error_is_preserved_and_prior_writes_are_rolled_back(self):
        failure = sqlite3.OperationalError('database is locked')
        with patch.object(store.time, 'sleep') as slept:
            with self.assertRaises(sqlite3.OperationalError) as raised:
                with store.connect(self.path) as conn:
                    conn.execute('INSERT INTO audit_values VALUES (4)')
                    raise failure
        self.assertIs(raised.exception, failure)
        slept.assert_not_called()
        self.assertEqual(self.values(), [])

    def test_non_lock_error_and_interrupt_roll_back_without_retry(self):
        for failure in (ValueError('business error'), KeyboardInterrupt()):
            with self.subTest(error=type(failure).__name__), patch.object(store.time, 'sleep') as slept:
                with self.assertRaises(type(failure)):
                    with store.connect(self.path) as conn:
                        conn.execute('INSERT INTO audit_values VALUES (5)')
                        raise failure
                slept.assert_not_called()
                self.assertEqual(self.values(), [])
