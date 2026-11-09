"""SQLite 连接与迁移：WAL 模式、忙等待、幂等建表。

设计要点：
- 每个请求/线程用短连接（`connect()` 是上下文管理器），避免 sqlite3 的跨线程限制；
- WAL + busy_timeout 让"运行线程写、HTTP 线程读"能并行；
- `schema.sql` 为权威 DDL，`schema_version` 表记录已应用的版本号，重复执行安全。
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_FILE = Path(__file__).with_name("schema.sql")
SCHEMA_VERSION = "0001-init"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=10.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    # journal_mode=WAL 是持久化设置：建库时设置一次即可。
    # 每个连接都执行 PRAGMA journal_mode 会尝试拿一次排它锁，
    # 在"运行线程写 + HTTP 线程读"的并发下正是 database is locked 的常见来源。
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=10000")
    return conn


def _migrate(conn) -> None:
    """幂等列迁移。

    `schema.sql` 用 `CREATE TABLE IF NOT EXISTS`，所以**给已有库加列必须显式 ALTER**，
    否则老库（用户已经存了几百条会话）会缺列、查询直接报 no such column。
    目前只有一条：
    - `sessions.protocol`（'full' | 'short'）：短协议（去掉训练与模型、SART/PVT 减半）需要按会话记住，
      运行线程、报告与界面都要读它。
    """
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()}
    if "protocol" not in columns:
        conn.execute("ALTER TABLE sessions ADD COLUMN protocol TEXT NOT NULL DEFAULT 'full'")


def initialize(path) -> Path:
    """建库建表（幂等），返回数据库路径。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = _connect(target)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(SCHEMA_FILE.read_text(encoding="utf-8"))
        _migrate(conn)
        conn.execute(
            "INSERT OR IGNORE INTO schema_version (version, applied_at) VALUES (?, ?)",
            (SCHEMA_VERSION, utcnow()),
        )
    finally:
        conn.close()
    return target


@contextmanager
def connect(path, *, retries: int = 5, base_delay: float = 0.05):
    """短连接写事务；只重试 BEGIN/COMMIT，调用方的业务代码始终只执行一次。

    contextmanager 无法重放 with 块。业务语句失败时完整回滚并保留原异常；
    COMMIT 遇到 SQLITE_BUSY 时事务仍有效，可以在同一连接上重试提交。
    """
    conn = _connect(Path(path))
    try:
        _transaction_command(conn, "BEGIN IMMEDIATE", retries, base_delay)
        yield conn
        _transaction_command(conn, "COMMIT", retries, base_delay)
    except BaseException:
        if conn.in_transaction:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
        raise
    finally:
        conn.close()


def _transaction_command(conn, statement: str, retries: int, base_delay: float) -> None:
    attempts = max(1, int(retries))
    for attempt in range(1, attempts + 1):
        try:
            conn.execute(statement)
            return
        except sqlite3.OperationalError as exc:
            code = getattr(exc, "sqlite_errorcode", 0) & 0xff
            busy = code in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED) or "locked" in str(exc).lower()
            if not busy or attempt == attempts:
                raise
            time.sleep(max(0.0, base_delay) * attempt)


@contextmanager
def read_only(path):
    """只读场景：不开显式事务，交给 SQLite 自动提交。"""
    conn = _connect(Path(path))
    try:
        yield conn
    finally:
        conn.close()


def schema_versions(path) -> list[str]:
    with read_only(path) as conn:
        rows = conn.execute("SELECT version FROM schema_version ORDER BY version").fetchall()
    return [row["version"] for row in rows]
