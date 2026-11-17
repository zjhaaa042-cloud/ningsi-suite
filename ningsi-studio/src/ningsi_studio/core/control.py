"""Shared runtime control signals, independent of protocol implementations."""


class SessionCancelled(RuntimeError):
    """会话被外部取消。"""
