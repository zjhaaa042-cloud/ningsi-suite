"""进程级数据目录锁：第二个服务不得把仍在运行的会话当作重启遗留记录。"""
from pathlib import Path
import os


class DataDirectoryLock:
    def __init__(self, data_dir):
        self.path = Path(data_dir) / '.studio.lock'
        self._file = None

    def acquire(self):
        handle = self.path.open('a+b')
        try:
            if os.name == 'nt':
                import msvcrt
                if self.path.stat().st_size == 0:
                    handle.write(b'\0')
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise RuntimeError(f'数据目录已被另一个服务占用：{self.path.parent}') from exc
        self._file = handle
        return self

    def release(self):
        handle, self._file = self._file, None
        if handle is not None:
            # 关闭文件句柄同时释放操作系统锁；进程意外退出时也会自动释放。
            handle.close()

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()
