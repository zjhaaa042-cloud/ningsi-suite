"""Shared runtime control signals, independent of protocol implementations."""

import threading
import time


class SessionCancelled(RuntimeError):
    """会话被外部取消。"""


class DemoClock:
    """A session-local clock. Pause takes effect at a worker checkpoint."""

    def __init__(self, bus):
        self.bus = bus
        self.condition = threading.Condition()
        self.requested = False
        self.stopped = False
        self.since = None
        self.total = 0.0
        self.revision = 0

    def now(self):
        with self.condition:
            return (self.since if self.since is not None else time.perf_counter()) - self.total

    def state(self):
        with self.condition:
            current = 0.0 if self.since is None else time.perf_counter() - self.since
            return {"available": not self.stopped, "paused": self.since is not None, "revision": self.revision,
                    "paused_seconds": round(self.total + current, 3)}

    def _resume(self):
        self.requested = False
        if self.since is not None:
            self.total += time.perf_counter() - self.since
            self.since = None
            self.revision += 1
            self.bus.publish("demo_pause", self.state())
        self.condition.notify_all()

    def set_paused(self, paused):
        with self.condition:
            if self.stopped:
                return self.state()
            if paused:
                self.requested = True
                self.condition.notify_all()
                self.condition.wait_for(lambda: self.since is not None or not self.requested)
            else:
                self._resume()
            return self.state()

    def stop(self):
        with self.condition:
            self.stopped = True
            self._resume()

    def wait(self, seconds, cancel):
        with self.condition:
            end = self.now() + seconds
            while True:
                if cancel.is_set():
                    raise SessionCancelled()
                if self.requested:
                    if self.since is None:
                        self.since = time.perf_counter()
                        self.revision += 1
                        self.bus.publish("demo_pause", self.state())
                        self.condition.notify_all()
                    self.condition.wait()
                    continue
                left = end - self.now()
                if left <= 0:
                    return
                self.condition.wait(left)
