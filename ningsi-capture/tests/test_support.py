import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ningsi" / "src"))

from ningsi_capture.clock import Stamp


class FakeClock:
    domain = "synthetic_test_clock"

    def __init__(self, seconds=1000.):
        self.seconds = seconds

    def now(self):
        return Stamp(self.seconds, self.seconds, "test-time")

    def advance(self, seconds):
        self.seconds += seconds


class MemoryStore:
    def __init__(self):
        self.metadata = {"participant": "TEST001", "visit": "01"}
        self.clock = FakeClock()
        self.events = []

    def event(self, kind, payload=None, stamp=None):
        self.events.append({"kind": kind, "payload": dict(payload or {}), "stamp": stamp or self.clock.now()})
        return len(self.events)
