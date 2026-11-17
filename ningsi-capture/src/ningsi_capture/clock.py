"""Clock pairs preserve the distinction between LSL, Python and wall time."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import time


@dataclass(frozen=True)
class Stamp:
    perf: float
    lsl: float
    utc: str
    uncertainty_sec: float = 0.0

    def as_dict(self):
        return asdict(self)


class Clock:
    def __init__(self):
        try:
            from pylsl import local_clock
            self._lsl = local_clock
            self.domain = "lsl_local_clock"
        except (ImportError, OSError, RuntimeError):
            self._lsl = time.perf_counter
            self.domain = "perf_counter_fallback"

    def now(self):
        first = time.perf_counter()
        lsl = self._lsl()
        last = time.perf_counter()
        return Stamp((first + last) / 2, lsl,
                     datetime.now(timezone.utc).isoformat(), last - first)
