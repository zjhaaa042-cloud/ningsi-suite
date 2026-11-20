"""Bounded collection of observed, nonduplicated EEG windows for assessment."""

from __future__ import annotations

import math

from ningsi import config


ASSESSMENT_BASELINE_TARGET_SEC = 60.0
ASSESSMENT_BASELINE_MAX_SEC = 180.0
ASSESSMENT_MONITOR_TARGET_SEC = 42.0
ASSESSMENT_MONITOR_MAX_SEC = 120.0
_CLOCK_EPSILON_SEC = 1e-6


class AssessmentSampleClock:
    """Place observed samples on the phase clock without crediting transport stalls."""

    def __init__(self, srate, begin, anchor):
        self.srate = srate
        self.begin = begin
        self.stream = self._stream(anchor) if anchor is not None else None
        self.index = anchor["sample_end"] if anchor is not None else 0
        self.end = 0.0

    @staticmethod
    def _stream(timing):
        return timing["buffer_id"], timing["connection_generation"]

    def end_of(self, timing):
        if timing is None or timing["received_at"] is None:
            return None
        stream = self._stream(timing)
        sample = timing["sample_end"]
        latest = timing["latest_sample_end"]
        arrival = timing["received_at"] - self.begin
        if stream != self.stream:
            # A new connection must supply a wholly new window. Start its
            # clock at the first observed position instead of reusing history.
            self.stream, self.index = stream, latest
            self.end = max(self.end, arrival)
            return None
        if sample < self.index:
            return None
        if sample == self.index:
            return self.end
        end = min(arrival - (latest - sample) / self.srate,
                  self.end + (sample - self.index) / self.srate)
        if end <= self.end:
            return None
        self.index, self.end = sample, end
        return end


class AssessmentEEGCollection:
    """Keep authentic windows and count the union of their usable source time.

    The scheduler supplies its elapsed time and the end of the analyzed source
    window. The latter is in the same clock domain as `phase_start`. The source
    window's width is `window_seconds`. A rolling window may not start before
    the phase.
    """

    def __init__(self, target_seconds, max_seconds, *, phase_start=0.0,
                 window_seconds=config.WINDOW_SEC, stable_windows=2):
        if (not all(math.isfinite(value) for value in
                    (target_seconds, max_seconds, phase_start, window_seconds))
                or not 0 < target_seconds <= max_seconds
                or not 0 < window_seconds <= max_seconds
                or not isinstance(stable_windows, int) or stable_windows < 1):
            raise ValueError("Invalid assessment collection limits.")
        self.target_seconds = target_seconds
        self.max_seconds = max_seconds
        self.phase_start = phase_start
        self.window_seconds = window_seconds
        self.stable_windows = stable_windows
        self.started = False
        self.steps = []
        self.sample_ends = []
        self.usable_mask = []
        self.usable_seconds = 0.0
        self.repeated_windows = 0
        self._pending = []
        self._last_source_end = None
        self._last_covered_end = None

    @property
    def target_reached(self):
        return self.usable_seconds + _CLOCK_EPSILON_SEC >= self.target_seconds

    @property
    def usable_windows(self):
        return sum(self.usable_mask)

    @property
    def waiting_for_confirmation(self):
        return bool(self._pending)

    def add(self, step, elapsed, *, sample_end):
        """Record an attempt and return whether it contributes a usable window.

        The first stable sequence is retained in full. After startup, every
        attempt inside the collection time limit remains in the denominator.
        A missing, outside-phase or repeated timestamp has a `None` entry in
        `sample_ends`. The step itself is always kept unmodified.
        """
        if not math.isfinite(elapsed) or not self.window_seconds <= elapsed <= self.max_seconds:
            return False

        if (sample_end is not None and math.isfinite(sample_end)
                and self.phase_start + self.window_seconds - _CLOCK_EPSILON_SEC <= sample_end
                <= self.phase_start + self.max_seconds + _CLOCK_EPSILON_SEC):
            if self._last_source_end is None or sample_end > self._last_source_end:
                self._last_source_end = sample_end
            else:
                self.repeated_windows += 1
                sample_end = None
        else:
            sample_end = None

        usable = bool(sample_end is not None and step.window is not None and step.window.usable)
        if self.started:
            self._retain(step, sample_end, usable)
            return usable
        if not usable:
            self._pending.clear()
            return False

        if self._pending and sample_end - self._pending[-1][1] > self.window_seconds + _CLOCK_EPSILON_SEC:
            self._pending.clear()
        self._pending.append((step, sample_end))
        if len(self._pending) < self.stable_windows:
            return False
        self.started = True
        for pending, end in self._pending:
            self._retain(pending, end, True)
        self._pending.clear()
        return True

    def _retain(self, step, sample_end, usable):
        self.steps.append(step)
        self.sample_ends.append(sample_end)
        self.usable_mask.append(usable)
        if usable:
            start = max(self.phase_start, sample_end - self.window_seconds)
            if self._last_covered_end is not None:
                start = max(start, self._last_covered_end)
            self.usable_seconds += sample_end - start
            self._last_covered_end = sample_end

    def state(self, elapsed):
        candidates = len(self.steps)
        usable = self.usable_windows
        return {
            "started": self.started,
            "candidate_windows": candidates,
            "usable_windows": usable,
            "valid_ratio": usable / candidates if candidates else 0.0,
            "usable_seconds": self.usable_seconds,
            "elapsed_seconds": elapsed,
            "target_seconds": self.target_seconds,
            "max_seconds": self.max_seconds,
            "target_reached": self.target_reached,
            "limit_reached": elapsed >= self.max_seconds,
            "repeated_windows": self.repeated_windows,
        }
