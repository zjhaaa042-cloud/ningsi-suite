"""Locate observed signal integrity failures on a verified Capture sample axis."""

from dataclasses import dataclass
import math

import numpy as np


_MAX_GAP_SEC = .100
_CLOCK_ROUNDING_SEC = 1e-9


@dataclass(frozen=True)
class IntegrityAudit:
    spans: list[dict]
    _bad_sample_count: np.ndarray
    _bad_edge_count: np.ndarray

    def window_intact(self, first: int, stop: int) -> bool:
        """Check both leads and the internal clock edges of a half-open sample window."""
        if not 0 <= first < stop < len(self._bad_sample_count):
            return False
        return bool(
            self._bad_sample_count[stop] == self._bad_sample_count[first]
            and self._bad_edge_count[stop] == self._bad_edge_count[first + 1]
        )


def _true_runs(mask):
    return np.flatnonzero(np.diff(mask, prepend=False, append=False)).reshape(-1, 2)


def _prefix_count(mask):
    counts = np.empty(len(mask) + 1, dtype=np.intp)
    counts[0] = 0
    np.cumsum(mask, out=counts[1:])
    return counts


def audit_integrity(values, stamps, rate) -> IntegrityAudit:
    """Find hard integrity evidence without treating amplitude or spectral changes as hardware faults.

    A sample span uses half-open indices. A zero-length clock span names the edge
    immediately before that index. Its timestamps are the measured values on
    either side, even when the source time moves backwards.
    """
    values = np.asarray(values)
    stamps = np.asarray(stamps, dtype=float)
    rate = float(rate)
    if stamps.ndim != 1 or values.shape != (2, len(stamps)):
        raise ValueError("Integrity audit requires two leads and one timestamp per sample.")
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("Integrity audit requires a positive finite sample rate.")

    finite_time = np.isfinite(stamps)
    finite_values = np.isfinite(values)
    finite_pair = finite_time[:-1] & finite_time[1:]
    elapsed = np.zeros(len(finite_pair))
    with np.errstate(over="ignore"):
        np.subtract(stamps[1:], stamps[:-1], out=elapsed, where=finite_pair)
    nonincreasing = finite_pair & (elapsed <= 0.)
    gaps = finite_pair & (elapsed > _MAX_GAP_SEC + _CLOCK_ROUNDING_SEC)
    broken_edges = nonincreasing | gaps
    bad_edges = np.zeros(len(stamps), dtype=bool)
    bad_edges[1:] = broken_edges
    bad_samples = ~(finite_time & finite_values.all(axis=0))
    spans = []

    def record_samples(first, stop, channel, reason):
        # Use an observed boundary; never extend a span across a clock gap.
        known_end = stop < len(stamps) and finite_time[stop] and not broken_edges[stop - 1]
        spans.append({
            "channel": channel,
            "start_sample_index": int(first),
            "stop_sample_index": int(stop),
            "start_lsl": float(stamps[first]) if finite_time[first] else None,
            "end_lsl": float(stamps[stop]) if known_end else None,
            "reason": reason,
        })

    for first, stop in _true_runs(~finite_time):
        record_samples(first, stop, None, "clock_timestamp_missing")
    for edge_mask, reason in ((nonincreasing, "clock_nonincreasing"), (gaps, "clock_gap_over_100ms")):
        for left in np.flatnonzero(edge_mask):
            right = int(left + 1)
            spans.append({
                "channel": None,
                "start_sample_index": right,
                "stop_sample_index": right,
                "start_lsl": float(stamps[left]),
                "end_lsl": float(stamps[right]),
                "reason": reason,
            })

    # At the nominal rate N samples cover N/rate seconds of a continuous signal.
    min_constant_samples = max(2, math.ceil(rate))
    for channel, (lead, finite) in enumerate(zip(values, finite_values), 1):
        for first, stop in _true_runs(~finite):
            record_samples(first, stop, channel, "raw_nonfinite")
        equal_edges = (lead[:-1] == lead[1:]) & finite[:-1] & finite[1:] & finite_pair & ~broken_edges
        for first, edge_stop in _true_runs(equal_edges):
            stop = edge_stop + 1
            if stop - first >= min_constant_samples:
                bad_samples[first:stop] = True
                record_samples(first, stop, channel, "constant_signal")

    return IntegrityAudit(spans, _prefix_count(bad_samples), _prefix_count(bad_edges))
