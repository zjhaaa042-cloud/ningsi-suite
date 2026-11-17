"""The capture/export and live-model feature contract. All filters use past data."""

from __future__ import annotations

import hashlib
import importlib.util
import marshal
from pathlib import Path
import sys

import numpy as np

from ningsi.signal.artifacts import Verdict
from ningsi.signal.window import WindowFeatures, analyze_window
from ningsi.signal.preprocess import preprocess
from .logistic import FEATURE_NAMES, features_from_window

FEATURE_SPEC = "capture-probe-6features-v1"
CONTRACT = {"spec": FEATURE_SPEC, "feature_names": list(FEATURE_NAMES),
            "window_sec": 4, "step_sec": 2, "context_sec": 6,
            "lookback_sec": [-12, -2], "min_windows": 3, "min_seconds": 6,
            "min_valid_ratio": 0.6, "min_srate": 100,
            "max_gap_sec": 0.1, "max_rate_error": 0.05,
            "aggregation": "arithmetic_mean_of_fully_contained_valid_windows"}
ENGINE_FILES = ("config.py", "signal/preprocess.py", "signal/window.py", "signal/artifacts.py",
                "signal/spectrum.py", "signal/bands.py", "signal/baseline.py", "signal/indicators.py")


def engine_fingerprint():
    import ningsi
    root = Path(ningsi.__file__).parent
    result = {}
    for name in ENGINE_FILES:
        path = root / name
        if not path.is_file() and getattr(sys, "frozen", False):
            path = Path(sys._MEIPASS) / "capture_engine_source" / name
        if path.is_file():
            data = path.read_bytes()
        else:
            module = "ningsi." + name.removesuffix(".py").replace("/", ".")
            spec = importlib.util.find_spec(module)
            data = marshal.dumps(spec.loader.get_code(module))
        result[name] = hashlib.sha256(data).hexdigest()
    return result


def implementation_fingerprint():
    """Protect model inference against drift in features and their aggregation."""
    import ningsi
    root = Path(ningsi.__file__).parent
    result = {}
    for name in ("models/research_features.py", "models/logistic.py"):
        path = root / name
        if not path.is_file() and getattr(sys, "frozen", False):
            path = Path(sys._MEIPASS) / "capture_engine_source" / name
        if path.is_file():
            data = path.read_bytes()
        else:
            module = "ningsi." + name.removesuffix(".py").replace("/", ".")
            spec = importlib.util.find_spec(module)
            data = marshal.dumps(spec.loader.get_code(module))
        result[name] = hashlib.sha256(data).hexdigest()
    return result


def condition_window(values, timestamps, rate, factor):
    """Input is channels x up-to-six-seconds, in original acquisition units."""
    values = np.asarray(values, dtype=float)
    stamps = np.asarray(timestamps, dtype=float)
    size = round(4 * rate)
    times = stamps[-size:]
    end = float(times[-1] + 1 / rate) if times.size and np.isfinite(times[-1]) else None
    start = float(times[0]) if times.size and np.isfinite(times[0]) else None
    reasons = []
    if factor is None or not np.isfinite(float(factor)) or float(factor) <= 0:
        reasons.append("unit_unconfirmed")
    if values.ndim != 2 or values.shape[1] < size or stamps.size < size:
        reasons.append("signal_too_short")
    if not np.isfinite(values).all():
        reasons.append("non_finite_signal")
    if not np.isfinite(stamps).all():
        reasons.append("clock_unsynchronized")
    elif stamps.size >= size:
        diffs = np.diff(stamps)
        if np.any(diffs <= 0):
            reasons.append("nonmonotonic_timestamps")
        if np.any(diffs > .1):
            reasons.append("timestamp_gap_over_100ms")
        if abs((times[-1] - times[0]) * rate / (size - 1) - 1) > .05:
            reasons.append("sample_rate_mismatch_over_5pct")
    if rate < 100:
        reasons.append("sample_rate_below_100hz")
    if reasons:
        window = WindowFeatures(end or 0., rate, values.shape[0], {}, {}, Verdict(False, tuple(reasons), {}), {})
    else:
        conditioned = []
        for channel in values:
            signal = channel * float(factor)
            filtered, _ = preprocess(signal - signal.mean(), rate)
            conditioned.append(filtered[-size:])
        window = analyze_window(np.asarray(conditioned), rate, t_end=end)
    return {"start": start, "end": end, "window": window,
            "features": features_from_window(window) if window.usable else {}}


def union_seconds(intervals):
    total, end = 0., -float("inf")
    for left, right in sorted(intervals):
        total += max(0., right - max(left, end))
        end = max(end, right)
    return total


def aggregate_probe(windows, onset, *, baseline_valid, phase_start):
    left, right = onset - 12, onset - 2
    candidates = [r for r in windows if r["start"] is not None and r["end"] is not None
                  and r["start"] >= left and r["end"] <= right]
    usable = [r for r in candidates if r["window"].usable]
    covered = union_seconds((r["start"], r["end"]) for r in usable)
    reasons = []
    if not baseline_valid:
        reasons.append("baseline_invalid")
    if len(usable) < 3 or covered < 6 or len(usable) < .6 * len(candidates):
        reasons.append("insufficient_usable_eeg")
    if phase_start > left:
        reasons.append("lookback_crosses_phase")
    features = {name: float(np.mean([r["features"][name] for r in usable])) for name in FEATURE_NAMES} if not reasons else {}
    return {"features": features, "reasons": reasons, "windows_total": len(candidates),
            "windows_usable": len(usable), "usable_seconds": covered,
            "window_start_lsl": left, "window_end_lsl": right}
