"""Studio live artifact control. Frozen research-model features use their original pipeline."""

from __future__ import annotations

import numpy as np

from ningsi.acquisition.eeg import UNIT_FACTORS, _unit
from ningsi.signal.artifacts import Verdict


SPEC = "studio-artifacts-v1"
_EOG = {"veog", "heog", "eog", "eogup", "eogdown", "eogleft", "eogright"}
ACCEL_LIMITS = {"g": .15, "mg": 150., "m/s2": 1.471}


def accel_unit(unit):
    return str(unit).strip().lower().replace("²", "2").replace("^", "")


def check_motion_reference(snapshot, eeg_times, axes, unit):
    """Compare only actual overlapping LSL clock samples from an operator-selected IMU."""
    result = {"state": "unavailable", "reason": "motion_reference_unavailable"}
    stamps = np.asarray(snapshot.get("timestamps", ()), dtype=float)
    raw = np.asarray(snapshot.get("samples", ()), dtype=float)
    eeg_times = np.asarray(eeg_times, dtype=float)
    if (snapshot.get("age") is None or snapshot["age"] > 3 or len(stamps) < 3 or len(eeg_times) < 2 or
            not np.isfinite(stamps).all() or not np.isfinite(eeg_times).all() or
            np.any(np.diff(eeg_times) <= 0) or np.any(np.diff(stamps) <= 0)):
        return result
    if raw.ndim != 2 or raw.shape[1] != len(stamps) or max(axes) >= raw.shape[0]:
        return result
    descriptor = snapshot.get("descriptor") or {}
    declared = [accel_unit(s) for s in descriptor.get("declared_units", ())]
    if any(i < len(declared) and declared[i] not in ("", "n/a", "unknown", unit) for i in axes):
        return {**result, "detail": "acceleration_unit_conflict"}
    # At least 5 samples/s; at both boundaries allow one sample at 5 Hz.
    include = (stamps >= eeg_times[0]) & (stamps <= eeg_times[-1])
    tt = stamps[include]
    if len(tt) < 3 or max(tt[0]-eeg_times[0], eeg_times[-1]-tt[-1], np.max(np.diff(tt))) > .21:
        return {**result, "detail": "motion_clock_coverage"}
    values = raw[np.array(axes)][:, include]
    if not np.isfinite(values).all():
        return result
    excursion = float(np.ptp(values, axis=1).max())
    active = excursion > ACCEL_LIMITS[unit]
    return {"state": "movement" if active else "below_threshold",
            "reason": "motion_reference_excursion" if active else "", "axis_peak_to_peak": excursion,
            "unit": unit, "samples": int(len(tt)), "axes": list(axes)}


def with_artifact_evidence(quality, evidence):
    if not evidence:
        return quality
    reasons = tuple(dict.fromkeys((*quality.reasons, *evidence["reasons"])))
    return Verdict(quality.ok and not reasons, reasons, {**quality.metrics, "artifact_control": evidence})


def channel_layout(labels, count):
    """Recognize only explicit reference labels. Ordinary Fp1/Fp2 always remain EEG."""
    keys = ["".join(c for c in str(s).lower() if c.isalnum()) for s in labels]
    keys = (keys + [""] * count)[:count]
    auxiliary = {i for i, k in enumerate(keys) if k in _EOG or k.startswith(
        ("eog", "emg", "accel", "gyro")) or k in ("accx", "accy", "accz")}
    return [i for i in range(count) if i not in auxiliary], keys


def voltage_scales(units, count, fallback, eeg=(), operator_eeg_scale=None):
    factors = []
    for i in range(count):
        declared = _unit(units[i]) if i < len(units) else None
        default = fallback
        if i in eeg and declared in (None, "ADC") and operator_eeg_scale is not None:
            default = operator_eeg_scale
        factors.append(UNIT_FACTORS.get(declared, default))
    return factors


def _smooth(x, samples):
    samples = max(1, min(samples, len(x)))
    pad = samples // 2
    padded = np.pad(x, (pad, samples - 1 - pad), mode="edge")
    return np.convolve(padded, np.ones(samples) / samples, mode="valid")


def _correct_eyes(eeg, raw_eeg, eog, rate, count):
    """Fit ocular propagation on the preceding history, apply to disjoint scoring samples."""
    start = eeg.shape[1] - count
    result = eeg[:, -count:].copy()
    changed = []
    if not eog or start < round(rate) or count < round(rate) or not np.isfinite(eeg).all():
        return result, changed
    train, target = [], []
    for raw_reference, filtered_reference in eog:
        if not np.isfinite(raw_reference).all() or not np.isfinite(filtered_reference).all():
            continue
        # The fitting interval is raw. Filtering the complete context zero-phase
        # before fitting would leak target samples across the boundary.
        earlier = _smooth(raw_reference[:start], round(.08 * rate))
        later = _smooth(filtered_reference[start:], round(.08 * rate))
        earlier -= earlier.mean()
        if np.std(earlier) >= 8.0:
            train.append(earlier)
            target.append(later - later.mean())
    if not train:
        return result, changed
    train, target = np.stack(train), np.stack(target)
    y = raw_eeg[:, :start] - raw_eeg[:, :start].mean(axis=1, keepdims=True)
    weights, _, rank, singular = np.linalg.lstsq(train.T, y.T, rcond=None)
    if rank != len(train) or singular[0] > 100 * singular[-1]:
        return result, changed
    explained = 1 - np.sum((y - weights.T @ train) ** 2, axis=1) / np.maximum(np.sum(y*y, axis=1), 1e-12)
    for i in range(len(eeg)):
        if explained[i] < .25:
            continue
        correction = weights[:, i] @ target
        candidate = result[i] - correction
        # Prevent unstable extrapolation and excessive subtraction on a different movement.
        if np.std(correction) > 1.5 * np.std(result[i]) or np.std(candidate) >= np.std(result[i]):
            continue
        result[i] = candidate
        changed.append(i)
    return result, changed


def _muscle_burst(row, rate, *, reference=False):
    size = round(rate / 2)
    if size < 8 or len(row) < size or rate < 100 or not np.isfinite(row).all():
        return False
    blocks = np.lib.stride_tricks.sliding_window_view(row, size)[::max(1, size // 2)]
    taper = np.hanning(size)
    power = np.abs(np.fft.rfft((blocks - blocks.mean(axis=1, keepdims=True)) * taper)) ** 2
    freq = np.fft.rfftfreq(size, 1 / rate)
    band = power[:, (freq >= 30) & (freq <= (90 if reference else 45))].sum(axis=1)
    total = power[:, (freq >= 3) & (freq <= (90 if reference else 45))].sum(axis=1)
    rms = np.sqrt(2 * band / (size * np.sum(taper*taper)))
    return bool(np.any((rms > (20 if reference else 8)) & (band > .55 * total)))


def process_live_artifacts(filtered, source, rate, expected, labels=(), units=(), fallback_scale=1.,
                           operator_eeg_scale=None, *, blink_evidence=None):
    """Return EEG-only window and per-window evidence. References share the EEG sample clock.

    filtered contains DC-removed, filtered source channels in µV for voltage channels.
    source contains untouched original device values (including any accelerometry).
    """
    pick, keys = channel_layout(labels, len(filtered))
    if not pick:
        raise ValueError("所选流只有辅助通道，无可用的 EEG 通道")
    count = min(expected, filtered.shape[1])
    if count < 1:
        raise ValueError("脑电窗长度必须大于零")
    units = list(units)
    known_voltage = {i for i, u in enumerate(units) if _unit(u) in UNIT_FACTORS}
    # Ambiguous duplicate reference labels cannot define a reproducible subtraction.
    mapping = {k: i for i, k in enumerate(keys) if keys.count(k) == 1 and i in known_voltage}
    scales = voltage_scales(units, len(filtered), fallback_scale, pick, operator_eeg_scale)
    uv = source * np.asarray(scales)[:, None]
    eog = [(uv[mapping[k]], filtered[mapping[k]]) for k in ("veog", "heog", "eog") if k in mapping]
    for plus, minus in (("eogup", "eogdown"), ("eogleft", "eogright")):
        if plus in mapping and minus in mapping:
            a, b = mapping[plus], mapping[minus]
            eog.append((uv[a] - uv[b], filtered[a] - filtered[b]))
    eeg, corrected = _correct_eyes(filtered[pick], uv[pick], eog, rate, count)
    reasons = list((blink_evidence or {}).get("reasons", ()))
    if not np.isfinite(eeg).all():
        reasons.append("non_finite_signal")
    else:
        for index, row in zip(pick, eeg):
            unaltered = uv[index, -count:]
            if len(unaltered) > 2:
                delta = np.diff(unaltered)
                noise = 1.4826 * np.median(np.abs(delta-np.median(delta)))
                if np.max(np.abs(delta)) > max(45., 10 * noise):
                    reasons.append("electrode_step")
            slow = _smooth(row, round(.12 * rate))
            centered = np.abs(slow - np.median(slow))
            if np.max(centered) > max(60., 5 * 1.4826 * np.median(centered)):
                reasons.append("ocular_transient")
            if _muscle_burst(row, rate):
                reasons.append("muscle_burst")
    for i, key in enumerate(keys):
        row = np.asarray(source[i, -count:])
        if key.startswith("emg") and i in known_voltage:
            if _muscle_burst(row * UNIT_FACTORS[_unit(units[i])], rate, reference=True):
                reasons.append("emg_reference_burst")
        if key in ("accx", "accy", "accz", "accelx", "accely", "accelz") and i < len(units):
            unit = accel_unit(units[i])
            limit = ACCEL_LIMITS.get(unit)
            if limit and np.ptp(row) > limit:
                reasons.append("motion_reference_excursion")
    info = {"spec": SPEC, "ocular_reference_axes": len(eog),
            "ocular_corrected_channels": [pick[i] for i in corrected],
            "eeg_source_indices": pick, "reasons": list(dict.fromkeys(reasons))}
    if blink_evidence:
        info["blink_control"] = blink_evidence
    return eeg, info
