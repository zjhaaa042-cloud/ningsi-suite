"""Past-only, finite-support EEG conditioning for capture and human models.

Each valid output sample uses the same two seconds of acquisition history,
regardless of chunking. Overlapping four-second windows agree sample for sample
before artifact masking. The masks are conservative EEG proxies, not diagnoses.
"""

from dataclasses import replace
from functools import lru_cache
import hashlib
from pathlib import Path
import sys

import numpy as np

from ningsi.models.logistic import features_from_window
from ningsi.models.research_features import condition_window, implementation_fingerprint
from .artifacts import Verdict
from .preprocess import _biquad, apply_biquad
from .window import WindowFeatures, analyze_window

POLICY = "causal-eeg-artifact-mask-v1"
FILE = "signal/conditioning.py"
EYE, MUSCLE, MOVEMENT, SIGNAL_FAULT, UNAVAILABLE = 1, 2, 4, 8, 16
MASK_BITS = {1: "ocular", 2: "muscle", 4: "movement", 8: "signal_fault", 16: "unavailable"}


def implementation(policy=None):
    hashes = implementation_fingerprint()
    if policy is None:
        return hashes
    if policy != POLICY:
        raise ValueError(f"未知的脑电预处理版本 {policy}")
    path = Path(__file__)
    if not path.is_file() or path.suffix != ".py":
        path = Path(sys._MEIPASS) / "capture_engine_source" / FILE
    hashes[FILE] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def description(rate):
    return {"policy": POLICY, "units": "uV", "causal": True, "history_sec": 2,
            "drift_highpass_hz": 0.8, "lowpass_hz": 42, "notch_q": 16,
            "notch_hz": [hz for hz in (50, 60) if hz < rate / 2],
            "lowpass_group_delay_samples": max(1, round(.18 * rate)),
            "time_alignment": "output clock; causal frequency-dependent delay; no event-time shift",
            "target_bands_hz": {"theta": [4, 8], "alpha": [8, 13], "beta": [13, 30]},
            "artifact_strategy": "mark contaminated samples and exclude the entire affected feature window",
            "mask_bits": MASK_BITS, "hardware_filter": "not assumed; acquired stream is the input"}


@lru_cache(maxsize=16)
def _kernel(rate):
    # Time-limit the causal biquad impulse, then taper its negligible tail.
    # Compact support makes a random-access export identical to live filtering.
    history = round(6 * rate) - round(4 * rate)
    h = np.zeros(history + 1)
    h[0] = 1
    h = apply_biquad(h, *_biquad("highpass", .8, rate))
    for hz in (50., 60.):
        if hz < rate / 2:
            h = apply_biquad(h, *_biquad("notch", hz, rate, q=16))
    half = max(1, round(.18 * rate))
    grid = np.arange(-half, half + 1)
    low = 84 / rate * np.sinc(84 / rate * grid) * np.kaiser(len(grid), 7.5)
    h = _convolve(h, low / low.sum(), "full")[:history + 1]
    tail = max(4, round(.25 * rate))
    h[-tail:] *= .5 + .5 * np.cos(np.linspace(0, np.pi, tail))
    weights = np.hanning(tail)
    h[-tail:] -= h.sum() * weights / weights.sum()  # exact DC rejection
    h.setflags(write=False)
    return h


def _convolve(x, h, mode):
    if len(x) * len(h) <= 2_000_000:
        return np.convolve(x, h, mode)
    length = len(x) + len(h) - 1
    nfft = 1 << (length - 1).bit_length()
    full = np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(h, nfft), nfft)[:length]
    return full if mode == "full" else full[len(h) - 1:len(x)]


def filter_uv(values, rate):
    """Input C x time in µV with complete history. Output has no padded ends."""
    h = _kernel(float(rate))
    return np.stack([_convolve(channel - channel[0], h, "valid") for channel in values])


def _runs(condition):
    edges = np.diff(np.r_[False, condition, False].astype(np.int8))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def _burst_mask(signal, raw, rate, bit, starts, width):
    freq = np.fft.rfftfreq(width, 1 / rate)
    taper = np.hanning(width)
    normalization = width * (taper @ taper)
    def segment_spectrum(channel):
        chunks = np.lib.stride_tricks.sliding_window_view(channel, width)[starts]
        centered = chunks - chunks.mean(axis=-1, keepdims=True)
        return np.abs(np.fft.rfft(centered * taper, axis=-1)) ** 2, np.ptp(chunks, axis=-1)

    power, span = segment_spectrum(signal)
    total = power[:, (freq >= .5) & (freq <= 45)].sum(axis=1)
    target = (freq >= .5) & (freq < 4) if bit == EYE else (freq >= 30) & (freq <= 45)
    part = power[:, target].sum(axis=1)
    rms = np.sqrt(2 * part / normalization)
    if bit == EYE:
        return (part > .60 * total) & (rms > 20) & (span > 75)
    hit = (part > .45 * total) & (rms > 9)
    if rate / 2 > 96:
        # High-band broadband bursts can disappear from the low-passed EEG.
        # Require energy across 66–94 Hz, away from the mains and its harmonics.
        raw_power, _ = segment_spectrum(raw)
        high = raw_power[:, (freq >= 66) & (freq <= 94)]
        energy = high.sum(axis=1)
        lower = raw_power[:, (freq >= 66) & (freq < 80)].sum(axis=1)
        high_rms = np.sqrt(2 * energy / normalization)
        hit |= ((high_rms > 12) & (lower > .2 * energy) & (lower < .8 * energy)
                & (high.max(axis=1) < .45 * energy))
    return hit


def _local_artifacts(wave, raw, rate):
    size = wave.shape[1]
    mask = np.zeros(size, dtype=np.uint8)
    def mark(first, last, bit):
        mask[max(0, first):min(size, last)] |= bit

    # Saturation or lost contact in ANY input channel. A plateau in the FIR
    # support contaminates output even after that channel starts moving again.
    history = raw.shape[1] - size
    for channel in raw:
        for first, last in _runs(np.diff(channel) == 0):
            if last - first + 1 >= max(3, round(.10 * rate)):
                mark(first - history, last + 1, SIGNAL_FAULT)

    # An abrupt deflection after suppression of the mains carrier (electrode
    # movement/step proxy). Allow for the impulse response around the transient.
    step = max(1, round(.020 * rate))
    jumps = np.any(np.abs(wave[:, step:] - wave[:, :-step]) > 75, axis=0)
    pad = round(.20 * rate)
    for first, last in _runs(jumps):
        mark(first - pad, last + step + pad, MOVEMENT)

    for seconds, bit in ((1., EYE), (.5, MUSCLE)):
        width, hop = round(seconds * rate), max(1, round(.125 * rate))
        starts = np.unique(np.r_[np.arange(0, size - width + 1, hop), size - width])
        hit = np.zeros(len(starts), dtype=bool)
        for signal, channel in zip(wave, raw):
            hit |= _burst_mask(signal, channel[-size:], rate, bit, starts, width)
        for first in starts[hit]:
            mark(first, first + width + (pad if bit == MUSCLE else 0), bit)
    return mask


def _condition(values, stamps, rate, factor, traces):
    size = round(4 * rate)
    width = round(6 * rate)
    values = np.asarray(values, dtype=float)[:, -width:]
    stamps = np.asarray(stamps, dtype=float)[-width:]
    start = float(stamps[-size]) if len(stamps) >= size and np.isfinite(stamps[-size]) else None
    end = float(stamps[-1] + 1 / rate) if len(stamps) and np.isfinite(stamps[-1]) else None
    reasons = []
    factor = float(factor) if factor is not None else None
    if factor is None or not np.isfinite(factor) or factor <= 0:
        reasons.append("unit_unconfirmed")
    if rate < 100:
        reasons.append("sample_rate_below_100hz")
    if values.shape[1] != len(stamps) or len(stamps) < size:
        reasons.append("signal_too_short")
    elif len(stamps) < width:
        reasons.append("preprocessing_warmup")
    if not np.isfinite(values).all():
        reasons.append("non_finite_signal")
    if not np.isfinite(stamps).all():
        reasons.append("clock_unsynchronized")
    elif len(stamps) >= size and size > 1:
        diffs = np.diff(stamps)
        if np.any(diffs <= 0):
            reasons.append("nonmonotonic_timestamps")
        if np.any(diffs > .1):
            reasons.append("timestamp_gap_over_100ms")
        elif np.any(diffs > 1.75 / rate):
            reasons.append("sample_gap_in_filter_history")
        if abs((stamps[-1] - stamps[-size]) * rate / (size - 1) - 1) > .05:
            reasons.append("sample_rate_mismatch_over_5pct")
    wave = mask = None
    if reasons:
        window = WindowFeatures(end or 0., rate, values.shape[0], {}, {}, Verdict(False, tuple(reasons), {}), {})
    else:
        raw = values * factor
        wave = filter_uv(raw, rate)
        mask = _local_artifacts(wave, raw, rate)
        window = analyze_window(wave, rate, t_end=end)
        reasons = list(window.quality.reasons)
        for bit, reason in ((EYE, "ocular_transient"), (MUSCLE, "muscle_burst"),
                            (MOVEMENT, "motion_transient"), (SIGNAL_FAULT, "contact_dropout")):
            if np.any(mask & bit):
                reasons.append(reason)
        for reason, bit in (("eog", EYE), ("emg", MUSCLE), ("flat_channel", SIGNAL_FAULT)):
            if reason in reasons and not np.any(mask & bit):
                mask |= bit
        if reasons and not mask.any():
            mask[:] = SIGNAL_FAULT
        window = replace(window, quality=Verdict(not reasons, tuple(reasons), window.quality.metrics))
    row = {"start": start, "end": end, "window": window,
           "features": features_from_window(window) if window.usable else {}}
    if traces:
        row.update(filtered_uv=wave, artifact_mask=mask)
    return row


def condition(values, stamps, rate, factor, policy, *, traces=False):
    if policy is None:
        return condition_window(values, stamps, rate, factor)
    if policy != POLICY:
        raise ValueError(f"未知的脑电预处理版本 {policy}")
    return _condition(values, stamps, rate, factor, traces)
