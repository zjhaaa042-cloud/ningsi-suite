"""Conservative, personally calibrated blink control for two frontal EEG channels.

All waveforms are C x time in microvolts. Calibration uses *marked blinks*, never
eyes-open/closed state labels or an inferred attention state. A complete acquired
window is examined: event fitting needs up to 0.58 s of already received samples
after a peak. Incomplete edge events are masked, not extrapolated. This is an
engineering artifact control, not separation of EEG and EOG sources.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


SPEC = "dual-frontal-blink-v1"
MIN_EVENTS = 8
_HALF_SEC = .45
_WARPS = (.8, .9, 1., 1.1, 1.2)
_TEMPLATE_SIGMA_SEC = .065


def _rate(value):
    value = float(value)
    if not math.isfinite(value) or not 100 <= value <= 2000:
        raise ValueError("眨眼控制要求已确认的 100–2000 Hz 采样率")
    return value


def _labels(value):
    labels = tuple(value)
    if (len(labels) != 2 or any(not isinstance(s, str) or not s.strip() or len(s) > 128 for s in labels)
            or labels[0].casefold() == labels[1].casefold()):
        raise ValueError("眨眼控制要求两个不同的 EEG 通道名称")
    return labels


def _smooth(row, size):
    size = max(1, min(int(size), len(row)))
    left = size // 2
    return np.convolve(np.pad(row, (left, size - 1 - left), mode="edge"),
                       np.ones(size) / size, mode="valid")


def _scale(values):
    return 1.4826 * np.median(np.abs(values - np.median(values)))


def _detrend(values, rate):
    edge = max(2, round(.04 * rate))
    return values - np.linspace(np.median(values[:edge]), np.median(values[-edge:]), len(values))


def _smooth_template(template, rate):
    # Median averaging alone can retain phase-locked/random EEG from calibration.
    # A Gaussian prior with sigma=65 ms attenuates 6 Hz by ~20x (exp(-2*pi²*f²*s²)).
    # Fit amplitude/width afresh at runtime; a sharp blink that this smooth prior
    # cannot describe is masked. This is deliberately not a general EEG lowpass.
    radius = round(4 * _TEMPLATE_SIGMA_SEC * rate)
    coordinates = np.arange(-radius, radius+1) / rate
    kernel = np.exp(-.5 * (coordinates / _TEMPLATE_SIGMA_SEC) ** 2)
    kernel /= kernel.sum()
    result = np.stack([np.convolve(row, kernel, mode="same") for row in template])
    result -= np.linspace(result[:, 0], result[:, -1], result.shape[1]).T
    return result / np.max(np.abs(result), axis=1)[:, None]


def _runs(condition):
    edges = np.diff(np.r_[False, condition, False].astype(np.int8))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


@dataclass(frozen=True)
class BlinkCalibration:
    rate: float
    channel_labels: tuple[str, str]
    montage_id: str
    templates: tuple[tuple[float, ...], tuple[float, ...]]
    noise_uv: tuple[float, float]
    amplitudes_uv: tuple[float, float]
    ratio_limits: tuple[float, float]
    events: int

    def __post_init__(self):
        # Make nested input immutable, including instances constructed without from_dict.
        object.__setattr__(self, "rate", _rate(self.rate))
        object.__setattr__(self, "channel_labels", _labels(self.channel_labels))
        if not isinstance(self.montage_id, str) or len(self.montage_id) > 512:
            raise ValueError("无效的设备／参考电极身份")
        shape = (2, 2 * round(_HALF_SEC * self.rate) + 1)
        array = np.asarray(self.templates, dtype=float)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError("眨眼模板的长度或数值无效")
        if np.any(np.abs(np.max(np.abs(array), axis=1) - 1) > .02):
            raise ValueError("眨眼模板必须逐通道归一化")
        if np.any(np.abs(array[:, [0, -1]]) > .01):
            raise ValueError("眨眼模板边缘必须归零")
        for name in ("noise_uv", "amplitudes_uv", "ratio_limits"):
            pair = tuple(float(v) for v in getattr(self, name))
            if len(pair) != 2 or not all(math.isfinite(v) and v > 0 for v in pair):
                raise ValueError("无效的眨眼校准统计量")
            object.__setattr__(self, name, pair)
        if (self.ratio_limits[0] >= self.ratio_limits[1] or
                any(v < 35 or v > 800 for v in self.amplitudes_uv)):
            raise ValueError("无效的眨眼幅度关系")
        if not isinstance(self.events, int) or not MIN_EVENTS <= self.events <= 1000:
            raise ValueError("有效眨眼校准事件不足")
        object.__setattr__(self, "templates", tuple(tuple(float(v) for v in row) for row in array))

    def as_dict(self):
        return {"spec": SPEC, "rate": self.rate, "channel_labels": list(self.channel_labels),
                "montage_id": self.montage_id, "templates": [list(row) for row in self.templates],
                "noise_uv": list(self.noise_uv), "amplitudes_uv": list(self.amplitudes_uv),
                "ratio_limits": list(self.ratio_limits), "events": self.events}

    @classmethod
    def from_dict(cls, payload):
        fields = {"spec", "rate", "channel_labels", "montage_id", "templates", "noise_uv",
                  "amplitudes_uv", "ratio_limits", "events"}
        if not isinstance(payload, dict) or set(payload) != fields or payload["spec"] != SPEC:
            raise ValueError("眨眼校准版本或字段不匹配")
        try:
            return cls(**{k: v for k, v in payload.items() if k != "spec"})
        except (TypeError, OverflowError) as exc:
            raise ValueError("眨眼校准格式无效") from exc


@dataclass(frozen=True)
class BlinkResult:
    cleaned_uv: np.ndarray
    mask: np.ndarray
    corrected_mask: np.ndarray
    events: tuple[dict, ...]
    reasons: tuple[str, ...]

    def as_dict(self, *, start_index=0):
        """Describe a scoring suffix while retaining event indices in the full context."""
        if type(start_index) is not int or not 0 <= start_index <= self.mask.size:
            raise ValueError("眨眼证据的评分起点超出已处理数据")
        mask, corrected = self.mask[start_index:], self.corrected_mask[start_index:]
        events = [event for event in self.events if event["stop_index"] > start_index]
        # A historical event can remain in the filter warmup without vetoing a
        # later scoring window. Whole-context signal/configuration faults have
        # no individual events and still retain their full masks and reasons.
        active = {event["reason"] for event in events if event["state"] == "masked"}
        reasons = [reason for reason in self.reasons if not self.events or reason in active]
        return {"spec": SPEC, "state": "rejected" if mask.any() else
                "corrected" if corrected.any() else "clean",
                "reasons": reasons, "events": events,
                "context_samples": int(self.mask.size), "analysis_start_index": start_index,
                "corrected_events": sum(e["state"] == "corrected" for e in events),
                "masked_fraction": float(np.mean(mask)) if mask.size else 1.,
                "corrected_fraction": float(np.mean(corrected)) if corrected.size else 0.,
                "latency_note": "完整历史窗；事件峰后最多约 0.58 秒，边缘事件不外推"}


def _fault(values, rate):
    if not np.isfinite(values).all():
        return "non_finite_signal"
    for row in values:
        if np.max(np.abs(row - np.median(row))) > 1000:
            return "blink_signal_fault"
        delta = np.diff(row)
        if len(delta) and np.max(np.abs(delta)) > max(45., 10 * _scale(delta)):
            return "electrode_step"
        if any(last - first + 1 >= max(3, round(.08 * rate)) for first, last in _runs(delta == 0)):
            return "contact_dropout"
    return None


def calibrate_blinks(values_uv, rate, peak_indices, *, channel_labels=("Fp1", "Fp2"), montage_id=""):
    """Fit separate templates from >=8 explicitly marked, well-separated blinks.

    Supply 15–20 natural short blinks several seconds apart when possible. Raw
    recording is retained by the caller. Bad/single-sided/inconsistent marked
    events are excluded; fewer than eight accepted events raises ValueError.
    """
    rate, labels = _rate(rate), _labels(channel_labels)
    data = np.asarray(values_uv, dtype=float)
    if data.ndim != 2 or data.shape[0] != 2 or data.shape[1] < round(12 * rate):
        raise ValueError("需至少 12 秒双通道原始微伏信号及独立眨眼标记")
    if _fault(data, rate):
        raise ValueError("校准信号存在非有限值、饱和、接触异常或阶跃，请重新采集")
    peaks = tuple(peak_indices)
    if len(peaks) > 1000 or any(isinstance(p, bool) or not isinstance(p, (int, np.integer)) for p in peaks):
        raise ValueError("眨眼标记必须是有界的整数采样点")
    peaks = np.asarray(sorted(peaks), dtype=int)
    half, pad = round(_HALF_SEC * rate), round(.1 * rate)
    if (len(peaks) < MIN_EVENTS or np.any(peaks < half + pad) or
            np.any(peaks + half + pad >= data.shape[1]) or np.any(np.diff(peaks) < round(1.25 * rate))):
        raise ValueError("需至少八次完整且间隔至少 1.25 秒的眨眼，避开记录边缘")
    slow = np.stack([_smooth(row, round(.08 * rate)) for row in data])
    background = np.ones(data.shape[1], dtype=bool)
    for p in peaks:
        background[p-half-pad:p+half+pad+1] = False
    if np.sum(background) < round(4 * rate):
        raise ValueError("需至少四秒无眨眼背景数据")
    noise = np.array([max(1., _scale(row[background])) for row in slow])
    segments, amps = [], []
    for mark in peaks:
        # Find each channel's peak in a bounded neighborhood of the actual marker.
        left, right = mark-pad, mark+pad+1
        centered = slow[:, left:right] - np.median(slow, axis=1)[:, None]
        locations = np.argmax(np.abs(centered), axis=1) + left
        if abs(locations[0] - locations[1]) > round(.08 * rate):
            continue
        peak = round(float(np.mean(locations)))
        segment = np.stack([_detrend(row[peak-half:peak+half+1], rate) for row in data])
        segment = np.stack([_smooth(row, round(.04 * rate)) for row in segment])
        amplitude = np.max(np.abs(segment), axis=1)
        if np.any(amplitude < np.maximum(40., 6 * noise)) or np.any(amplitude > 800):
            continue
        # Peaks must be short transients, not sustained eye closure/contact drift.
        widths = np.sum(np.abs(segment) > amplitude[:, None] * .5, axis=1) / rate
        if np.any(widths < .08) or np.any(widths > .40):
            continue
        taper = np.ones(2 * half + 1)
        edge = max(2, round(.08 * rate))
        taper[:edge] = np.sin(np.linspace(0, np.pi / 2, edge)) ** 2
        taper[-edge:] = taper[:edge][::-1]
        segments.append(segment / amplitude[:, None] * taper)
        amps.append(amplitude)
    if len(segments) < MIN_EVENTS:
        raise ValueError("可信双通道短眨眼不足八次，请重新校准")
    segments, amps = np.stack(segments), np.stack(amps)
    template = np.median(segments, axis=0)
    template /= np.max(np.abs(template), axis=1)[:, None]
    similarity = np.sum(segments * template, axis=2) / np.sqrt(
        np.maximum(np.sum(segments ** 2, axis=2) * np.sum(template ** 2, axis=1), 1e-12))
    ratios = amps[:, 0] / amps[:, 1]
    log_ratio = np.log(ratios)
    keep = np.all(similarity > .9, axis=1) & (np.abs(log_ratio - np.median(log_ratio)) <= max(.20, 3 * _scale(log_ratio)))
    if np.sum(keep) < MIN_EVENTS:
        raise ValueError("双通道眨眼形状或幅度关系不稳定，可信事件不足八次")
    template = np.median(segments[keep], axis=0)
    template /= np.max(np.abs(template), axis=1)[:, None]
    template = _smooth_template(template, rate)
    center = float(np.median(log_ratio[keep]))
    spread = max(.25, 4 * _scale(log_ratio[keep]))
    return BlinkCalibration(rate, labels, montage_id, tuple(map(tuple, template)), tuple(noise),
                            tuple(np.median(amps[keep], axis=0)),
                            (math.exp(center-spread), math.exp(center+spread)), int(np.sum(keep)))


def _candidates(data, rate, calibration):
    slow = np.stack([_smooth(row, round(.08 * rate)) for row in data])
    slow -= np.median(slow, axis=1)[:, None]
    noise = np.array([_scale(row) for row in slow])
    if calibration:
        # A dense artifact window must not inflate its own threshold above the
        # personal clean baseline. Ordinary EEG up to the absolute floor remains.
        noise = np.minimum(noise, np.asarray(calibration.noise_uv) * 1.5)
    thresholds = np.maximum(35., 6 * noise)
    score = np.max(np.abs(slow) / thresholds[:, None], axis=0)
    peaks = []
    for first, last in _runs(score > 1):
        p = first + int(np.argmax(score[first:last]))
        if peaks and p - peaks[-1] < round(.25 * rate):
            if score[p] > score[peaks[-1]]:
                peaks[-1] = p
        else:
            peaks.append(p)
    return peaks, slow


def _fit(data, peak, rate, profile):
    half = round(_HALF_SEC * rate)
    # Include the entire shifted/warped template. Truncating at the unshifted
    # radius could introduce a subtraction step at an otherwise clean boundary.
    radius = math.ceil(half * max(_WARPS)) + round(.04 * rate)
    start, stop = peak-radius, peak+radius+1
    segment = np.stack([_detrend(row[start:stop], rate) for row in data])
    smooth = np.stack([_smooth(row, round(.08 * rate)) for row in segment])
    coordinates = np.arange(-radius, radius+1)
    templates = np.asarray(profile.templates)
    best = None
    for shift in (-round(.04*rate), -round(.02*rate), 0, round(.02*rate), round(.04*rate)):
        channel_best = []
        for c in range(2):
            fits = []
            for warp in _WARPS:
                shape = np.interp((coordinates-shift) / warp, np.arange(-half, half+1), templates[c], left=0., right=0.)
                low_shape = _smooth(shape, round(.08 * rate))
                amplitude = float(smooth[c] @ low_shape / max(low_shape @ low_shape, 1e-12))
                residual = smooth[c] - amplitude * low_shape
                error = float(residual @ residual / max(smooth[c] @ smooth[c], 1e-12))
                fits.append((error, amplitude, warp, shape))
            channel_best.append(min(fits, key=lambda item: item[0]))
        score = max(f[0] for f in channel_best)
        if best is None or score < best[0]:
            best = (score, channel_best)
    error, fits = best
    amplitudes = np.array([item[1] for item in fits])
    if (error > .15 or np.any(amplitudes < np.maximum(35., np.asarray(profile.amplitudes_uv)*.3)) or
            np.any(amplitudes > np.asarray(profile.amplitudes_uv)*2.5)):
        return None, "blink_shape_mismatch"
    if not profile.ratio_limits[0] <= amplitudes[0] / amplitudes[1] <= profile.ratio_limits[1]:
        return None, "blink_channel_mismatch"
    local_peaks = np.argmax(np.abs(smooth), axis=1)
    if abs(local_peaks[0] - local_peaks[1]) > round(.08 * rate):
        return None, "blink_channel_mismatch"
    correction = np.stack([amp * item[3] for amp, item in zip(amplitudes, fits)])
    candidate = segment - correction
    residual = np.stack([_smooth(row, round(.08 * rate)) for row in candidate])
    if (np.any(np.max(np.abs(residual), axis=1) > np.maximum(25., amplitudes*.25)) or
            np.any(np.std(candidate, axis=1) >= np.std(segment, axis=1))):
        return None, "blink_residual"
    return (start, stop, correction, amplitudes, [f[2] for f in fits], error), ""


def process_blinks(values_uv, rate, calibration=None, *, channel_labels=("Fp1", "Fp2"), montage_id=""):
    """Mask untrusted transients and locally correct only personally matched ones.

    The original input is never mutated. No model/profile is adapted during task
    data. A mismatch masks the window rather than applying another device's
    template. Other quality checks must still run on original and corrected data.
    """
    rate, labels = _rate(rate), _labels(channel_labels)
    data = np.asarray(values_uv, dtype=float)
    if data.ndim != 2 or data.shape[0] != 2 or not data.shape[1]:
        raise ValueError("眨眼控制要求形状为 (2, 采样点) 的原始微伏信号")
    output = data.copy()
    mask, corrected = np.zeros(data.shape[1], dtype=bool), np.zeros(data.shape[1], dtype=bool)
    if calibration is not None and not isinstance(calibration, BlinkCalibration):
        calibration = BlinkCalibration.from_dict(calibration)
    fault = _fault(data, rate)
    mismatch = calibration is not None and (calibration.rate != rate or
        calibration.channel_labels != labels or calibration.montage_id != montage_id)
    if fault or mismatch:
        mask[:] = True
        return BlinkResult(output, mask, corrected, (), (fault or "blink_calibration_mismatch",))
    peaks, slow = _candidates(data, rate, calibration)
    radius = math.ceil(round(_HALF_SEC * rate) * max(_WARPS)) + round(.04*rate)
    intervals = [(max(0, p-radius), min(data.shape[1], p+radius+1)) for p in peaks]
    occupied = np.zeros(data.shape[1], dtype=bool)
    for start, stop in intervals:
        occupied[start:stop] = True
    dense = np.mean(occupied) > .6 or any(b-a < 2*radius for a, b in zip(peaks, peaks[1:]))
    events, reasons = [], []
    for p, (start, stop) in zip(peaks, intervals):
        reason = ("blink_boundary" if p-radius < 0 or p+radius >= data.shape[1] else
                  "blink_dense" if dense else "blink_uncalibrated" if calibration is None else "")
        fitted = None
        if not reason:
            fitted, reason = _fit(data, p, rate, calibration)
        event = {"peak_index": int(p), "start_index": int(start), "stop_index": int(stop),
                 "state": "masked", "reason": reason}
        if fitted is None:
            mask[start:stop] = True
            reasons.append(reason)
        else:
            first, last, correction, amplitudes, warps, error = fitted
            output[:, first:last] -= correction
            corrected[first:last] = np.any(correction != 0., axis=0)
            event.update(state="corrected", amplitudes_uv=amplitudes.tolist(), width_scales=warps,
                         fit_error=float(error), reason="")
        events.append(event)
    return BlinkResult(output, mask, corrected, tuple(events), tuple(dict.fromkeys(reasons)))
