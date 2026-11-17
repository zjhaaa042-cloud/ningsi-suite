"""Bounded, read-only EEG analysis for the competition workbench.

This service neither opens devices nor reads recordings. Demo signals are
explicitly synthetic. Imported matrices stay imported/unverified data, even
when they pass the signal-quality proxies. Frozen model paths are unchanged.
"""

from __future__ import annotations

from collections import Counter
import math
import time

import numpy as np

from ningsi import config
from ningsi.acquisition.simulate import SyntheticEEG
from ningsi.models.research_features import union_seconds
from ningsi.signal import conditioning
from ningsi.signal.bands import band_power
from ningsi.signal.baseline import build_baseline
from ningsi.signal.indicators import compute_indicators
from ningsi.signal.spectrum import welch_psd


SCHEMA = "ningsi-showcase-analysis-v1"
MAX_CHANNELS = 8
MAX_DURATION = 120
MAX_VALUES = 240_000
MAX_TRACE_POINTS = 1000
ARTIFACTS = ("clean", "blink", "muscle", "movement", "mixed")
STATES = ("rest", "focused", "drowsy", "loaded")


def metadata() -> dict:
    """Return capabilities and their interpretation without starting acquisition."""
    return {
        "schema": SCHEMA,
        "processing": conditioning.description(250),
        "spectrum_spec": config.SPECTRUM_SPEC,
        "indicator_spec": config.INDICATOR_SPEC,
        "bands_hz": {name: list(config.BANDS[name]) for name in ("theta", "alpha", "beta")},
        "features": ["absolute_band_power_uv2", "relative_band_power", "log_band_ratios",
                     "rms_uv", "ptp_uv", "channel_correlation"],
        "artifacts": list(ARTIFACTS),
        "states": list(STATES),
        "input_limits": {"channels": [1, MAX_CHANNELS], "rate_hz": [100, 1000],
                         "duration_sec": [6, MAX_DURATION], "total_values": MAX_VALUES},
        "baseline": {"independent_record_required": True,
                     "minimum_usable_sec": config.BASELINE_PROTOCOL["min_usable_sec"],
                     "minimum_windows": config.BASELINE_PROTOCOL["min_windows"]},
        "limitations": [
            "演示信号为明确生成的仿真数据，不代表真人识别精度。",
            "导入数据的设备合规、佩戴位置、电压标定及来源真实性需要另行核实。",
            "无源时间戳时按声明采样率生成回放时标，不等于设备时钟验证。",
            "处理后曲线将疑似污染样点留空，含污染的整窗退出评分；留空不代表恢复出无伪迹脑电。",
            "眼动、肌电和运动标记为信号代理，仍需独立标注验证漏检与误拒。",
            "仅在独立基线通过现有门槛后输出相对基线指数；指数不是专注概率。",
        ],
    }


def _number(value, label, low, high) -> float:
    try:
        number = float(value) if type(value) in (int, float) else float("nan")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label}须为 {low} 至 {high} 的有限数值。") from exc
    if not math.isfinite(number) or not low <= number <= high:
        raise ValueError(f"{label}须为 {low} 至 {high} 的有限数值。")
    return number


def _labels(value, count):
    if value is None:
        return [f"Ch{i + 1}" for i in range(count)]
    if (not isinstance(value, list) or len(value) != count
            or any(not isinstance(label, str) or not label.strip() or len(label) > 40 for label in value)
            or len(set(value)) != count):
        raise ValueError("channel_labels 须与通道数一致，名称非空且不重复。")
    return list(value)


def _matrix(value, rate, label, *, channels=None):
    if (not isinstance(value, list) or not 1 <= len(value) <= MAX_CHANNELS
            or any(not isinstance(row, list) for row in value)):
        raise ValueError(f"{label}须为 1 至 {MAX_CHANNELS} 通道的二维数值数组。")
    width = len(value[0])
    if (any(len(row) != width for row in value) or len(value) * width > MAX_VALUES
            or not round(6 * rate) <= width <= round(MAX_DURATION * rate)):
        raise ValueError(f"{label}各通道须等长，包含 6 至 {MAX_DURATION} 秒，且总数值不超过 {MAX_VALUES}。")
    if channels is not None and len(value) != channels:
        raise ValueError("独立基线与任务的通道数须一致。")
    if any(type(item) not in (int, float) for row in value for item in row):
        raise ValueError(f"{label}只能包含有限数值。")
    try:
        data = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label}含非有限或超范围数值。") from exc
    if not np.isfinite(data).all() or np.max(np.abs(data)) > 1e9:
        raise ValueError(f"{label}含非有限或超范围数值。")
    return data


def _timestamps(value, count, rate, label):
    if value is None:
        return np.arange(count, dtype=float) / rate
    if (not isinstance(value, list) or len(value) != count
            or any(type(item) not in (int, float) for item in value)):
        raise ValueError(f"{label}须与样点一一对应，并全部为有限时间戳。")
    try:
        stamps = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label}须与样点一一对应，并全部为有限时间戳。") from exc
    if not np.isfinite(stamps).all() or (np.abs(stamps) > 1e12).any():
        raise ValueError(f"{label}须与样点一一对应，并全部为有限时间戳。")
    # Nonmonotonic and discontinuous clocks remain observable quality failures.
    return stamps


def _demo(rate, channels, seconds, state, artifact):
    data = SyntheticEEG(srate=rate, channels=channels, noise_uv=1.5, seed=42).window(state, seconds=seconds)
    t = np.arange(data.shape[1]) / rate
    # Drift and mains are deliberately present, including in the clean preset.
    # "clean" means no injected transient artifact, not an acquired clean truth.
    noise = 35 * np.sin(2 * np.pi * .15 * t) + 35 * np.sin(2 * np.pi * 50 * t)
    if rate / 2 > 60:
        noise += 18 * np.sin(2 * np.pi * 60 * t)
    data += 400 + noise
    center = seconds - 2.0
    if artifact in ("blink", "mixed"):
        data += 210 * np.exp(-((t - center) / .16) ** 2)
    if artifact in ("muscle", "mixed"):
        burst = 55 * np.sin(2 * np.pi * 38 * t) * ((t > center - .55) & (t < center + .45))
        if rate / 2 > 96:
            burst += (25 * np.sin(2 * np.pi * 71 * t) + 27 * np.sin(2 * np.pi * 87 * t)) * (
                (t > center - .55) & (t < center + .45))
        data += burst
    if artifact in ("movement", "mixed"):
        data += 420 * (t >= center)
    return data


def _windows(values, stamps, rate, factor):
    width, hop = round(6 * rate), round(2 * rate)
    rows = []
    for end in range(width, values.shape[1] + 1, hop):
        row = conditioning.condition(values[:, end - width:end], stamps[end - width:end], rate,
                                     factor, conditioning.POLICY, traces=True)
        row["sample_end_index"] = end
        rows.append(row)
    # Include the newest full window even if the supplied record has a remainder.
    if rows[-1]["sample_end_index"] != values.shape[1]:
        end = values.shape[1]
        row = conditioning.condition(values[:, end - width:end], stamps[end - width:end], rate,
                                     factor, conditioning.POLICY, traces=True)
        row["sample_end_index"] = end
        rows.append(row)
    return rows


def _spectrum(values, rate):
    spectra = [welch_psd(channel, rate, fmax=min(100., rate / 2)) for channel in values]
    return spectra[0].freqs, np.mean([item.psd for item in spectra], axis=0)


def _array(values, decimals=6):
    return [None if not np.isfinite(value) else round(float(value), decimals) for value in values]


def _intervals(mask, rate):
    """Return constant-mask spans so overlapping artifact types stay explicit."""
    edges = np.r_[0, np.flatnonzero(np.diff(mask)) + 1, len(mask)]
    return [{"start_sec": round(first / rate, 6), "end_sec": round(last / rate, 6),
             "reasons": [name for bit, name in conditioning.MASK_BITS.items() if int(mask[first]) & bit]}
            for first, last in zip(edges[:-1], edges[1:]) if mask[first]]


def _line_fraction(freqs, psd):
    total = band_power(freqs, psd, 1, min(100., float(freqs[-1]) + 1e-6))
    return {str(hz): (band_power(freqs, psd, hz - 1, hz + 1) / total
                     if total > 0 and hz + 1 <= freqs[-1] else None) for hz in (50, 60)}


def build_analysis(payload) -> dict:
    """Analyze a bounded JSON matrix or an explicitly selected synthetic example.

    ``baseline_samples`` is an independent same-profile recording. It is never
    inferred from the task windows. Without it, imported data has no scores.
    Missing source timestamps are labelled as generated nominal-rate playback.
    """
    started = time.perf_counter()
    if not isinstance(payload, dict) or payload.get("mode") not in ("demo", "raw"):
        raise ValueError("请明确选择 mode=demo（仿真演示）或 mode=raw（导入分析）。")
    demo = payload["mode"] == "demo"
    rate = _number(payload.get("rate", 250 if demo else None), "rate", 100, 1000)
    if demo:
        count = _number(payload.get("channels", 2), "channels", 1, MAX_CHANNELS)
        if count != int(count):
            raise ValueError("channels 须为整数。")
        count = int(count)
        seconds = _number(payload.get("duration_sec", 18), "duration_sec", 6, 60)
        if round(seconds * rate) * count > MAX_VALUES:
            raise ValueError("仿真样点总数超过限制。")
        state, artifact = payload.get("state", "focused"), payload.get("artifact", "mixed")
        if state not in STATES or artifact not in ARTIFACTS:
            raise ValueError("未知仿真状态或伪迹预设。")
        values = _demo(rate, count, seconds, state, artifact)
        stamps = _timestamps(None, values.shape[1], rate, "timestamps")
        factor = 1.
        baseline_values = _demo(rate, count, 36, "rest", "clean")
        baseline_stamps = _timestamps(None, baseline_values.shape[1], rate, "baseline_timestamps")
        labels = _labels(payload.get("channel_labels"), count)
        source = {"kind": "sim", "label": "明确选择的仿真 EEG 演示", "simulated": True,
                  "real_hardware_verified": False, "artifact_preset": artifact, "state_preset": state,
                  "timestamp_basis": "synthetic_nominal_rate", "clock_verified": False,
                  "baseline_source": "separately_generated_synthetic_rest"}
    else:
        factor = _number(payload.get("scale_to_uv"), "scale_to_uv", 1e-9, 1e9)
        values = _matrix(payload.get("samples"), rate, "samples")
        if np.max(np.abs(values)) * factor > 1e9:
            raise ValueError("换算后电压超过分析范围，请核对源单位。")
        count = values.shape[0]
        labels = _labels(payload.get("channel_labels"), count)
        stamps = _timestamps(payload.get("timestamps"), values.shape[1], rate, "timestamps")
        baseline_values = None
        baseline_stamps = None
        if payload.get("baseline_samples") is not None:
            baseline_values = _matrix(payload["baseline_samples"], rate, "baseline_samples", channels=count)
            if np.max(np.abs(baseline_values)) * factor > 1e9:
                raise ValueError("独立基线换算后电压超过分析范围。")
            baseline_stamps = _timestamps(payload.get("baseline_timestamps"), baseline_values.shape[1], rate,
                                          "baseline_timestamps")
        source = {"kind": "imported", "label": "导入矩阵的离线分析（来源待核实）", "simulated": False,
                  "real_hardware_verified": False,
                  "timestamp_basis": "provided_source_timestamps" if payload.get("timestamps") is not None
                  else "nominal_rate_playback_not_device_clock",
                  "clock_verified": False, "baseline_source": "independent_input" if baseline_values is not None else None}

    rows = _windows(values, stamps, rate, factor)
    baseline = None
    if baseline_values is not None:
        baseline_rows = _windows(baseline_values, baseline_stamps, rate, factor)
        baseline = build_baseline([row["window"] for row in baseline_rows], device="showcase-input", srate=rate)
    baseline_public = baseline.as_dict() if baseline is not None else {
        "valid": False, "reasons": ["independent_baseline_missing"], "n_windows": 0, "quality": {}}
    baseline_public["source"] = source["baseline_source"]
    windows = []
    reasons = Counter()
    for row in rows:
        window = row["window"]
        indicators = compute_indicators(window, baseline)
        reasons.update(window.quality.reasons)
        # Display time is sample-relative; original clock remains separate metadata.
        end_sec = row["sample_end_index"] / rate
        windows.append({"start_sec": round(end_sec - 4, 6), "end_sec": round(end_sec, 6),
                        "usable": window.usable, "reasons": list(window.quality.reasons),
                        "scores": dict(indicators.scores), "indicator_reason": indicators.reason,
                        "relative_band_powers": dict(window.rel), "time_domain": dict(window.time_domain)})
    latest_row = rows[-1]
    latest_window = latest_row["window"]
    last_end = latest_row["sample_end_index"]
    size = round(4 * rate)
    raw_uv = values[:, last_end - size:last_end] * factor
    filtered = latest_row["filtered_uv"]
    mask = latest_row["artifact_mask"]
    if filtered is None:
        filtered = np.full(raw_uv.shape, np.nan)
        mask = np.full(size, conditioning.UNAVAILABLE, dtype=np.uint8)
    masked = filtered.copy()
    masked[:, mask != 0] = np.nan
    # Select endpoints and per-bin extrema/mask coverage together. At normal
    # 250 Hz no decimation occurs; high-rate display never hides a rejected bin.
    stride = max(1, math.ceil(size / MAX_TRACE_POINTS))
    selected = np.arange(0, size, stride)
    display_mask = np.array([np.bitwise_or.reduce(mask[i:min(i + stride, size)]) for i in selected])
    trace_channels = []
    for label, raw, wave, clean in zip(labels, raw_uv, filtered, masked):
        raw_display = np.array([raw[i:min(i + stride, size)][np.argmax(np.abs(raw[i:min(i + stride, size)]))]
                                for i in selected])
        wave_display = np.array([wave[i:min(i + stride, size)][np.argmax(np.abs(wave[i:min(i + stride, size)]))]
                                 for i in selected])
        clean_display = clean[selected].copy()
        clean_display[display_mask != 0] = np.nan
        trace_channels.append({"label": label, "raw_uv": _array(raw_display),
                               "filtered_uv": _array(wave_display), "cleaned_uv": _array(clean_display)})
    raw_f, raw_p = _spectrum(raw_uv, rate)
    if np.isfinite(filtered).all():
        filtered_f, filtered_p = _spectrum(filtered, rate)
        assert np.array_equal(raw_f, filtered_f)
    else:
        filtered_p = np.full_like(raw_p, np.nan)
    bands = [{"name": name, "low_hz": config.BANDS[name][0], "high_hz": config.BANDS[name][1],
              "raw_power_uv2": round(band_power(raw_f, raw_p, *config.BANDS[name]), 6),
              "filtered_power_uv2": round(latest_window.powers[name], 6) if name in latest_window.powers else None,
              "relative_power": round(latest_window.rel[name], 6) if name in latest_window.rel else None}
             for name in ("theta", "alpha", "beta")]
    usable = [row for row in rows if row["window"].usable]
    result = {
        "schema": SCHEMA, "source": source, "rate": rate, "channels": count, "channel_labels": labels,
        "pipeline": {**conditioning.description(rate), "window_sec": 4, "step_sec": 2, "context_sec": 6,
                     "source_unit_factor_to_uv": factor, "display_decimation": "per_bin_peak_with_mask_union",
                     "diagnostic_features_only_when_rejected": True},
        "timing": {"compute_ms": 0., "window_sec": 4, "step_sec": 2, "context_sec": 6,
                   "filter_history_sec": 2, "lowpass_delay_ms": 1000 * max(1, round(.18 * rate)) / rate,
                   "physical_latency_validated": False,
                   "note": "计算耗时、4秒信息跨度、2秒更新和滤波延迟分别报告；未测物理采集到显示的绝对延迟。"},
        "quality": {"windows_total": len(rows), "windows_usable": len(usable),
                    "valid_ratio": round(len(usable) / len(rows), 6),
                    "usable_seconds": round(union_seconds((row["start"], row["end"]) for row in usable), 6),
                    "reason_counts": dict(reasons), "threshold_status": config.VALIDATION_STATUS},
        "baseline": baseline_public,
        "traces": {"time_sec": _array(selected / rate), "channels": trace_channels,
                   "artifact_mask": display_mask.tolist(), "masked_intervals": _intervals(mask, rate),
                   "unit": "uV", "window_start_sec": round(last_end / rate - 4, 6)},
        "spectrum": {"freqs_hz": _array(raw_f), "raw_psd_uv2_hz": _array(raw_p, 9),
                     "filtered_psd_uv2_hz": _array(filtered_p, 9), "unit": "uV²/Hz",
                     "feature_eligible": latest_window.usable,
                     "raw_line_fraction": _line_fraction(raw_f, raw_p),
                     "filtered_line_fraction": _line_fraction(raw_f, filtered_p) if np.isfinite(filtered_p).all()
                     else {"50": None, "60": None}},
        "bands": bands,
        "latest": {**windows[-1], "quality_metrics": latest_window.quality.metrics,
                   "features": dict(latest_row["features"]),
                   "score_semantics": "relative_personal_baseline_index_not_state_probability"},
        "windows": windows,
        "limitations": metadata()["limitations"],
    }
    result["timing"]["compute_ms"] = round(1000 * (time.perf_counter() - started), 3)
    return result
