"""Read-only segment audit of versioned Capture masks; never produces state scores.

Unmasked means 'not detected by this detector', not clean EEG. Two-second
segments are descriptive only; even four-second candidates retain no additional
production, baseline, model or training eligibility. No samples are interpolated,
padded, or joined across a missing sample. Optional MNE is audit-only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import html
import importlib.metadata
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ningsi/src"))
from ningsi import config
from ningsi.acquisition.eeg import UNIT_FACTORS, normalize_unit, source_type_conflict, unit_conflict
from ningsi.signal.artifacts import check_channels
from audit_field_signal import checked_file, load_json

POLICY = "causal-eeg-artifact-mask-v1"
MASK_BITS = {"1": "ocular", "2": "muscle", "4": "movement", "8": "signal_fault", "16": "unavailable"}
LOCAL_REASONS = {"ocular_transient", "muscle_burst", "motion_transient", "eog", "emg", "amplitude", "channel_span"}
TIMESTAMP_COLUMNS = ["source_timestamp", "corrected_timestamp", "received_lsl", "received_perf"]


def runs(values):
    """Half-open contiguous index intervals; no gap closure."""
    edges = np.diff(np.r_[False, np.asarray(values, dtype=bool), False].astype(np.int8))
    return [(int(a), int(b)) for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def union(intervals):
    merged = []
    for a, b in sorted(intervals):
        if b <= a:
            raise ValueError("Invalid sample interval")
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([int(a), int(b)])
    return merged


def coverage(intervals, rate, total, strict=()):
    merged = union(intervals)
    selected = np.zeros(total, dtype=bool)
    reference = np.zeros(total, dtype=bool)
    for a, b in merged:
        selected[a:b] = True
    for a, b in strict:
        reference[a:b] = True
    return {"sample_intervals_half_open": merged, "samples": int(selected.sum()),
            "seconds_at_nominal_rate": float(selected.sum() / rate),
            "recording_fraction": float(selected.mean()) if total else 0.,
            "additional_samples_outside_strict": int((selected & ~reference).sum()),
            "additional_seconds_outside_strict": float((selected & ~reference).sum() / rate)}


def protect(mask, samples):
    """Symmetric conservative guard around every detected span, including edges."""
    result = np.asarray(mask, dtype=bool).copy()
    for a, b in runs(mask):
        result[max(0, a - samples):min(len(result), b + samples)] = True
    return result


def screen_segments(data, available, rate, seconds, *, step_seconds=.25, strict=(), mne=False):
    """Screen fixed-duration blocks within real contiguous runs, never concatenate."""
    size, hop = round(rate * seconds), max(1, round(rate * step_seconds))
    total = data.shape[1]
    if available.shape != data.shape or size < 2:
        raise ValueError("Segment dimensions do not match")
    per_channel, joint, spectra = [], [], []
    selected_blocks = []
    groups = [(f"channel_{i + 1}", [i]) for i in range(data.shape[0])]
    groups.append(("all_channels", list(range(data.shape[0]))))
    for name, channels in groups:
        intervals, rejected, examined = [], Counter(), 0
        for a, b in runs(np.all(available[channels], axis=0)):
            if b - a < size:
                continue
            starts = sorted(set(range(a, b - size + 1, hop)) | {b - size})
            for first in starts:
                block = data[channels, first:first + size]
                verdict = check_channels(block, rate)
                examined += 1
                if not verdict.ok:
                    rejected.update(verdict.reasons)
                    continue
                intervals.append((first, first + size))
                if name == "all_channels":
                    selected_blocks.append((first, first + size))
                    if mne and len(spectra) < 100:
                        from mne.time_frequency import psd_array_welch
                        power, frequencies = psd_array_welch(
                            block, rate, fmin=.5, fmax=45., n_fft=size, n_per_seg=size,
                            n_overlap=0, window="hann", average="mean", verbose=False)
                        denominator = power.sum(axis=-1)
                        spectra.append({"sample_interval_half_open": [first, first + size],
                                        "relative_alpha_by_channel": (power[:, (frequencies >= 8) & (frequencies < 13)].sum(axis=-1) / denominator).tolist()})
        result = {"name": name, "examined_blocks": examined, "passed_blocks": len(intervals),
                  "rejected_block_reasons": dict(rejected), **coverage(intervals, rate, total, strict)}
        (joint if name == "all_channels" else per_channel).append(result)
    return {"duration_seconds": seconds, "start_step_seconds": step_seconds,
            "interpretation": "descriptive_only" if seconds == 2 else "candidate_only_not_scoring_eligible",
            "channels": per_channel, "all_channels": joint[0],
            "mne_reference": {"enabled": mne, "method": "psd_array_welch; one real contiguous block per call; no annotation omission concatenation",
                              "blocks": spectra, "maximum_blocks": 100,
                              "interpretation": "Relative alpha is descriptive, not a state score or independent cleanliness label"}}


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("Invalid CSV columns")
        return list(reader)


def boolean(value):
    if value not in ("True", "False"):
        raise ValueError("Invalid window boolean")
    return value == "True"


def verify_fingerprint(exported):
    """The guard assumes this finite-support filter; do not guess older contracts."""
    expected = {"signal/conditioning.py": (exported.get("implementation_fingerprint_at_export") or {}).get("signal/conditioning.py")}
    for name in ("config.py", "signal/artifacts.py", "signal/spectrum.py", "signal/bands.py"):
        expected[name] = (exported.get("engine_fingerprint_at_export") or {}).get(name)
    for name, digest in expected.items():
        actual = hashlib.sha256((ROOT / "ningsi/src/ningsi" / name).read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"Unsupported or changed processing implementation: {name}")
    return expected


def audit_export(directory, *, mne=False):
    root = Path(directory).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Export must be a directory")
    checksum_path = (root / "SHA256.json").resolve(strict=True)
    checksum_path.relative_to(root)
    checksums = load_json(checksum_path)
    if not isinstance(checksums, dict):
        raise ValueError("SHA256.json must contain an object")
    names = ("manifest.json", "export.json", "raw_samples.npy", "timestamps.npy",
             "cleaned_windows_uv.npy", "artifact_mask.npy", "windows.csv", "events.csv")
    files = {name: checked_file(root, name, checksums) for name in names}
    manifest, exported = load_json(files["manifest.json"]), load_json(files["export.json"])
    source, preprocessing = manifest.get("source") or {}, exported.get("preprocessing") or {}
    if (source.get("preprocessing_policy") != POLICY or preprocessing.get("policy") != POLICY
            or preprocessing.get("units") != "uV" or preprocessing.get("causal") is not True
            or preprocessing.get("history_sec") != 2 or preprocessing.get("mask_bits") != MASK_BITS
            or exported.get("timestamp_columns") != TIMESTAMP_COLUMNS):
        raise ValueError("Unsupported mask, filtering or timestamp contract; evidence remains unknown")
    fingerprint = verify_fingerprint(exported)
    rate = float(source.get("srate", 0))
    channels = source.get("channels")
    if not math.isfinite(rate) or not 100 <= rate <= 10000 or type(channels) is not int or not 1 <= channels <= 256:
        raise ValueError("Invalid sampling rate or channel count")
    raw = np.load(files["raw_samples.npy"], mmap_mode="r", allow_pickle=False)
    stamps = np.load(files["timestamps.npy"], mmap_mode="r", allow_pickle=False)
    data = np.load(files["cleaned_windows_uv.npy"], mmap_mode="r", allow_pickle=False)
    masks = np.load(files["artifact_mask.npy"], mmap_mode="r", allow_pickle=False)
    rows, events = read_rows(files["windows.csv"]), read_rows(files["events.csv"])
    width, hop, history = round(4 * rate), round(2 * rate), round(2 * rate)
    total, count = len(raw), len(rows)
    if (raw.ndim != 2 or raw.shape[1] != channels or raw.dtype.kind not in "fi"
            or stamps.shape != (total, 4) or stamps.dtype.kind != "f"
            or total != exported.get("raw_samples") or total != manifest.get("sample_count")
            or count != exported.get("windows_total") or count != max(0, 1 + (total - width) // hop)
            or data.shape != (count, channels, width) or data.dtype.kind != "f"
            or masks.shape != (count, width) or masks.dtype.kind != "u" or masks.dtype.itemsize != 1):
        raise ValueError("Export array/window shape, count or dtype mismatch; per-channel masks are unsupported")
    if np.any(masks & np.uint8(224)):
        raise ValueError("Unknown artifact mask bits")
    timing = source.get("timestamp_policy") or "source-timestamps-v1"
    if timing != exported.get("timestamp_policy"):
        raise ValueError("Source and export timing declarations disagree")
    clocks = {"biomulti-lsl-packet-clock-v1": "packet_timestamps.npy",
              "biomulti-rec-device-clock-v1": "device_timestamps.npy"}
    if timing in clocks:
        name = clocks[timing]
        files[name] = checked_file(root, name, checksums)
        axis = np.load(files[name], mmap_mode="r", allow_pickle=False)
        if axis.shape != (total,) or axis.dtype.kind != "f":
            raise ValueError("Analysis clock shape/dtype mismatch")
    elif timing == "source-timestamps-v1":
        axis = stamps[:, 1]
    else:
        raise ValueError("Unsupported timing policy; cannot infer a continuous clock")
    gaps, pending = [], None
    for event in events:
        if event.get("kind") not in ("source_gap_start", "source_gap_end"):
            continue
        when = float(event["time_perf"])
        if not math.isfinite(when):
            raise ValueError("Non-finite gap event time")
        if event["kind"] == "source_gap_start":
            if pending is not None:
                raise ValueError("Nested unresolved source gaps")
            pending = when
        elif pending is None or when < pending:
            raise ValueError("Unmatched source gap end")
        else:
            gaps.append((pending, when))
            pending = None
    if pending is not None:
        gaps.append((pending, math.inf))
    aggregate = np.full((channels, total), np.nan)
    artifact = np.zeros(total, dtype=bool)
    seen_unmasked = np.zeros(total, dtype=bool)
    strict, hard_counts, audits = [], Counter(), []
    global_reasons = []
    factor = source.get("scale_to_uv")
    if (exported.get("unit_conflict") or exported.get("source_type_conflict")
            or unit_conflict(source) or source_type_conflict(source)
            or normalize_unit(source.get("source_unit")) not in UNIT_FACTORS or factor is None
            or not math.isfinite(float(factor)) or float(factor) <= 0):
        global_reasons.append("unit_or_source_unconfirmed")
    for i, row in enumerate(rows):
        end = int(row["sample_end_index"]) + 1
        if end != width + i * hop or end > total:
            raise ValueError("Window rows do not align with recording sample indices")
        start, context_start = end - width, max(0, end - width - history)
        reasons = set(filter(None, row["reasons"].split("|")))
        hard = set(global_reasons) | (reasons - LOCAL_REASONS)
        if context_start == 0 and end < width + history:
            hard.add("preprocessing_warmup")
        context_axis = np.asarray(axis[context_start:end])
        if not np.isfinite(context_axis).all():
            hard.add("clock_unsynchronized")
        else:
            dt = np.diff(context_axis)
            if np.any(dt <= 0):
                hard.add("nonmonotonic_timestamps")
            if np.any(dt > 1.75 / rate):
                hard.add("sample_gap_in_filter_history")
            if abs((axis[end - 1] - axis[start]) * rate / (width - 1) - 1) > .05:
                hard.add("sample_rate_mismatch_over_5pct")
            declared_bounds = [float(row["start_lsl"]), float(row["end_lsl"])]
            if not np.isfinite(declared_bounds).all() or not np.allclose(
                    declared_bounds, [axis[start], axis[end - 1] + 1 / rate], atol=1e-6, rtol=0):
                hard.add("window_clock_bounds_mismatch")
        received = np.asarray(stamps[context_start:end, 3])
        if not np.isfinite(received).all() or np.any(np.diff(received) < 0):
            hard.add("invalid_received_clock")
        elif (np.any(np.diff(received) > 3.)
              or any(a <= received[-1] and b >= received[0] for a, b in gaps)):
            hard.add("source_stopped_or_gap_event")
        if not np.isfinite(raw[context_start:end]).all():
            hard.add("non_finite_raw_history")
        mask, wave = np.asarray(masks[i]), np.asarray(data[i])
        if np.any(mask & 24):
            hard.add("signal_fault_or_unavailable_mask")
        if np.any(~np.isfinite(wave[:, mask == 0])):
            hard.add("non_finite_unmasked_output")
        usable = boolean(row["usable"])
        if not usable and not reasons:
            hard.add("unavailable_reason_unspecified")
        localized_bits = {"ocular_transient": 1, "muscle_burst": 2, "motion_transient": 4, "eog": 1, "emg": 2}
        if any(reason in reasons and not np.any(mask & bit) for reason, bit in localized_bits.items()):
            hard.add("artifact_reason_without_localized_mask")
        if usable and (reasons or np.any(mask)):
            raise ValueError("Contradictory usable window and reasons/mask")
        if usable:
            strict.append((start, end))
        audits.append({"window": i, "sample_interval_half_open": [start, end], "recorded_usable": usable,
                       "hard_exclusions": sorted(hard), "original_reasons": sorted(reasons),
                       "masked_fraction": float(np.count_nonzero(mask) / width)})
        hard_counts.update(hard)
        # Preserve local evidence even when a different whole-window fault vetoes
        # this row. Do not treat UNAVAILABLE or global fault bits as localization.
        artifact[start:end] |= (mask & 7) != 0
        if hard:
            continue
        seen_unmasked[start:end] |= mask == 0
        if usable and not check_channels(wave, rate).ok:
            raise ValueError("Recorded usable window fails the identical current quality implementation")
        previous = aggregate[:, start:end]
        shared = np.isfinite(previous) & np.isfinite(wave)
        if np.any(shared) and not np.allclose(previous[shared], wave[shared], atol=1e-8, rtol=1e-10):
            raise ValueError("Overlapping filtered samples disagree; cannot form a continuous audit trace")
        take = ~np.isfinite(previous) & np.isfinite(wave)
        previous[take] = wave[take]
    if len(strict) != exported.get("windows_usable"):
        raise ValueError("Usable window count disagrees with windows.csv")
    strict_coverage = coverage(strict, rate, total, strict)
    strict_samples = np.zeros(total, dtype=bool)
    for a, b in strict:
        strict_samples[a:b] = True
    scenarios = []
    for seconds in (0., 2.):
        blocked = protect(artifact, round(seconds * rate))
        available = np.isfinite(aggregate) & ~blocked[None, :]
        scenarios.append({"guard_seconds_before_and_after": seconds,
                          "interpretation": "optimistic_mask_only_upper_bound_not_clean" if seconds == 0 else "primary_conservative_2s_filter_support_guard",
                          "unmasked_eligible_before_basic_screening": coverage(runs(np.all(available, axis=0)), rate, total, strict),
                          "segments": [screen_segments(aggregate, available, rate, duration, strict=strict, mne=mne)
                                       for duration in (2., 4.)]})
    libraries = {"numpy": np.__version__}
    if mne:
        libraries["mne"] = importlib.metadata.version("mne")
    return {"schema": "ningsi-artifact-segment-audit-v1", "created_at": datetime.now(timezone.utc).isoformat(),
            "input_directory": str(root), "input_files_sha256": {name: checksums[name] for name in files},
            "processing_fingerprint_verified": fingerprint, "libraries": libraries,
            "source": {key: source.get(key) for key in ("name", "channels", "srate", "channel_labels", "source_unit", "unit_basis", "reference", "timestamp_policy")},
            "mask_scope": "joint_time_mask_from_any_channel; triggering_channel_unknown",
            "recording_samples": total, "nominal_seconds": total / rate,
            "recorded_strict": {"windows_total": count, "windows_usable": len(strict), **strict_coverage},
            "overlapping_mask_disagreement_samples": int((artifact & seen_unmasked).sum()),
            "local_mask_union_inside_recorded_strict_samples": int((artifact & strict_samples).sum()),
            "hard_exclusion_window_counts": dict(hard_counts), "window_audit": audits,
            "scenarios": scenarios, "quality_thresholds_unchanged": dict(config.QUALITY),
            "state_scoring_eligibility_changed": False, "new_train_eligible": 0, "new_test_eligible": 0,
            "hardware_or_clean_signal_truth_established": False,
            "notes": ["样本区间是原始 sample index 的半开区间；覆盖取并集，重叠窗不重复累计。秒数为标称采样率换算。",
                      "只合并对齐且数值一致的相同/相邻原始索引；缺失索引、NaN、屏蔽样本与时标故障不插值、不拼接作频谱。",
                      "掩码按时间联合所有通道；零值只表示检测器未命中，不能证明排除了眼电、肌电、动作或工频残余。",
                      "同一原始索引在不同重叠窗可能因检测边界得到不同掩码；本核查取局部掩码并集，保留所有命中。因此离线保守覆盖可能小于原严格窗覆盖；原生产判断未被修改。",
                      "主结果对已有局部掩码前后各扩 2 秒，覆盖 Capture 2 秒有限因果滤波支持；原掩码并非原始伪迹真值，保护带也不能证明干净。",
                      "无额外保护带仅为乐观上限对照；不是建议用于产品的阈值或通过标准。",
                      "逐通道及联合通道均重新执行现有 check_channels；2 秒只有描述资格，4 秒也只有离线候选资格。",
                      "Capture 使用因果 0.8–42 Hz、有限 2 秒历史和时间联合 mask；Studio 使用另一预处理及 whole-window live_artifacts。不能把此覆盖率当作 Studio 状态可用率。",
                      "没有改变基线、评分、反馈、模型、训练或测试资格。signal-only 导出未安排基线任务或正式标签，不能据此声称基线任务失败或状态已恢复。"],
            "references": ["https://mne.tools/stable/auto_tutorials/preprocessing/20_rejecting_bad_data.html",
                           "https://mne.tools/stable/generated/mne.time_frequency.psd_array_welch.html"]}


def write_report(results, output):
    output = Path(output)
    # Validate serialization before making an exclusive output directory.
    encoded = json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False)
    output.mkdir(parents=True, exist_ok=False)
    (output / "audit.json").write_text(encoded, encoding="utf-8")
    rows = []
    for item in results["exports"]:
        strict = item["recorded_strict"]
        for scenario in item["scenarios"]:
            for segment in scenario["segments"]:
                joint = segment["all_channels"]
                rows.append("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in (
                    Path(item["input_directory"]).name, f"{strict['windows_usable']}/{strict['windows_total']}",
                    f"{strict['seconds_at_nominal_rate']:.2f}", scenario["guard_seconds_before_and_after"],
                    segment["duration_seconds"], f"{joint['seconds_at_nominal_rate']:.2f}",
                    f"{joint['additional_seconds_outside_strict']:.2f}")) + "</tr>")
    report = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
              '<title>凝思 · 局部伪迹离线验证</title><style>body{font:15px/1.8 sans-serif;color:#203640;max-width:1150px;'
              'margin:40px auto;padding:0 20px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #dce5e8;'
              'padding:8px;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f7f8;padding:16px}</style>'
              '<h1>局部伪迹离线验证</h1><p>2 秒只作描述；4 秒仍是候选。未增加任何状态评分、基线或训练资格。</p>'
              '<p>主结果保留前后各 2 秒滤波支持保护带。0 秒保护带仅给出乐观上限，不能证明信号干净。'
              '掩码是所有通道的时间联合结果；无法知道由哪个通道触发。覆盖按原始样本索引并集计时。</p>'
              '<table><tr><th>导出</th><th>原严格窗</th><th>原覆盖秒</th><th>前后保护秒</th><th>连续片段秒</th>'
              '<th>联合通道粗筛覆盖秒</th><th>原严格窗之外秒</th></tr>' + ''.join(rows) + '</table>'
              '<p>Capture 的因果滤波与联合 mask 不等同于 Studio 的预处理和 live_artifacts；不能直接用于推算 Studio 可评分率。'
              '这些非受控采集只作描述比较，不能归因到某次电极或软件调整。</p>'
              f'<details><summary>完整证据、区间与哈希</summary><pre>{html.escape(encoded)}</pre></details></html>')
    (output / "audit.html").write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New output directory; never overwrites")
    parser.add_argument("--mne", action="store_true", help="Optional descriptive Welch reference, no production dependency")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    resolved_output = args.output.resolve()
    for path in args.export:
        if resolved_output == path.resolve() or path.resolve() in resolved_output.parents:
            parser.error("Output must be outside the immutable input export")
    results = {"schema": "ningsi-artifact-segment-comparison-v1",
               "exports": [audit_export(path, mne=args.mne) for path in args.export]}
    write_report(results, args.output)
    print(json.dumps({"report": str(args.output.resolve() / "audit.html"),
                      "state_scoring_eligibility_changed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
