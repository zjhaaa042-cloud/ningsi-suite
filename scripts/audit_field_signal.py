"""Read-only, independent MNE spectral audit of a Capture raw export.

This tool does not clean samples, relax production quality gates, train models,
or certify that an operator-declared real source was actually human EEG.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ningsi/src"))
from ningsi.models.human_data import files_from_paths, import_files, summarize


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Non-finite JSON")))


def checked_file(root, name, checksums):
    # Only tool-owned literal filenames are used; never follow paths in a manifest.
    path = (root / name).resolve(strict=True)
    path.relative_to(root)
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if checksums.get(name) != actual:
        raise ValueError(f"Export checksum mismatch: {name}; re-export from the original recording")
    return path


def clock_summary(values):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    adjacent = finite[:-1] & finite[1:]
    differences = np.diff(values)[adjacent]
    return {"samples": len(values), "finite_samples": int(finite.sum()),
            "nonincreasing_adjacent_pairs": int((differences <= 0).sum()),
            "gaps_over_100ms": int((differences > .1).sum()),
            "positive_interval_median_ms": float(np.median(differences[differences > 0]) * 1000)
            if np.any(differences > 0) else None}


def spectral_reference(samples, rate, *, max_windows=1200):
    """MNE Welch on nonoverlapping complete four-second blocks, in sample order."""
    from mne.time_frequency import psd_array_welch

    if not np.isfinite(rate) or not 4 <= rate <= 10000:
        raise ValueError("Invalid nominal sampling rate")
    samples = np.asarray(samples)
    if samples.ndim != 2 or not 1 <= samples.shape[1] <= 256:
        raise ValueError("Expected sample-by-channel matrix")
    if not 1 <= max_windows <= 10000:
        raise ValueError("max_windows must be between 1 and 10000")
    size = round(4 * rate)
    possible = len(samples) // size
    if not possible:
        raise ValueError("At least four seconds of raw samples are required")
    ratios = {"50": [], "60": []}
    accepted, nonfinite, constant = 0, 0, 0
    high = min(100., rate / 2)
    for index in range(min(possible, max_windows)):
        block = np.array(samples[index * size:(index + 1) * size], dtype=float).T
        if not np.isfinite(block).all():
            nonfinite += 1
            continue
        scales = np.max(np.abs(block), axis=1)
        if np.any(scales == 0) or np.any(np.ptp(block, axis=1) == 0):
            constant += 1
            continue
        # Ratios are scale invariant. Unknown physical units remain unknown;
        # this never turns source counts into purported calibrated microvolts.
        block /= scales[:, None]
        power, frequencies = psd_array_welch(
            block, rate, fmin=1., fmax=high, n_fft=size, n_per_seg=size,
            n_overlap=0, window="hann", average="mean", verbose=False)
        total = power.sum(axis=-1)
        if not np.isfinite(total).all() or np.any(total <= 0):
            nonfinite += 1
            continue
        for center in (50, 60):
            if high >= center + 1:
                band = (frequencies >= center - 1) & (frequencies <= center + 1)
                ratios[str(center)].append(power[:, band].sum(axis=-1) / total)
        accepted += 1
    median = {key: np.median(values, axis=0).tolist() if values else None for key, values in ratios.items()}
    return {"method": "mne.time_frequency.psd_array_welch", "window_seconds": 4,
            "window": "hann", "overlap_seconds": 0, "denominator_hz": [1, high],
            "complete_blocks_available": possible, "blocks_examined": min(possible, max_windows),
            "spectrally_computable_blocks": accepted, "nonfinite_blocks": nonfinite,
            "constant_channel_blocks": constant, "truncated": possible > max_windows,
            "line_fraction_median_by_channel": median,
            "clock_basis": "sample order at nominal rate; not a validation of temporal alignment",
            "quality_interpretation": "Computable spectra are not production-accepted EEG windows"}


def audit_export(directory, *, max_windows=1200):
    root = Path(directory).resolve(strict=True)
    checksums = load_json(root / "SHA256.json")
    if not isinstance(checksums, dict):
        raise ValueError("SHA256.json must contain an object")
    named = {name: checked_file(root, name, checksums) for name in
             ("manifest.json", "export.json", "raw_samples.npy", "timestamps.npy")}
    manifest, exported = load_json(named["manifest.json"]), load_json(named["export.json"])
    expected_columns = ["source_timestamp", "corrected_timestamp", "received_lsl", "received_perf"]
    declared_columns = exported.get("timestamp_columns")
    if declared_columns is not None and declared_columns != expected_columns:
        raise ValueError("Unsupported timestamp column contract; re-export using the matching Capture version")
    source = manifest.get("source") or {}
    samples = np.load(named["raw_samples.npy"], mmap_mode="r", allow_pickle=False)
    timestamps = np.load(named["timestamps.npy"], mmap_mode="r", allow_pickle=False)
    if (samples.ndim != 2 or timestamps.shape != (len(samples), 4)
            or samples.shape[1] != source.get("channels")
            or len(samples) != exported.get("raw_samples")):
        raise ValueError("Raw array dimensions do not match the export")
    rate = float(source.get("srate", 0))
    spectra = spectral_reference(samples, rate, max_windows=max_windows)
    clocks = {name: clock_summary(timestamps[:, index]) for index, name in
              ((0, "source"), (1, "corrected"), (3, "received_perf"))}
    for filename, clock_name in (("packet_timestamps.npy", "declared_packet_clock"),
                                 ("device_timestamps.npy", "declared_device_clock")):
        if (root / filename).exists():
            named[filename] = checked_file(root, filename, checksums)
            clock = np.load(named[filename], allow_pickle=False)
            if clock.shape != (len(samples),):
                raise ValueError("Declared clock dimension mismatch")
            clocks[clock_name] = clock_summary(clock)
    try:
        # Reuse the production importer, labels, checksums and exclusion policy.
        imported = import_files(files_from_paths([root]))
        training = {"import_valid": True, "summary": summarize(imported)}
    except (ValueError, OSError, KeyError) as error:
        training = {"import_valid": False, "reason": str(error)}
    checks = [
        {"id": "source", "status": "pending", "message": "真实来源、电极位置、参考与电压单位需现场核对；文件声明不能代替核验。"},
        {"id": "baseline", "status": "pass" if exported.get("baseline_valid") is True else "fail",
         "message": "已有导出记录基线有效" if exported.get("baseline_valid") is True else "现有导出未取得有效基线，不能出个体评分。"},
    ]
    paired = (exported.get("training_readiness") or {}).get("paired")
    checks.append({"id": "paired_labels", "status": "pass" if type(paired) is int and paired > 0 else "fail",
                   "message": f"导出报告中的有效 EEG 与正式标签配对数：{paired if paired is not None else '未提供'}。"})
    for center, values in spectra["line_fraction_median_by_channel"].items():
        if values is not None and any(value > .5 for value in values):
            checks.append({"id": f"line_{center}", "status": "warning",
                           "message": f"{center} Hz 在至少一个通道占可分析频带功率多数；先核对电极接触、参考、接线与环境干扰，陷波后变平不代表恢复了有效脑电。"})
    if clocks["source"]["nonincreasing_adjacent_pairs"]:
        checks.append({"id": "source_clock", "status": "warning",
                       "message": "源时间戳存在非递增；需与原生记录或独立 XDF 对照。分包推算时标不能自动证明逐点物理时间正确。"})
    return {"schema": "ningsi-independent-field-audit-v1", "created_at": datetime.now(timezone.utc).isoformat(),
            "libraries": {name: importlib.metadata.version(name) for name in ("mne", "numpy", "scipy")},
            "input_files_sha256": {name: checksums[name] for name in named},
            "source_declaration": {"data_origin": source.get("data_origin"), "channels": samples.shape[1],
                                   "srate": rate, "source_unit": source.get("source_unit"),
                                   "timestamp_policy": source.get("timestamp_policy")},
            "production_counts": {key: exported.get(key) for key in
                                  ("windows_total", "windows_usable", "baseline_valid", "train_eligible", "test_eligible")},
            "timestamp_columns": expected_columns,
            "timestamp_column_contract": "declared" if declared_columns is not None else "legacy_capture_v1_assumption",
            "spectral_reference": spectra, "clocks": clocks, "training_data_validation": training,
            "checks": checks, "hardware_or_human_validity_established": False,
            "notes": ["只读输入，不更改源数据、预处理版本、质控门槛或冻结模型。",
                      "PSD 以标称采样率按样本顺序计算；不能用它证明时标、清洁信号真值或状态识别准确率。",
                      "received_perf 是软件收到分包的时刻，同包内重复是正常现象，不作为逐样本设备时钟。",
                      "pass 仅表示对应导出字段满足当前软件条件，不等于目标设备或模型效果验收通过。"],
            "references": ["https://mne.tools/stable/generated/mne.time_frequency.psd_array_welch.html",
                           "https://github.com/labstreaminglayer/App-LabRecorder"]}


def write_report(result, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    (output / "audit.json").write_text(encoded, encoding="utf-8")
    rows = "".join(f"<tr><td>{html.escape(row['id'])}</td><td>{html.escape(row['status'])}</td>"
                   f"<td>{html.escape(row['message'])}</td></tr>" for row in result["checks"])
    report = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
              '<title>凝思 · 独立信号核查</title><style>body{font:15px/1.8 sans-serif;color:#203640;max-width:1050px;'
              'margin:40px auto;padding:0 20px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #dce5e8;'
              'padding:10px;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f7f8;padding:16px}'
              '</style><h1>独立信号与采集准备核查</h1><p>离线参考诊断；不代表真人识别、硬件或训练效果已验收。</p>'
              f'<table><tr><th>项目</th><th>结果</th><th>解释</th></tr>{rows}</table>'
              f'<details><summary>完整核查数据与来源</summary><pre>{html.escape(encoded)}</pre></details></html>')
    (output / "audit.html").write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", required=True, type=Path, help="Capture完整原始导出目录")
    parser.add_argument("--output", required=True, type=Path, help="新的核查报告目录，不覆盖已有文件")
    parser.add_argument("--max-windows", type=int, default=1200, help="最多分析多少个不重叠4秒块；报告明确记录截断")
    args = parser.parse_args()
    result = audit_export(args.export, max_windows=args.max_windows)
    write_report(result, args.output)
    print(json.dumps({"report": str(args.output.resolve() / "audit.html"), "checks": result["checks"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
