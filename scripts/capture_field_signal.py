"""Record a bounded, exact-name LSL signal check using existing Capture code.

This pilot has no task, probe, labels or protocol_complete event and cannot
contribute training/test rows. It never changes the vendor application.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ningsi/src"))
sys.path.insert(0, str(ROOT / "ningsi-capture/src"))

from ningsi.acquisition.eeg import unit_conflict
from ningsi.acquisition.timing import BIOMULTI_LSL_TIMING, STRICT_TIMING
from ningsi_capture import __version__
from ningsi_capture.acquisition import LslRecorder, discover
from ningsi_capture.biomulti import automatic_unit, supported
from ningsi_capture.export import export_session
from ningsi_capture.storage import SessionStore


def source_for_check(source):
    """Use declared/software-verifiable units, never infer electrodes or gain."""
    source = dict(source)
    resolved = automatic_unit(source)
    source.update(resolved or {"source_unit": "unknown", "scale_to_uv": None,
                               "unit_basis": "unconfirmed_raw_only"})
    # LslRecorder fetches the full XML after discovery. This permits its existing
    # metadata resolver to confirm units that were absent from a short descriptor.
    source["unit_selection"] = "automatic"
    source["reference"] = source.get("reference") or "unknown"
    source["timestamp_policy"] = BIOMULTI_LSL_TIMING if supported(source) else STRICT_TIMING
    source["signal_check_scope"] = "source_metadata_or_verified_local_software; not physical calibration or wear proof"
    conflict = unit_conflict(source)
    if conflict:
        raise ValueError(conflict)
    return source


def capture_signal(*, device, seconds, data, export):
    if not isinstance(device, str) or not device.strip():
        raise ValueError("--device 必须是精确的 LSL EEG 流名。")
    if not math.isfinite(seconds) or not 5 <= seconds <= 60:
        raise ValueError("--seconds 必须在5至60秒之间。")
    data, export = Path(data).resolve(), Path(export).resolve()
    if data.exists():
        raise FileExistsError("--data 必须是新的记录父目录；不会混入或覆盖已有采集。")
    if export.exists():
        raise FileExistsError("--export 必须是全新的导出目录。")
    if data == export:
        raise ValueError("记录父目录与导出目录必须不同。")
    selected = [(info, source) for info, source in discover(timeout=2.) if source["name"] == device]
    if not selected:
        raise RuntimeError("未发现精确匹配的数值 EEG 流；不会改连其他设备或模拟源。")
    if len(selected) != 1:
        raise RuntimeError("发现多个同名 EEG 流，请先消除同名冲突后重试。")
    info, source = selected[0]
    source = source_for_check(source)
    participant = "QC" + datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d")
    store = SessionStore(data, participant=participant, cohort="pilot", source=source,
                         protocol={"name": "signal-check-only-v1", "requested_seconds": seconds,
                                   "purpose": "raw_signal_check_no_task_no_training"},
                         notes="Signal-only pilot: no task, labels, human-wear proof or training eligibility.",
                         lsl_markers=False)
    recorder = None
    failure = None
    elapsed = 0.
    ready_at = None
    try:
        recorder = LslRecorder(store, info)
        recorder.start()
        startup_deadline = time.monotonic() + 12.
        while not recorder.ready.is_set():
            if recorder.error:
                raise RuntimeError("采集启动失败：" + recorder.error)
            if not recorder.is_alive():
                raise RuntimeError("采集线程在收到原始样本前结束。")
            if time.monotonic() >= startup_deadline:
                raise RuntimeError("12秒内未收到原始样本，未开始采集计时。")
            recorder.ready.wait(.1)
        ready_at = time.monotonic()
        initial_samples = store.sample_count
        store.event("signal_check_start", {"requested_seconds": seconds, "no_task": True})
        while time.monotonic() - ready_at < seconds:
            if recorder.error:
                raise RuntimeError("采集中断：" + recorder.error)
            if not recorder.is_alive():
                raise RuntimeError("采集线程提前结束。")
            last = store.last_received
            if last is None or store.clock.now().perf - last >= 3.:
                raise RuntimeError("连续3秒未收到新样本，记录保留但本次信号检查失败。")
            time.sleep(min(.1, max(0., seconds - (time.monotonic() - ready_at))))
        elapsed = time.monotonic() - ready_at
        if store.sample_count < max(1, math.ceil(4 * source["srate"])):
            raise RuntimeError("原始样本不足一个完整4秒窗口，不能生成所需信号历史。")
        store.event("signal_check_stop", {"elapsed_seconds_after_ready": elapsed,
                                          "samples_after_ready": store.sample_count - initial_samples,
                                          "no_task": True})
    except BaseException as exc:
        failure = exc
    finally:
        if recorder is not None:
            try:
                recorder.stop()
            except Exception as exc:
                failure = failure or RuntimeError("停止采集失败：" + str(exc))
                recorder.stop_event.set()
            try:
                if recorder.ident is not None:
                    recorder.join(timeout=10.)
                if recorder.is_alive():
                    failure = failure or RuntimeError("采集线程未在10秒内结束。")
            except Exception as exc:
                failure = failure or RuntimeError("采集线程收尾失败：" + str(exc))
            if recorder.error:
                failure = failure or RuntimeError("采集或关闭信号源失败：" + recorder.error)
        with store.lock:
            final_source = store.metadata["source"]
            if final_source.get("scale_to_uv") is None:
                final_source.update(source_unit="unknown", unit_selection="raw_only",
                                    unit_basis="unconfirmed_raw_only")
        state = "aborted" if isinstance(failure, KeyboardInterrupt) else "failed" if failure else "completed"
        store.close(state, str(failure) if failure else "signal_check_only_no_task_no_training")
    if failure is not None:
        raise RuntimeError(f"信号检查失败；原始记录保留在 {store.path}：{failure}") from failure
    summary = export_session(store.path, export)
    required = ("raw_samples.npy", "timestamps.npy", "events.csv", "windows.csv", "manifest.json",
                "export.json", "SHA256.json")
    missing = [name for name in required if not (export / name).is_file()]
    if missing or not summary.get("raw_samples") or not summary.get("windows_total"):
        raise RuntimeError("导出缺少原始数据或窗口历史：" + ", ".join(missing or ["raw_samples/windows_total"]))
    if summary.get("protocol_complete") or summary.get("train_eligible") or summary.get("test_eligible"):
        raise RuntimeError("信号检查不应包含研究协议完成标记或训练/测试资格。")
    return {"status": "recorded", "scope": "signal_check_only_not_field_acceptance",
            "capture_version": __version__, "recording": str(store.path), "export": str(export),
            "requested_seconds": seconds, "elapsed_seconds_after_ready": elapsed,
            "raw_samples": summary["raw_samples"], "windows_total": summary["windows_total"],
            "windows_usable": summary.get("windows_usable"), "descriptive_spectrum": summary.get("descriptive_spectrum"),
            "baseline_valid": summary.get("baseline_valid"),
            "protocol_complete": False, "train_eligible": 0, "test_eligible": 0,
            "lsl_markers_enabled": False, "physical_wear_and_latency_validated": False,
            "source": {key: store.metadata["source"].get(key) for key in (
                "name", "channels", "srate", "channel_labels", "reference", "source_unit",
                "scale_to_uv", "unit_basis", "unit_selection", "timestamp_policy", "data_origin")}}


def main(argv=None):
    parser = argparse.ArgumentParser(description="短时LSL原始信号检查；无任务/答案，不进入训练集")
    parser.add_argument("--device", required=True, help="精确的LSL EEG流名，不使用模糊匹配")
    parser.add_argument("--seconds", type=float, default=30., help="收到首批样本后采集5至60秒，默认30")
    parser.add_argument("--data", type=Path, required=True, help="全新的原始记录父目录")
    parser.add_argument("--export", type=Path, required=True, help="全新的导出目录")
    args = parser.parse_args(argv)
    try:
        result = capture_signal(device=args.device, seconds=args.seconds, data=args.data, export=args.export)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": type(exc).__name__, "message": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
