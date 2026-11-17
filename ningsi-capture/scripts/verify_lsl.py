"""Native LSL transport acceptance with a synthetic counter, never human data."""

import argparse
import json
from pathlib import Path
import sys
import threading
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ningsi_capture.bootstrap import ensure_engine
ensure_engine()

import numpy as np
from pylsl import StreamInfo, StreamOutlet, cf_float32, local_clock, resolve_byprop
from ningsi_capture.acquisition import LslRecorder, describe
from ningsi_capture.export import export_session
from ningsi_capture.protocol import ProtocolConfig
from ningsi_capture.storage import SessionStore, write_json


def run(root, seconds=8):
    source_id = "ningsi-capture-synthetic-" + uuid.uuid4().hex
    info = StreamInfo("NingsiCapture-simulation-acceptance", "EEG", 2, 250., cf_float32, source_id)
    channel_root = info.desc().append_child("channels")
    for name in ("SyntheticCounter", "NegativeCounter"):
        node = channel_root.append_child("channel")
        node.append_child_value("label", name)
        node.append_child_value("unit", "uV")
    outlet = StreamOutlet(info, chunk_size=25)
    stop = threading.Event()
    def push():
        index, start = 0, local_clock()
        while not stop.is_set():
            target = int((local_clock() - start) * 250)
            end = min(index + 25, target)
            if end > index:
                values = np.arange(index, end, dtype=float)
                outlet.push_chunk(np.stack([values, -values], axis=1).tolist(), start + (end-1)/250)
                index = end
            stop.wait(.01)
    sender = threading.Thread(target=push, daemon=True)
    sender.start()
    store = recorder = None
    try:
        streams = resolve_byprop("source_id", source_id, minimum=1, timeout=5)
        if not streams:
            raise RuntimeError("Synthetic LSL stream discovery failed.")
        source = describe(streams[0])
        source.update(source_unit="uV", scale_to_uv=1.)
        store = SessionStore(root / "recordings", participant="LSL_SYNTHETIC", cohort="pilot",
                             source=source, protocol=ProtocolConfig().as_dict())
        recorder = LslRecorder(store, streams[0])
        recorder.start()
        if not recorder.ready.wait(6):
            raise RuntimeError(recorder.error or "No samples received from synthetic outlet.")
        started, marked = time.perf_counter(), False
        while time.perf_counter() - started < seconds:
            if recorder.error:
                raise RuntimeError(recorder.error)
            if not marked and time.perf_counter() - started > seconds / 2:
                store.event("manual_marker", {"code": "synthetic_transport_midpoint"})
                marked = True
            stop.wait(.05)
    finally:
        if recorder:
            recorder.stop()
            recorder.join(8)
        stop.set()
        sender.join(2)
        if store:
            store.close("aborted", "synthetic_lsl_transport_acceptance")
    summary = export_session(store.path, root / "export")
    raw = np.load(root / "export" / "raw_samples.npy")
    stamps = np.load(root / "export" / "timestamps.npy")
    assert len(raw) >= seconds * 200
    assert np.all(np.diff(raw[:, 0]) == 1), "Synthetic sequence has a lost/duplicate row."
    assert np.array_equal(raw[:, 1], -raw[:, 0]), "Raw values changed."
    assert np.isfinite(stamps[:, 1]).all(), "Clock correction was not retained."
    assert summary["source_origin"] == "simulation_lsl"
    assert summary["train_eligible"] == summary["test_eligible"] == 0
    with (root / "export" / "manifest.json").open(encoding="utf-8") as file:
        exported_source = json.load(file)["source"]
    assert exported_source["channel_labels"] == ["SyntheticCounter", "NegativeCounter"]
    assert exported_source["declared_units"] == ["uV", "uV"]
    report = {"kind": "synthetic_native_lsl_transport", "duration_sec": seconds,
              "raw_samples": len(raw), "counter_contiguous": True, "raw_values_unchanged": True,
              "connected_channel_labels": exported_source["channel_labels"],
              "declared_units": exported_source["declared_units"],
              "original_timestamp_median_step_sec": float(np.median(np.diff(stamps[:, 0]))),
              "source_origin": summary["source_origin"], "human_hardware_tested": False,
              "session": str(store.path), "export": str(root / "export")}
    write_json(root / "verification.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=8)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.out, arguments.seconds), indent=2))
