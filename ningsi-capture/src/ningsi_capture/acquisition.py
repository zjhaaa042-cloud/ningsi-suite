"""Independent LSL inlet with timestamp post-processing disabled.

Each received sample is saved exactly once. Offset estimates, failures, gaps
and raw source XML are retained. Reconnection requires a new recording so a
replacement device cannot silently enter an existing participant session.
"""

from __future__ import annotations

import math
import re
import threading
import time

import numpy as np
from .bootstrap import ensure_engine
ensure_engine()
from ningsi.acquisition.eeg import is_eeg_stream, unit_conflict


def describe(info):
    channels = int(info.channel_count())
    labels, units = [], []
    channel = info.desc().child("channels").child("channel")
    for index in range(channels):
        labels.append(channel.child_value("label").strip() if not channel.empty() else "")
        units.append(channel.child_value("unit").strip() if not channel.empty() else "")
        if not channel.empty():
            channel = channel.next_sibling()
    synthetic = bool(re.search(r"(^|[^a-z])(sim|simulation|synthetic|demo)([^a-z]|$)",
                               (info.name() + " " + info.source_id()).lower()))
    return {
        "kind": "lsl", "data_origin": "simulation_lsl" if synthetic else "real", "name": info.name(), "type": info.type(),
        "channels": channels, "srate": float(info.nominal_srate()),
        "channel_labels": [label or f"Ch{index + 1}" for index, label in enumerate(labels)],
        "channel_labels_declared": labels, "declared_units": units,
        "source_id": info.source_id(), "uid": info.uid(), "hostname": info.hostname(),
        "stream_xml": info.as_xml(), "source_unit": "unknown", "scale_to_uv": None,
        "reference": "unknown",
        "origin_verification": "name-based simulation detection plus operator hardware declaration; not proof of human EEG",
    }


def discover(timeout=1.0):
    from pylsl import resolve_streams
    found = []
    for info in resolve_streams(timeout):
        if not is_eeg_stream(info.type(), info.name(), info.channel_format()):
            continue
        item = describe(info)
        if 1 <= item["channels"] <= 256 and 1 <= item["srate"] <= 10000:
            found.append((info, item))
    return found


def simulation_source(channels=2, srate=250):
    return {
        "kind": "simulation", "data_origin": "simulation", "name": "显式模拟 EEG",
        "channels": channels, "srate": float(srate),
        "channel_labels": [f"Sim{index + 1}" for index in range(channels)],
        "source_unit": "uV", "scale_to_uv": 1.0, "reference": "synthetic",
        "simulation_seed": 20261113,
    }


class Recorder(threading.Thread):
    def __init__(self, store):
        super().__init__(name="ningsi-raw-recorder", daemon=True)
        self.store = store
        self.stop_event = threading.Event()
        self.error = None
        self.ready = threading.Event()

    def stop(self):
        self.stop_event.set()

    def run(self):
        try:
            self.capture()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            try:
                self.store.event("acquisition_error", {"error": self.error})
            except Exception:
                pass  # The UI receives error even when the disk itself failed.


class LslRecorder(Recorder):
    def __init__(self, store, info, inlet_factory=None):
        super().__init__(store)
        self.info = info
        self.inlet_factory = inlet_factory

    def capture(self):
        if self.store.clock.domain != "lsl_local_clock":
            raise RuntimeError("真实流必须使用 LSL 时钟。")
        if self.inlet_factory is None:
            from pylsl import StreamInlet
            inlet = StreamInlet(self.info, max_buflen=360, recover=False, processing_flags=0)
        else:
            inlet = self.inlet_factory(self.info)
        try:
            inlet.open_stream(timeout=3)
            info = inlet.info(timeout=3)
            if not is_eeg_stream(info.type(), info.name(), info.channel_format()):
                raise RuntimeError("连接后的流不属于数值 EEG。请重新发现并选择设备的原始脑电流。")
            full = describe(info)
            source = self.store.metadata["source"]
            if full["channels"] != source["channels"] or full["srate"] != source["srate"]:
                raise RuntimeError("连接后的采样率或通道数发生变化，请重新创建采集。")
            if any(source.get(key) and full[key] != source[key]
                   for key in ("uid", "source_id", "name", "data_origin")):
                raise RuntimeError("连接后的流身份与所选设备不一致，请重新发现设备。")
            with self.store.lock:
                # Discovery may provide only a short description. Keep explicit
                # operator labels, but replace channel placeholders with the full XML.
                source["channel_labels"] = [
                    declared if declared and label == f"Ch{index}" else label
                    for index, (label, declared) in enumerate(
                        zip(source["channel_labels"], full["channel_labels_declared"]), 1)
                ]
                source["stream_xml_full"] = full["stream_xml"]
                source["declared_units"] = full["declared_units"]
                source["channel_labels_declared"] = full["channel_labels_declared"]
                if source.get("unit_selection") == "automatic" and any(full["declared_units"]):
                    from .biomulti import automatic_unit
                    resolved = automatic_unit(source)
                    if resolved:
                        source.pop("unit_evidence", None)
                        source.update(resolved)
                self.store._save_meta()
                conflict = unit_conflict(source)
                if conflict:
                    raise ValueError(conflict)
            next_correction = -math.inf
            next_pair = -math.inf
            offset, correction_id = None, None
            last_data = time.perf_counter()
            gap_start = None
            last_timestamp = None
            while not self.stop_event.is_set():
                now = self.store.clock.now()
                if now.perf >= next_correction:
                    try:
                        offset = float(inlet.time_correction(timeout=1.0 if correction_id is None else .05))
                        correction_id = self.store.correction(offset)
                    except Exception as exc:
                        offset = None
                        correction_id = self.store.correction(None, error=str(exc))
                    next_correction = now.perf + 5
                if now.perf >= next_pair:
                    self.store.event("clock_pair", now.as_dict(), now)
                    next_pair = now.perf + 1
                samples, timestamps = inlet.pull_chunk(timeout=.1, max_samples=1024)
                received = self.store.clock.now()
                if timestamps:
                    self.store.append(samples, timestamps, offset=offset,
                                      correction_id=correction_id, received=received)
                    values = np.asarray(timestamps, dtype=float)
                    finite = np.isfinite(values)
                    bad_order = bool(np.any(np.diff(values) <= 0))
                    if last_timestamp is not None and values[0] <= last_timestamp:
                        bad_order = True
                    if not finite.all() or bad_order:
                        self.store.event("source_timestamp_anomaly", {
                            "nonfinite": int(np.sum(~finite)), "nonmonotonic": bad_order}, received)
                    last_timestamp = values[-1]
                    last_data = received.perf
                    self.ready.set()
                    if gap_start is not None:
                        self.store.event("source_gap_end", {"gap_sec": received.perf - gap_start}, received)
                        gap_start = None
                elif received.perf - last_data >= 3 and gap_start is None:
                    gap_start = last_data
                    self.store.event("source_gap_start", {"last_data_perf": last_data}, received)
        finally:
            inlet.close_stream()


def synthetic_samples(indices, srate=250.0, channels=2, seed=20261113):
    """Deterministic synthetic values; never represented as participant EEG."""
    indices = np.asarray(indices, dtype=float)
    seconds = indices / srate
    rows = []
    for channel in range(channels):
        phase = .12 * channel
        beta = 7 + 2 * np.sin(seconds * .19)
        values = (8 * np.sin(2*np.pi*6*seconds + phase)
                  + 7 * np.sin(2*np.pi*10*seconds + phase)
                  + beta * np.sin(2*np.pi*18*seconds + phase)
                  + .6 * np.sin(2*np.pi*37.1*seconds + seed % 17))
        rows.append(values)
    return np.asarray(rows).T


class SimRecorder(Recorder):
    def capture(self):
        source = self.store.metadata["source"]
        rate, channels = source["srate"], source["channels"]
        start = self.store.clock.now()
        correction = self.store.correction(0.0, start)
        count, next_pair = 0, start.perf
        while not self.stop_event.is_set():
            now = self.store.clock.now()
            target = int((now.perf - start.perf) * rate)
            end = min(target, count + 1024)
            if end > count:
                indices = np.arange(count, end)
                self.store.append(synthetic_samples(indices, rate, channels),
                                  start.lsl + indices / rate, offset=0.0,
                                  correction_id=correction, received=now)
                count = end
                self.ready.set()
            if now.perf >= next_pair:
                self.store.event("clock_pair", now.as_dict(), now)
                next_pair = now.perf + 1
            self.stop_event.wait(.02 if end == target else 0)
