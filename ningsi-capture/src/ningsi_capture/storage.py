"""Append-only raw samples and research events, durably committed to SQLite.

Raw amplitudes are little-endian float64 blobs, without filtering, scaling,
deduplication, interpolation or timestamp post-processing. Non-finite values
are preserved in the blobs and excluded from later feature computation.
"""

from __future__ import annotations

from collections import deque
import json
import math
from pathlib import Path
import re
import sqlite3
import threading
import uuid

import numpy as np

from . import __version__
from .clock import Clock, Stamp
from .markers import MarkerPublisher

SCHEMA_VERSION = "ningsi-capture-v1"
COHORTS = ("pilot", "development", "test", "unassigned")


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def engine_fingerprint():
    from .bootstrap import ensure_engine
    ensure_engine()
    from ningsi.models.research_features import engine_fingerprint as fingerprint
    return fingerprint()


def implementation_fingerprint(policy=None):
    from .bootstrap import ensure_engine
    ensure_engine()
    from ningsi.signal.conditioning import implementation
    return implementation(policy)


class SessionStore:
    def __init__(self, root, *, participant, visit="01", cohort="unassigned", source,
                 protocol, clock=None, notes="", lsl_markers=False):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", participant):
            raise ValueError("参与者请使用 1–40 位匿名字母、数字、横线或下划线编号。")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,20}", visit) or cohort not in COHORTS:
            raise ValueError("到访编号或分组无效。")
        channels = int(source["channels"])
        rate = float(source["srate"])
        if not 1 <= channels <= 256 or not math.isfinite(rate) or not 1 <= rate <= 10000:
            raise ValueError("通道数或声明采样率无效。")
        factor = source.get("scale_to_uv")
        if factor is not None and (not math.isfinite(float(factor)) or float(factor) <= 0):
            raise ValueError("转换到 µV 的比例必须为正数。")
        from .bootstrap import ensure_engine
        ensure_engine()
        from ningsi.signal.conditioning import POLICY
        source = {**source, "preprocessing_policy": POLICY}
        self.clock = clock or Clock()
        self.lock = threading.RLock()
        self.close_lock = threading.Lock()
        self.closing = False
        self.closed = False
        self.sample_count = 0
        self.event_count = 0
        self.protocol_start_perf = None
        self.formal_start_perf = None
        self.formal_probe_count = 0
        self.last_received = None
        self.latest = deque(maxlen=1000)
        self.session_id = uuid.uuid4().hex
        self.path = Path(root).resolve() / f"{participant}_{visit}_{self.session_id[:12]}"
        self.path.mkdir(parents=True, exist_ok=False)
        self.db = sqlite3.connect(self.path / "recording.sqlite3", check_same_thread=False, timeout=10)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE samples (
                sample_index INTEGER PRIMARY KEY,
                source_timestamp REAL, source_timestamp_bytes BLOB NOT NULL,
                corrected_timestamp REAL, received_lsl REAL NOT NULL,
                received_perf REAL NOT NULL, correction_id INTEGER, values_f64 BLOB NOT NULL);
            CREATE TABLE corrections (
                id INTEGER PRIMARY KEY, measured_lsl REAL NOT NULL,
                measured_perf REAL NOT NULL, offset_sec REAL, error TEXT NOT NULL);
            CREATE TABLE events (
                event_id INTEGER PRIMARY KEY, kind TEXT NOT NULL, time_lsl REAL NOT NULL,
                time_perf REAL NOT NULL, utc TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE INDEX event_kind ON events(kind);
            CREATE INDEX sample_corrected_clock ON samples(corrected_timestamp);
            CREATE TABLE probe_predictions (onset_event_id INTEGER PRIMARY KEY, result TEXT NOT NULL);
        """)
        self.metadata = {
            "schema": SCHEMA_VERSION, "app_version": __version__, "session_id": self.session_id,
            "participant": participant, "visit": visit, "cohort": cohort,
            "source": source, "protocol": protocol, "notes": str(notes),
            "clock_domain": self.clock.domain, "created": self.clock.now().as_dict(),
            "state": "recording", "sample_count": 0, "event_count": 0,
            "raw_storage": "unfiltered_source_values_little_endian_float64",
            "source_timestamp_postprocessing": "none",
            "engine_fingerprint_at_capture": engine_fingerprint(),
            "implementation_fingerprint_at_capture": implementation_fingerprint(POLICY),
            "stimulus_timing": "tk_idle_estimate; physical display and keyboard latency unvalidated",
        }
        from .probe_prediction import ProbePredictions
        self.predictions = ProbePredictions(self.path, self.metadata, self._save_prediction)
        self.metadata["probe_prediction"] = {
            "mode": "blind_at_onset", "montage": "Fp1-Fp2", "locally_calibrated": False,
            "model_id": self.predictions.model["model_id"] if self.predictions.model else None,
        }
        self.metadata["shadow_250"] = self.predictions.shadow_metadata()
        self.markers = MarkerPublisher(enabled=lsl_markers, clock_domain=self.clock.domain,
                                       origin=source.get("data_origin", "unknown"))
        self.marker_adapter_failures = 0
        self._save_meta()
        self.event("recording_start", {"source_kind": source["kind"]})

    def _save_meta(self):
        self.metadata["lsl_markers"] = {**self.markers.snapshot(),
                                        "adapter_failures": self.marker_adapter_failures}
        encoded = json.dumps(self.metadata, ensure_ascii=False, allow_nan=False)
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO meta VALUES ('session', ?)", (encoded,))
        write_json(self.path / "manifest.json", self.metadata)

    def event(self, kind, payload=None, stamp=None):
        stamp = stamp or self.clock.now()
        payload = dict(payload or {})
        with self.lock:
            if self.closed or self.closing:
                raise RuntimeError("采集已关闭，不能继续写入事件。")
            phase = str(payload.get("phase", ""))
            formal = re.fullmatch(r"sart_[1-9][0-9]*", phase) is not None and not payload.get("practice")
            if kind == "probe_onset":
                payload.update(probe_ordinal=self.formal_probe_count + 1 if formal else None,
                               task_elapsed_sec=(stamp.perf - self.formal_start_perf)
                               if formal and self.formal_start_perf is not None else None,
                               protocol_elapsed_sec=(stamp.perf - self.protocol_start_perf)
                               if self.protocol_start_perf is not None else None)
            encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False)
            with self.db:
                cur = self.db.execute("INSERT INTO events(kind,time_lsl,time_perf,utc,payload) VALUES (?,?,?,?,?)",
                                      (kind, stamp.lsl, stamp.perf, stamp.utc, encoded))
            self.event_count += 1
            # The SQLite commit and original Stamp above are authoritative.
            # Marker delivery can fail/drop independently without failing capture.
            try:
                self.markers.publish(cur.lastrowid, kind, payload, stamp.lsl)
            except Exception:
                self.marker_adapter_failures += 1
            if kind == "protocol_start" and self.protocol_start_perf is None:
                self.protocol_start_perf = stamp.perf
            elif kind == "phase_start" and formal and self.formal_start_perf is None:
                self.formal_start_perf = stamp.perf
            elif kind == "probe_onset" and formal:
                self.formal_probe_count += 1
            if kind in ("phase_start", "phase_end", "probe_onset"):
                self.predictions.observe(kind, payload, stamp, cur.lastrowid, self.sample_count)
            return cur.lastrowid

    def _save_prediction(self, onset_event_id, result):
        encoded = json.dumps(result, ensure_ascii=False, allow_nan=False)
        with self.lock, self.db:
            self.db.execute("INSERT INTO probe_predictions VALUES (?, ?)", (onset_event_id, encoded))

    def correction(self, offset, stamp=None, error=""):
        stamp = stamp or self.clock.now()
        if offset is not None and not math.isfinite(offset):
            raise ValueError("非有限的时钟偏移。")
        with self.lock:
            if self.closed or self.closing:
                raise RuntimeError("采集已关闭。")
            with self.db:
                cur = self.db.execute("INSERT INTO corrections(measured_lsl,measured_perf,offset_sec,error) VALUES (?,?,?,?)",
                                      (stamp.lsl, stamp.perf, offset, str(error)))
            return cur.lastrowid

    def append(self, samples, timestamps, *, offset=None, correction_id=None, received=None):
        received = received or self.clock.now()
        data = np.asarray(samples, dtype="<f8")
        stamps = np.asarray(timestamps, dtype="<f8")
        if data.size == 0 and stamps.size == 0:
            return 0
        if data.ndim != 2 or data.shape[1] != self.metadata["source"]["channels"] or stamps.shape != (len(data),):
            raise ValueError("样本/时间戳/通道数不一致；未静默丢弃数据。")
        if offset is not None and not math.isfinite(offset):
            raise ValueError("时钟偏移无效。")
        with self.lock:
            if self.closed or self.closing:
                raise RuntimeError("采集已关闭。")
            start = self.sample_count
            rows = []
            for index, (stamp, values) in enumerate(zip(stamps, data)):
                raw = float(stamp) if np.isfinite(stamp) else None
                corrected = raw + offset if raw is not None and offset is not None else None
                rows.append((start + index, raw, stamp.tobytes(), corrected, received.lsl,
                             received.perf, correction_id, values.tobytes()))
            with self.db:
                self.db.executemany("INSERT INTO samples VALUES (?,?,?,?,?,?,?,?)", rows)
            self.sample_count += len(rows)
            self.last_received = received.perf
            self.latest.extend(data[-1000:].tolist())
        return len(rows)

    def snapshot(self):
        with self.lock:
            return {"samples": self.sample_count, "events": self.event_count,
                    "last_received_perf": self.last_received, "latest": list(self.latest),
                    "closed": self.closed, "path": str(self.path),
                    "lsl_markers": self.markers.snapshot()}

    def close(self, state="completed", reason=""):
        if state not in ("completed", "aborted", "failed"):
            raise ValueError("无效的结束状态。")
        with self.close_lock:
            with self.lock:
                if self.closed:
                    return
                if not self.closing:
                    self.event("recording_stop", {"state": state, "reason": str(reason)})
                self.closing = True
            self.predictions.close()
            try:
                self.markers.close()
            except Exception:
                self.marker_adapter_failures += 1
            with self.lock:
                if self.predictions.write_failures:
                    self.metadata["probe_prediction"]["write_failures"] = self.predictions.write_failures
                self.metadata.update(state=state, finished=self.clock.now().as_dict(), reason=str(reason),
                                     sample_count=self.sample_count, event_count=self.event_count)
                self._save_meta()
                self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                self.db.close()
                self.closed = True


def open_readonly(session):
    path = Path(session)
    if path.is_dir():
        path = path / "recording.sqlite3"
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def read_metadata(connection):
    meta = json.loads(connection.execute("SELECT value FROM meta WHERE key='session'").fetchone()[0])
    # A process crash leaves state=recording. Never silently call it completed.
    return meta
