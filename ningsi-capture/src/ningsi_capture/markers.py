"""Opt-in, bounded LSL task markers for an independent recorder.

The database remains authoritative. Network work happens only on a daemon
worker; the acquisition/event path performs a nonblocking bounded enqueue.
Only fixed event names and selected numeric/enum fields can leave the app.
"""

from __future__ import annotations

import json
import math
import queue
import re
import threading
import uuid


SCHEMA = "ningsi-capture-task-marker-v1"
EVENTS = frozenset({
    "recording_start", "recording_stop", "protocol_start", "protocol_complete",
    "protocol_abort", "phase_start", "phase_end", "stimulus_requested",
    "stimulus_onset", "probe_onset", "probe_recovery_start", "probe_cancelled",
    "source_gap_start", "source_gap_end", "task_focus_lost",
})
PHASE = re.compile(
    r"(?:baseline_(?:calibration|holdout)|sart_practice|probe_practice|"
    r"pvt_(?:practice|pre|post)|artifact_(?:still|blink|jaw|head|rest_(?:blink|jaw|head))|"
    r"(?:sart|rest)_(?:[1-9]|1[0-9]|20))"
)
BOUNDARY = ("Software event timestamps on lsl_local_clock; physical display, "
            "keyboard and device latency not validated. Best-effort markers "
            "do not prove that LabRecorder subscribed or saved an XDF file.")


def _create_outlet(source_id, stream_name, origin):
    # Imported only after explicit opt-in and only on the network worker.
    from pylsl import StreamInfo, StreamOutlet
    info = StreamInfo(stream_name, "Markers", 1, 0., "string", source_id)
    info.desc().append_child_value("marker_schema", SCHEMA)
    info.desc().append_child_value("declared_data_origin", origin)
    info.desc().append_child_value("timestamp_basis", "original_event_lsl_local_clock")
    info.desc().append_child_value("scope", BOUNDARY)
    info.desc().append_child_value("privacy", "No participant, notes, answers, scores or full task payload")
    return StreamOutlet(info, chunk_size=1, max_buffered=60)


def marker_sample(event_id, kind, payload, timestamp):
    """Build the entire network sample using a restrictive value allowlist."""
    if kind not in EVENTS or type(event_id) is not int or event_id < 1:
        return None
    if type(timestamp) not in (int, float) or not math.isfinite(timestamp):
        return None
    item = {"schema": SCHEMA, "event_id": event_id, "kind": kind}
    phase = payload.get("phase")
    if isinstance(phase, str) and PHASE.fullmatch(phase):
        item["phase"] = phase
    index = payload.get("index", payload.get("after_trial"))
    if type(index) is int and 0 <= index <= 10000:
        item["trial_index"] = index
    block = payload.get("block")
    if type(block) is int and 0 <= block <= 20:
        item["block"] = block
    if type(payload.get("practice")) is bool:
        item["practice"] = payload["practice"]
    if payload.get("task") in ("sart", "pvt"):
        item["task"] = payload["task"]
    return json.dumps(item, ensure_ascii=True, separators=(",", ":")), float(timestamp)


class MarkerPublisher:
    def __init__(self, *, enabled=False, clock_domain, origin="unknown", capacity=256):
        self.requested = bool(enabled)
        self.enabled = self.requested and clock_domain == "lsl_local_clock"
        self.clock_domain = str(clock_domain)
        token = uuid.uuid4().hex
        self.source_id = "ningsi-capture-markers-" + token
        self.stream_name = "NingsiCapture-TaskMarkers-" + token[:8]
        self.origin = (origin if origin in ("real", "simulation") else
                       "simulation" if origin == "simulation_lsl" else "unknown")
        self._queue = queue.Queue(maxsize=max(1, min(int(capacity), 4096)))
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        self.state = "starting" if self.enabled else "disabled"
        self.error_code = "" if not self.requested or self.enabled else "clock_domain_unavailable"
        if self.error_code:
            self.state = "unavailable"
        self.enqueued = self.published = self.dropped = self.failures = 0
        self.in_flight = 0
        self.shutdown_pending = False
        if self.enabled:
            self._thread = threading.Thread(target=self._run, name="ningsi-lsl-markers", daemon=True)
            try:
                self._thread.start()
            except Exception:
                self._failure("worker_start_failed")

    def _failure(self, code):
        # Do not store exception messages: they can contain arbitrary upstream text.
        with self._lock:
            self.failures += 1
            self.state, self.error_code = "failed", code
        self._discard_pending()

    def _discard_pending(self):
        count = 0
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
            self._queue.task_done()
            count += 1
        with self._lock:
            self.dropped += count

    def publish(self, event_id, kind, payload, timestamp):
        if not self.requested:
            return False
        try:
            sample = marker_sample(event_id, kind, payload, timestamp)
        except Exception:
            sample = None
        if sample is None:
            return False
        with self._lock:
            if not self.enabled or self.state in ("failed", "closed") or self._stop.is_set():
                self.dropped += 1
                return False
            try:
                self._queue.put_nowait(sample)
            except queue.Full:
                self.dropped += 1
                return False
            self.enqueued += 1
        return True

    def _run(self):
        try:
            outlet = _create_outlet(self.source_id, self.stream_name, self.origin)
        except Exception:
            self._failure("outlet_creation_failed")
            return
        with self._lock:
            self.state = "streaming"
        try:
            while not self._stop.is_set() or not self._queue.empty():
                try:
                    sample, timestamp = self._queue.get(timeout=.05)
                except queue.Empty:
                    continue
                with self._lock:
                    self.in_flight = 1
                try:
                    outlet.push_sample([sample], timestamp=timestamp)
                except Exception:
                    with self._lock:
                        self.dropped += 1
                    self._failure("outlet_publish_failed")
                    return
                finally:
                    self._queue.task_done()
                    with self._lock:
                        self.in_flight = 0
                with self._lock:
                    self.published += 1
        finally:
            # The outlet is owned by this worker, never by the SQLite thread.
            del outlet
            with self._lock:
                if self.state != "failed":
                    self.state = "closed"

    def close(self, timeout=.25):
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.ident is not None:
            thread.join(timeout=max(0., min(float(timeout), 1.)))
            if thread.is_alive():
                with self._lock:
                    self.shutdown_pending = True
                    self.error_code = self.error_code or "worker_shutdown_pending"
        return self.snapshot()

    def snapshot(self):
        with self._lock:
            return {"schema": SCHEMA, "requested": self.requested, "enabled": self.enabled,
                    "state": self.state, "clock_domain": self.clock_domain,
                    "stream_name": self.stream_name if self.requested else None,
                    "source_id": self.source_id if self.requested else None,
                    "declared_data_origin": self.origin,
                    "enqueued": self.enqueued, "published": self.published,
                    "dropped": self.dropped, "pending": self._queue.qsize(),
                    "in_flight": self.in_flight,
                    "failures": self.failures, "error_code": self.error_code,
                    "shutdown_pending": self.shutdown_pending,
                    "event_kinds": sorted(EVENTS),
                    "payload_fields": ["schema", "event_id", "kind", "phase", "trial_index",
                                       "block", "practice", "task"],
                    "delivery": "best_effort_no_recorder_acknowledgement",
                    "timestamp_basis": "original_saved_event_stamp_lsl",
                    "scope": BOUNDARY}
