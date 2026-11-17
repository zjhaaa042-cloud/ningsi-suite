"""Consistent recording snapshots, raw exports and one observation per probe."""

from __future__ import annotations

from collections import deque
import csv
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import struct
import tempfile

import numpy as np
from .bootstrap import ensure_engine
ensure_engine()
from ningsi.acquisition.eeg import source_type_conflict, unit_conflict
from ningsi.acquisition.timing import (BIOMULTI_CONFIRM_SAMPLES, BIOMULTI_LSL_TIMING, BIOMULTI_REC_TIMING, BIOMULTI_WARM_SAMPLES,
                                      PACKET_TIMING, STRICT_TIMING, analysis_timestamps, biomulti_packet_clock)
from ningsi.models.capture_protocols import RESEARCH_PROTOCOLS
from ningsi.signal import conditioning

from .readiness import formal_probe_summary, line_power_ratios
from .storage import engine_fingerprint, implementation_fingerprint, open_readonly, read_metadata, write_json

FEATURES = ("rel_theta", "rel_alpha", "rel_beta", "log_beta_theta", "log_alpha_beta", "log_theta_alpha")
PROBE_FIELDS = ("session_id", "participant", "visit", "cohort", "data_origin", "probe_id", "phase",
                "practice", "onset_lsl", "choice", "label", "label_name", "timeout", "cancelled",
                "window_start_lsl", "window_end_lsl", "windows_total", "windows_usable", "usable_seconds",
                "baseline_valid", "focus", "focus_z", *FEATURES, "feature_usable", "train_eligible",
                "test_eligible", "binary_label_valid", "unavailable_reasons")


def csv_write(path, fields, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def union_seconds(intervals):
    total, end = 0., -math.inf
    for left, right in sorted(intervals):
        total += max(0., right - max(left, end))
        end = max(end, right)
    return total


def read_events(connection):
    return [{"event_id": row["event_id"], "kind": row["kind"], "time_lsl": row["time_lsl"],
             "time_perf": row["time_perf"], "utc": row["utc"], "payload": json.loads(row["payload"])}
            for row in connection.execute("SELECT * FROM events ORDER BY event_id")]


def _analyze_samples(context, source):
    values = np.stack([item[0] for item in context])
    stamps = np.asarray([item[1] for item in context], dtype=float)
    factor = None if unit_conflict(source) or source_type_conflict(source) else source.get("scale_to_uv")
    rate = float(source["srate"])
    policy = source.get("preprocessing_policy")
    row = conditioning.condition(values.T, stamps, rate, factor, policy, traces=True)
    row["raw_50hz_ratios"] = line_power_ratios(values[:, :2], rate)
    if policy:
        row["raw_60hz_ratios"] = line_power_ratios(values[:, :2], rate, 60)
        wave = row["filtered_uv"]
        for center in (50, 60):
            row[f"filtered_{center}hz_ratios"] = (line_power_ratios(wave[:2].T, rate, center) if wave is not None
                                                  else (None,) * min(2, values.shape[1]))
        mask = row["artifact_mask"]
        row["masked_fraction"] = float(np.count_nonzero(mask) / len(mask)) if mask is not None else None
    return row


def analyze_context(context, source, analyze=None):
    times, audit = analysis_timestamps([item[1] for item in context], float(source["srate"]),
                                      source.get("timestamp_policy"))
    timed = [(item[0], stamp) for item, stamp in zip(context, times)]
    row = (analyze or _analyze_samples)(timed, source)
    row["timestamp_adjustment"] = audit
    return row


def recent_window_quality(session):
    connection = open_readonly(session)
    try:
        metadata = read_metadata(connection)
        source = metadata["source"]
        policy = source.get("timestamp_policy")
        if policy == BIOMULTI_LSL_TIMING:
            total = connection.execute("SELECT COALESCE(MAX(sample_index)+1,0) FROM samples").fetchone()[0]
            end = max(1000, (total - BIOMULTI_CONFIRM_SAMPLES) // 250 * 250)
            if total < 1000:
                return ("signal_too_short",)
            start = max(0, end - 1500)
            origin, axis = packet_axis(connection, start, min(total, end + BIOMULTI_CONFIRM_SAMPLES))
            values = connection.execute("SELECT sample_index, values_f64 FROM samples WHERE sample_index>=? "
                                        "AND sample_index<? ORDER BY sample_index", (start, end))
            context = [(np.frombuffer(row[1], dtype="<f8"), axis[row[0] - origin]) for row in values]
            reasons = _analyze_samples(context, source)["window"].quality.reasons
            return tuple("packet_clock_unresolved" if r == "clock_unsynchronized" else r for r in reasons)
        count = round(6 * source["srate"])
        rows = connection.execute("SELECT values_f64, corrected_timestamp FROM samples "
                                  "ORDER BY sample_index DESC LIMIT ?", (count,)).fetchall()
    finally:
        connection.close()
    if not rows:
        return ("signal_too_short",)
    context = [(np.frombuffer(row[0], dtype="<f8"), row[1]) for row in reversed(rows)]
    if source.get("timestamp_policy") == BIOMULTI_REC_TIMING:
        # Check the amplitude/shape on sample order only. No online features or
        # baseline may use this provisional axis. Device timing follows REC save.
        provisional = [(item[0], i / source["srate"]) for i, item in enumerate(context)]
        reasons = _analyze_samples(provisional, source)["window"].quality.reasons
        return (*reasons, "native_clock_pending")
    return analyze_context(context, source)["window"].quality.reasons


def packet_axis(connection, first_index, stop_index, received_by=None):
    # Include the anchor and the possible three-second transport/reset history.
    origin = max(0, first_index // 250 * 250 - BIOMULTI_WARM_SAMPLES)
    if stop_index is None:
        stop_index = connection.execute("SELECT COALESCE(MAX(sample_index)+1,0) FROM samples").fetchone()[0]
    raw = np.full(max(0, stop_index - origin), np.nan)
    sql = ("SELECT sample_index, corrected_timestamp FROM samples WHERE sample_index>=? AND sample_index<? "
           "AND ABS(received_lsl - corrected_timestamp)<3")
    args = [origin, stop_index]
    if received_by is not None:
        sql += " AND received_perf<=?"
        args.append(received_by)
    for index, stamp in connection.execute(sql, args):
        raw[index - origin] = stamp
    return origin, biomulti_packet_clock(raw, first_index=origin)


def analysis_windows(connection, metadata, *, first_index=0, stop_index=None,
                     analyze=None, received_by=None, analysis_times=None, sample_limit=None,
                     keep_traces=False, window_sink=None):
    """Use recording-aligned windows with end indices in [first_index, stop_index)."""
    source = metadata["source"]
    rate = float(source["srate"])
    size, hop = round(4 * rate), max(1, round(2 * rate))
    context = deque(maxlen=round(6 * rate))
    analyze = analyze or _analyze_samples
    policy = source.get("timestamp_policy")
    separate_axis = policy in (BIOMULTI_REC_TIMING, BIOMULTI_LSL_TIMING)
    raw_times = deque(maxlen=context.maxlen) if separate_axis else None
    # Read enough warm-up samples for the earliest requested window.
    start = max(0, first_index - context.maxlen + 1)
    origin = 0
    if policy == BIOMULTI_LSL_TIMING and analysis_times is None:
        if rate != 250:
            raise ValueError("BioMulti 分包时钟仅适用于 250 Hz EEG。")
        axis_stop = None if stop_index is None else stop_index + BIOMULTI_CONFIRM_SAMPLES
        if sample_limit is not None:
            axis_stop = sample_limit if axis_stop is None else min(sample_limit, axis_stop)
        origin, analysis_times = packet_axis(connection, start, axis_stop, received_by)
    if separate_axis and analysis_times is not None:
        end_of_axis = origin + len(analysis_times)
        stop_index = end_of_axis if stop_index is None else min(stop_index, end_of_axis)
    query = ("SELECT sample_index, values_f64, corrected_timestamp, received_perf FROM samples "
             "WHERE sample_index >= ?")
    params = [start]
    if stop_index is not None:
        query += " AND sample_index < ?"
        params.append(stop_index)
    results = []
    for row in connection.execute(query + " ORDER BY sample_index", params):
        stamp = row["corrected_timestamp"]
        if separate_axis:
            raw_times.append(stamp)
            stamp = analysis_times[row["sample_index"] - origin] if analysis_times is not None else None
        if received_by is not None and row["received_perf"] > received_by:
            stamp = None  # Preserve the gap; a late arrival cannot qualify a live window.
        context.append((np.frombuffer(row["values_f64"], dtype="<f8"), stamp))
        count = row["sample_index"] + 1
        if row["sample_index"] < first_index or count < size or (count - size) % hop:
            continue
        if separate_axis:
            result = analyze(context, source)
            device = np.array([item[1] for item in context], dtype=float)
            raw = np.array(raw_times, dtype=float)
            paired = np.isfinite(device) & np.isfinite(raw)
            result["timestamp_adjustment"] = {"policy": policy, "applied": bool(paired.all()),
                                              "max_shift_sec": float(np.max(np.abs(device[paired] - raw[paired]))) if paired.any() else 0.}
            # Unmatched windows still count against baseline/probe coverage.
            # Their provisional bounds do not make them usable for features.
            if result["start"] is None and np.isfinite(raw[-size]):
                result["start"] = float(raw[-size])
            if result["end"] is None and np.isfinite(raw[-1]):
                result["end"] = float(raw[-1] + 1 / rate)
        else:
            result = analyze_context(context, source, analyze)
        result["sample_end_index"] = count - 1
        if window_sink is not None:
            window_sink(result)
        if not keep_traces:
            result.pop("filtered_uv", None)
            result.pop("artifact_mask", None)
        results.append(result)
    return results


def recorded_predictions(connection):
    has_table = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='probe_predictions'").fetchone()
    return {row[0]: json.loads(row[1]) for row in connection.execute(
        "SELECT onset_event_id, result FROM probe_predictions")} if has_table else {}


def prediction_table(connection, metadata, events, probes, output, *, results=None):
    """Pair an immutable, capture-time prediction with the independently recorded response."""
    if results is None:
        results = recorded_predictions(connection)
    onsets = [event for event in events if event["kind"] == "probe_onset"]
    responses = {event["payload"]["probe_id"]: event for event in events if event["kind"] == "probe_response"}
    rows = []
    for onset, probe in zip(onsets, probes, strict=True):
        response = responses.get(probe["probe_id"])
        payload = response["payload"] if response else {}
        if probe["cancelled"]:
            response_status = "cancelled"
        elif response is None:
            response_status = "unanswered"
        elif probe["timeout"]:
            response_status = "timeout"
        elif probe["binary_label_valid"]:
            response_status = "binary"
        else:
            response_status = "nonbinary"
        prediction = results.get(onset["event_id"], {})
        status = prediction.get("status", "not_recorded")
        comparable = status == "scored" and response_status == "binary" and not probe["practice"]
        rows.append({
            **{name: probe[name] for name in ("session_id", "participant", "visit", "cohort", "data_origin",
                                              "probe_id", "phase", "practice", "onset_lsl", "choice", "label",
                                              "label_name", "window_start_lsl", "window_end_lsl")},
            **prediction, "onset_event_id": onset["event_id"], "status": status,
            "reason": prediction.get("reason", "" if prediction else "no_live_prediction_record"),
            "response_status": response_status, "response_rt_sec": payload.get("response_rt_sec"),
            "comparable": comparable,
            "agreement": prediction["predicted_label"] == probe["label"] if comparable else None,
            **prediction.get("features", {}),
        })
    fields = ("session_id", "participant", "visit", "cohort", "data_origin", "probe_id", "phase", "practice",
              "onset_event_id", "onset_lsl", "sample_count_at_onset", "window_start_lsl", "window_end_lsl",
              "status", "reason", "error_type", "model_id", "windows_total", "windows_usable", "usable_seconds",
              *FEATURES, "mw_score", "threshold", "predicted_label", "choice", "response_status", "response_rt_sec",
              "label", "label_name", "comparable", "agreement")
    csv_write(output / "probe_predictions.csv", fields, rows)
    return {"file": "probe_predictions.csv", "model_id": (metadata.get("probe_prediction") or {}).get("model_id"),
            "scored": sum(row["status"] == "scored" for row in rows),
            "comparable": sum(row["comparable"] for row in rows),
            "agreed": sum(row["agreement"] is True for row in rows),
            "not_recorded": sum(row["status"] == "not_recorded" for row in rows),
            "unavailable": sum(row["status"] == "unavailable" for row in rows),
            "interpretation": "external candidate; blind at onset; exploratory, not locally calibrated"}


def _export_windows(connection, metadata, output, analysis_times):
    source = metadata["source"]
    if not source.get("preprocessing_policy"):
        return analysis_windows(connection, metadata, analysis_times=analysis_times)
    rate, channels = source["srate"], source["channels"]
    width, hop = round(4 * rate), max(1, round(2 * rate))
    samples = connection.execute("SELECT count(*) FROM samples").fetchone()[0]
    count = max(0, 1 + (samples - width) // hop)
    with ((output / "cleaned_windows_uv.npy").open("wb") as clean,
          (output / "artifact_mask.npy").open("wb") as masks):
        for file, shape, dtype in ((clean, (count, channels, width), "<f8"), (masks, (count, width), "|u1")):
            np.lib.format.write_array_header_1_0(file, {"descr": dtype, "fortran_order": False, "shape": shape})

        def write_trace(row):
            wave, mask = row["filtered_uv"], row["artifact_mask"]
            if wave is None:
                wave = np.full((channels, width), np.nan)
                mask = np.full(width, conditioning.UNAVAILABLE, dtype=np.uint8)
            else:
                wave = np.where(mask, np.nan, wave)
            clean.write(np.asarray(wave, dtype="<f8").tobytes())
            masks.write(mask.tobytes())

        windows = analysis_windows(connection, metadata, analysis_times=analysis_times, window_sink=write_trace)
        if len(windows) != count:
            raise ValueError("处理窗与原始样本索引不一致。")
    return windows


def feature_tables(connection, metadata, events, output, engine_changed=False, analysis_times=None):
    from .comparison_export import comparison_table
    from ningsi.signal.baseline import build_baseline
    from ningsi.signal.indicators import compute_indicators
    from ningsi.models.logistic import features_from_window
    from ningsi.models.research_features import aggregate_probe

    windows = _export_windows(connection, metadata, output, analysis_times)
    conflict = unit_conflict(metadata["source"])
    type_conflict = source_type_conflict(metadata["source"])
    phase_starts = {e["payload"]["phase"]: e["time_lsl"] for e in events if e["kind"] == "phase_start"}
    phase_ends = {e["payload"]["phase"]: e["time_lsl"] for e in events if e["kind"] == "phase_end"}
    begin = phase_starts.get("baseline_calibration")
    finish = phase_ends.get("baseline_calibration")
    calibration = [row["window"] for row in windows
                   if begin is not None and finish is not None and row["start"] is not None
                   and row["end"] is not None and row["start"] >= begin and row["end"] <= finish]
    baseline = build_baseline(calibration, device=metadata["source"]["name"], srate=metadata["source"]["srate"])
    write_json(output / "baseline.json", baseline.as_dict())
    window_rows = []
    line_columns = [f"raw_50hz_ratio_ch{i+1}" for i in range(min(2, metadata["source"]["channels"]))]
    measures = ("raw_60hz", "filtered_50hz", "filtered_60hz") if metadata["source"].get("preprocessing_policy") else ()
    extra_columns = [f"{measure}_ratio_ch{i+1}" for measure in measures for i in range(len(line_columns))]
    for row in windows:
        window = row["window"]
        result = compute_indicators(window, baseline)
        row["indicators"] = result
        row["features"] = features_from_window(window) if window.usable else {}
        window_rows.append({"start_lsl": row["start"], "end_lsl": row["end"],
                            "sample_end_index": row["sample_end_index"], "usable": window.usable,
                            "reasons": "|".join(window.quality.reasons),
                            "focus": result.score("focus"), "focus_z": result.index_z.get("focus"),
                            "timestamp_policy": row["timestamp_adjustment"]["policy"],
                            "timestamp_adjusted": row["timestamp_adjustment"]["applied"],
                            "timestamp_max_shift_ms": row["timestamp_adjustment"]["max_shift_sec"] * 1000,
                            **row["features"], **dict(zip(line_columns, row["raw_50hz_ratios"])),
                            **dict(zip(extra_columns, (v for m in measures for v in row[m + "_ratios"]))),
                            **({"masked_fraction": row["masked_fraction"]} if measures else {})})
    csv_write(output / "windows.csv", ("start_lsl", "end_lsl", "sample_end_index", "usable", "reasons",
                                       "focus", "focus_z", "timestamp_policy", "timestamp_adjusted",
                                       "timestamp_max_shift_ms", *FEATURES, *line_columns, *extra_columns,
                                       *(("masked_fraction",) if measures else ())), window_rows)
    onsets = [e for e in events if e["kind"] == "probe_onset"]
    responses = {e["payload"]["probe_id"]: e for e in events if e["kind"] == "probe_response"}
    cancellations = {e["payload"]["probe_id"] for e in events if e["kind"] == "probe_cancelled"}
    complete = metadata["state"] == "completed" and any(e["kind"] == "protocol_complete" for e in events)
    research = metadata["protocol"].get("name") in RESEARCH_PROTOCOLS
    probe_rows = []
    for event in onsets:
        payload, onset = event["payload"], event["time_lsl"]
        probe_id, phase = payload["probe_id"], payload["phase"]
        response = responses.get(probe_id, {}).get("payload", {})
        choice = response.get("choice")
        label = 0 if choice == 1 else 1 if choice in (3, 5) else None
        left, right = onset - 12, onset - 2
        candidates = [row for row in windows if row["start"] is not None and row["end"] is not None
                      and row["start"] >= left and row["end"] <= right]
        usable = [row for row in candidates if row["window"].usable]
        covered = union_seconds((row["start"], row["end"]) for row in usable)
        aggregation = aggregate_probe(windows, onset, baseline_valid=baseline.valid,
                                      phase_start=phase_starts.get(phase, math.inf))
        reasons = list(aggregation["reasons"])
        if conflict:
            reasons.append("unit_conflict")
        if type_conflict:
            reasons.append("source_type_mismatch")
        feature_ok = not reasons
        label_ok = label is not None and probe_id not in cancellations and bool(response) and not response.get("timeout")
        if payload.get("practice"):
            reasons.append("practice")
        if not label_ok:
            reasons.append("missing_or_nonbinary_label")
        if probe_id in cancellations or not response:
            reasons.append("probe_cancelled_or_unanswered")
        real = metadata["source"].get("data_origin") == "real"
        if not real:
            reasons.append("simulation_or_unverified_origin")
        if not complete:
            reasons.append("recording_or_protocol_incomplete")
        if engine_changed:
            reasons.append("engine_changed_since_capture")
        if metadata["cohort"] in ("pilot", "unassigned"):
            reasons.append("pilot_cohort" if metadata["cohort"] == "pilot" else "unassigned_cohort")
        if not research:
            reasons.append("nonresearch_protocol")
        eligible = (feature_ok and label_ok and real and complete and not payload.get("practice")
                    and not engine_changed and research)
        row = {"session_id": metadata["session_id"], "participant": metadata["participant"],
               "visit": metadata["visit"], "cohort": metadata["cohort"],
               "data_origin": metadata["source"].get("data_origin", "unknown"),
               "probe_id": probe_id, "phase": phase, "practice": bool(payload.get("practice")),
               "onset_lsl": onset, "choice": choice, "label": label,
               "label_name": "on_task" if label == 0 else "internal_mind_wandering" if label == 1 else "",
               "timeout": response.get("timeout", False), "cancelled": probe_id in cancellations,
               "window_start_lsl": left, "window_end_lsl": right, "windows_total": len(candidates),
               "windows_usable": len(usable), "usable_seconds": round(covered, 6),
               "baseline_valid": baseline.valid, "feature_usable": feature_ok,
               "train_eligible": eligible and metadata["cohort"] == "development",
               "test_eligible": eligible and metadata["cohort"] == "test", "binary_label_valid": label_ok,
               "unavailable_reasons": "|".join(reasons)}
        if feature_ok:
            row.update(aggregation["features"])
            row["focus"] = float(np.mean([item["indicators"].score("focus") for item in usable]))
            row["focus_z"] = float(np.mean([item["indicators"].index_z["focus"] for item in usable]))
        probe_rows.append(row)
    csv_write(output / "probes.csv", PROBE_FIELDS, probe_rows)
    csv_write(output / "features.csv", PROBE_FIELDS, probe_rows)
    predictions = recorded_predictions(connection)
    return {"probe_predictions": prediction_table(connection, metadata, events, probe_rows, output, results=predictions),
            "probe_comparison": comparison_table(metadata, events, predictions, output, csv_write),
            "baseline_valid": baseline.valid, "unit_conflict": conflict, "source_type_conflict": type_conflict,
            "training_readiness": {**formal_probe_summary(probe_rows), "cohort": metadata["cohort"],
                                   "baseline_windows": len(calibration), "baseline_usable": baseline.n_windows,
                                   "baseline_reasons": list(baseline.reasons)},
            "windows_total": len(windows),
            "windows_usable": sum(row["window"].usable for row in windows), "probes_total": len(probe_rows),
            "feature_rows_usable": sum(row["feature_usable"] for row in probe_rows),
            "train_eligible": sum(row["train_eligible"] for row in probe_rows),
            "test_eligible": sum(row["test_eligible"] for row in probe_rows)}


def export_session(session, output, *, recover=False, progress=None, native_db=None, native_record=None,
                   calibration=None, reprocess=False):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("导出目录已存在，请选择新的目录，避免覆盖既有实验数据。")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".ningsi-export-") as temp:
        stage = Path(temp) / "data"
        summary = _write_export(session, stage, recover, progress, native_db, native_record, calibration, reprocess)
        if output.exists():
            raise FileExistsError("导出目录已存在，请选择新的目录。")
        stage.rename(output)
    return summary


def _write_export(session, output, recover, progress, native_db, native_record, calibration, reprocess):
    source = open_readonly(session)
    try:
        metadata = read_metadata(source)
        if metadata["state"] == "recording" and not recover:
            raise ValueError("该记录尚未正常关闭。请先停止采集；进程异常退出的记录可用 --recover 保留式导出。")
        output.mkdir(parents=True)
        snapshot = sqlite3.connect(output / "recording.sqlite3")
        try:
            source.backup(snapshot)
            snapshot.execute("PRAGMA journal_mode=DELETE")
        finally:
            snapshot.close()
    finally:
        source.close()
    connection = open_readonly(output)
    try:
        metadata = read_metadata(connection)
        events = read_events(connection)
        actual = connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        source = metadata["source"]
        analysis_times = None
        alignment = None
        from . import biomulti
        original_source = dict(source)
        recomputed = reprocess and source.get("preprocessing_policy") is None
        if recomputed:
            source["preprocessing_policy"] = conditioning.POLICY
        legacy = "unit_selection" not in source and biomulti.local_publisher(source)
        if calibration:
            if (source.get("kind") != "lsl" or source.get("data_origin") != "real" or
                    (source.get("scale_to_uv") is not None and source.get("unit_basis") != "biomulti_software_inference")):
                raise ValueError("补录单位仅用于尚未确认单位的真实 LSL 源。")
            source.update(biomulti.confirmed_calibration(source, **calibration), unit_basis="documented_calibration")
            source.pop("unit_evidence", None)
        elif legacy and source.get("scale_to_uv") is None:
            source.update(biomulti.automatic_unit(source))
        if legacy and source.get("timestamp_policy") in (None, PACKET_TIMING, STRICT_TIMING):
            source["timestamp_policy"] = BIOMULTI_LSL_TIMING
        if native_record is not None or source.get("timestamp_policy") == BIOMULTI_REC_TIMING:
            from .biomulti import align, default_database
            if progress:
                progress("正在逐包核对 BioMulti 原生 REC 与 LSL 样本")
            bound = source.get("native_rec", {})
            record_id = bound.get("record_id", native_record)
            if record_id is None or (native_record is not None and native_record != record_id):
                raise ValueError("原生 REC 编号与采集时绑定的记录不符。")
            database = native_db or default_database()
            try:
                analysis_times, alignment, packets = align(connection, source, database, record_id, bound.get("created_ms"))
            except (OSError, ValueError, TypeError, KeyError, OverflowError, sqlite3.Error) as exc:
                # Preserve raw exports even if REC has not been saved or moved.
                # No derived device clock -> no eligible EEG features.
                alignment = {"status": "unavailable", "record_id": record_id, "reason": str(exc)}
            else:
                alignment["status"] = "matched"
                np.save(output / "device_timestamps.npy", analysis_times, allow_pickle=False)
                with (output / "device_packets.csv").open("w", encoding="utf-8-sig", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(("sample_start_index", "packet_number", "device_tick_ms", "outlet_last_lsl", "estimated_last_lsl"))
                    writer.writerows(packets)
            if source.get("timestamp_policy") != BIOMULTI_REC_TIMING:
                source["timestamp_policy"] = BIOMULTI_REC_TIMING
            source["native_rec"] = {**bound, "record_id": record_id}
        elif native_db is not None:
            raise ValueError("请指定原生 REC 编号。")
        elif source.get("timestamp_policy") == BIOMULTI_LSL_TIMING:
            _, analysis_times = packet_axis(connection, 0, None)
            np.save(output / "packet_timestamps.npy", analysis_times, allow_pickle=False)
        if original_source != source:
            metadata["source_at_capture"] = original_source
        write_json(output / "manifest.json", metadata)
        with (output / "events.jsonl").open("w", encoding="utf-8") as file:
            for event in events:
                file.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        csv_write(output / "events.csv", ("event_id", "kind", "time_lsl", "time_perf", "utc", "payload"),
                  ({**e, "payload": json.dumps(e["payload"], ensure_ascii=False)} for e in events))
        csv_write(output / "clock_offsets.csv", ("id", "measured_lsl", "measured_perf", "offset_sec", "error"),
                  (dict(row) for row in connection.execute("SELECT * FROM corrections ORDER BY id")))
        trials = [{**e["payload"], "result_lsl": e["time_lsl"]} for e in events if e["kind"] == "trial_result"]
        csv_write(output / "trials.csv", ("trial_id", "task", "phase", "block", "practice", "index", "digit",
                                          "nogo", "scheduled_perf", "onset_perf", "onset_lsl", "responded",
                                          "rt_sec", "valid", "invalid_reason", "correct", "omission", "lapse_500ms",
                                          "result_lsl"), trials)
        if progress:
            progress("正在导出逐样本原始数据")
        channels = metadata["source"]["channels"]
        with ((output / "raw_samples.npy").open("wb") as raw, (output / "timestamps.npy").open("wb") as times,
              (output / "samples.csv").open("w", encoding="utf-8-sig", newline="") as file):
            for binary, columns in ((raw, channels), (times, 4)):
                np.lib.format.write_array_header_1_0(binary, {"descr": "<f8", "fortran_order": False,
                                                              "shape": (actual, columns)})
            writer = csv.writer(file)
            timed = source.get("timestamp_policy") in (BIOMULTI_LSL_TIMING, BIOMULTI_REC_TIMING)
            header = ["sample_index", "source_timestamp", "corrected_timestamp", "received_lsl", "received_perf",
                      *[f"ch{index+1}_source_value" for index in range(channels)]]
            if timed:
                header.append("analysis_timestamp")
            writer.writerow(header)
            for row in connection.execute("SELECT * FROM samples ORDER BY sample_index"):
                values = np.frombuffer(row["values_f64"], dtype="<f8")
                original_time = np.frombuffer(row["source_timestamp_bytes"], dtype="<f8")[0]
                corrected = row["corrected_timestamp"]
                raw.write(row["values_f64"])
                times.write(struct.pack("<4d", original_time, corrected if corrected is not None else np.nan,
                                        row["received_lsl"], row["received_perf"]))
                fields = [row["sample_index"], original_time, corrected, row["received_lsl"], row["received_perf"], *values]
                if timed:
                    analysis = analysis_times[row["sample_index"]] if analysis_times is not None else np.nan
                    fields.append(float(analysis) if np.isfinite(analysis) else None)
                writer.writerow(fields)
        if progress:
            progress("正在计算脑电窗口和探针特征")
        fingerprint = engine_fingerprint()
        policy = source.get("preprocessing_policy")
        implementation = implementation_fingerprint(policy)
        changed = fingerprint != metadata.get("engine_fingerprint_at_capture")
        previous = metadata.get("implementation_fingerprint_at_capture")
        captured_impl = implementation_fingerprint() if recomputed else implementation
        changed = changed or (previous is not None and captured_impl != previous)
        summary = feature_tables(connection, metadata, events, output, changed, analysis_times)
        summary.update(schema="ningsi-capture-export-v1", checksum_required=True, session_id=metadata["session_id"],
                       protocol_id=metadata["protocol"].get("name"),
                       source_state=metadata["state"], raw_samples=actual, events=len(events),
                       protocol_complete=any(e["kind"] == "protocol_complete" for e in events),
                       source_origin=metadata["source"].get("data_origin"),
                       engine_fingerprint_at_export=fingerprint,
                       implementation_fingerprint_at_export=implementation,
                       engine_changed_since_capture=changed,
                       timing_status="software_onset_estimate; physical latency not validated",
                       timestamp_columns=["source_timestamp", "corrected_timestamp", "received_lsl", "received_perf"],
                       timestamp_policy=metadata["source"].get("timestamp_policy", "source-timestamps-v1"),
                       timestamp_estimate="optional per-context nominal grid, <=50ms shift; raw timestamps unchanged; gaps >100ms rejected",
                       label_policy="1=on_task(0);3/5=internal_mind_wandering(1);others=unlabelled",
                       lookback_sec=[-12, -2], filtering="past context up to 6 seconds; last 4 seconds analyzed",
                       window_gate="engine strict quality plus finite synchronized monotonic timestamps; gap <=100ms; rate error <=5%",
                       stimulus_timing=metadata["stimulus_timing"])
        if policy:
            summary.update(preprocessing=conditioning.description(source["srate"]), preprocessing_recomputed=recomputed,
                           cleaned_windows_file="cleaned_windows_uv.npy", mask_file="artifact_mask.npy",
                           filtering="causal 2s FIR history; 4s windows; artifact spans are NaN; rejected windows have no training features")
        if alignment is not None:
            summary.update(native_alignment=alignment,
                           timestamp_estimate="matched REC device packet ticks aligned offline to LSL; unmatched samples have no analysis time")
        elif source.get("timestamp_policy") == BIOMULTI_LSL_TIMING:
            summary.update(timestamp_estimate="BioMulti LSL 10-sample packet release estimate, on the task LSL clock",
                           packet_clock_file="packet_timestamps.npy",
                           packet_clock_valid_samples=int(np.isfinite(analysis_times).sum()))
    finally:
        connection.close()
    # Publish the discovery marker only after every member has its final bytes.
    pending = output / "summary.pending"
    write_json(pending, summary)
    checksums = {}
    for path in output.iterdir():
        with path.open("rb") as file:
            checksums["export.json" if path == pending else path.name] = hashlib.file_digest(file, "sha256").hexdigest()
    write_json(output / "SHA256.json", checksums)
    pending.rename(output / "export.json")
    return summary


def build_dataset(exports, output):
    """Use explicit participant cohorts; never split windows or mix demo rows."""
    from ningsi.models.human_data import files_from_paths, import_files

    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("数据集目录已经存在。")
    exports = [Path(p).resolve() for p in exports]
    if not exports or any(not (p / "export.json").is_file() for p in exports):
        raise ValueError("合并需要至少一份完整的采集导出。")
    data = import_files(files_from_paths(exports))
    output.mkdir(parents=True)
    for role in ("development", "test", "excluded"):
        csv_write(output / (role + ".csv"), PROBE_FIELDS, data[role])
    manifest = {"schema": "ningsi-capture-dataset-v1", "feature_names": list(FEATURES), "label_column": "label",
                "group_column": "participant", "participants": data["participants"],
                "development_rows": len(data["development"]), "test_rows": len(data["test"]),
                "excluded_rows": len(data["excluded"]),
                "policy": "Group cross-validation by participant in development only; test remains held out.",
                "input_exports": [str(Path(p).resolve()) for p in exports],
                "engine_fingerprint": data["engine_fingerprint"],
                "implementation_fingerprint": data["implementation_fingerprint"],
                "preprocessing_policy": (conditioning.POLICY if conditioning.FILE in (data["implementation_fingerprint"] or {}) else None),
                "source_profiles": data["source_profiles"], "session_metadata": data["session_metadata"],
                "file_sha256": {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                                for name in ("development.csv", "test.csv", "excluded.csv")}}
    write_json(output / "dataset.json", manifest)
    return manifest
