"""Join the frozen 250 Hz EEG results, task timing and responses at the displayed probe."""

from __future__ import annotations

import json

from ningsi.models.capture_protocols import RESEARCH_PROTOCOLS


TCN_FIELDS = ("model_id", "input_policy", "error_type", "window_start_lsl", "window_end_lsl", "windows_total",
              "windows_usable", "recovered_windows", "usable_seconds", "frame_mask", "physical_qc_status",
              "strict_qc_status", "strict_windows_usable",
              "strict_usable_seconds")
TCN_JSON_FIELDS = ("frame_end_sample_indices", "physical_window_rejections", "strict_window_rejections")

FIELDS = (
    "participant", "visit", "session_id", "cohort", "data_origin", "protocol_id", "session_state", "protocol_complete",
    "onset_event_id", "probe_id", "phase", "block", "practice", "probe_ordinal", "after_trial",
    "task_elapsed_sec", "protocol_elapsed_sec", "onset_lsl", "onset_perf", "onset_basis",
    "window_start_lsl", "window_end_lsl", "sample_count_at_onset",
    "status", "reason", "error_type", "quality_error_type", "logistic_mw_score", "rbf_mw_score",
    "model_bundle_id", "logistic_model_id", "rbf_model_id",
    "windows_total", "windows_usable", "usable_seconds", "physical_qc_status",
    "physical_qc_reasons", "physical_window_rejections",
    "strict_windows_usable", "strict_usable_seconds", "strict_qc_status", "strict_qc_reasons",
    "strict_window_rejections", "baseline_valid",
    "live_input_window_status", "live_input_window_reasons", "live_input_window_start_lsl",
    "live_input_window_end_lsl", "live_input_window_sample_end_index",
    "live_input_last_packet_age_sec", "live_input_window_received_age_sec",
    "response_event_id", "response_lsl", "response_status", "response_rt_sec",
    "choice", "choice_text", "label", "label_name", "time_comparable", "comparable",
    "tcn_status", "tcn_reason", "tcn_mw_score", "tcn_comparable", "tcn_paired_comparable",
    *(f"tcn_{key}" for key in (*TCN_FIELDS, *TCN_JSON_FIELDS)),
)


def _join_responses(events):
    """Pair with the open onset, so an interrupted probe cannot inherit a later answer."""
    opened, terminal = {}, {}
    for event in events:
        kind = event["kind"]
        if kind == "probe_onset":
            opened[event["payload"]["probe_id"]] = event["event_id"]
        elif kind in ("probe_response", "probe_cancelled"):
            onset_id = opened.pop(event["payload"]["probe_id"], None)
            if onset_id is not None:
                terminal[onset_id] = event
    return terminal


def _response_columns(event, choices):
    payload = event["payload"] if event else {}
    choice = payload.get("choice")
    label = None
    if event is None:
        status = "unanswered"
    elif event["kind"] == "probe_cancelled":
        status = "cancelled"
    elif payload.get("timeout"):
        status = "timeout"
    elif choice in (1, 3, 5):
        status = "binary"
        label = int(choice != 1)
    else:
        status = "nonbinary"
    text = choices[choice - 1] if type(choice) is int and 1 <= choice <= len(choices) else None
    return {"response_event_id": event["event_id"] if event else None,
            "response_lsl": event["time_lsl"] if event else None, "response_status": status,
            "response_rt_sec": payload.get("response_rt_sec"), "choice": choice, "choice_text": text,
            "label": label, "label_name": {0: "on_task", 1: "internal_mind_wandering"}.get(label)}


def comparison_table(metadata, events, predictions, output, csv_write):
    terminal = _join_responses(events)
    common = {key: metadata[key] for key in ("session_id", "visit", "participant", "cohort")}
    common.update(data_origin=metadata["source"].get("data_origin"), protocol_id=metadata["protocol"].get("name"),
                  session_state=metadata.get("state"), protocol_complete=any(e["kind"] == "protocol_complete" for e in events))
    rows = []
    for event in events:
        if event["kind"] != "probe_onset":
            continue
        payload = event["payload"]
        event_id, onset = event["event_id"], event["time_lsl"]
        record = predictions.get(event_id, {})
        shadow = record.get("shadow_250") or {}
        tcn = record.get("tcn_250") or {}
        # Historical sessions have no candidate record. Never rescore them at export.
        status = shadow.get("status", "not_recorded")
        model_ids = shadow.get("model_ids") or {}
        response = _response_columns(terminal.get(event_id), payload.get("choices") or [])
        logistic = shadow.get("logistic_score") if status == "scored" else None
        rbf = shadow.get("rbf_score") if status == "scored" else None
        tcn_status = tcn.get("status", "not_recorded")
        tcn_score = tcn.get("mw_score") if tcn_status == "scored" else None
        time_comparable = (common["data_origin"] == "real" and common["protocol_id"] in RESEARCH_PROTOCOLS
                           and not payload.get("practice") and response["label"] is not None
                           and payload.get("task_elapsed_sec") is not None and payload.get("probe_ordinal") is not None)
        row = {**common, **{key: payload.get(key) for key in ("probe_id", "phase", "block", "after_trial",
               "probe_ordinal", "task_elapsed_sec", "protocol_elapsed_sec", "onset_basis")},
               "practice": bool(payload.get("practice")), "onset_event_id": event_id,
               "onset_lsl": onset, "onset_perf": event["time_perf"],
               "window_start_lsl": shadow.get("window_start_lsl", onset - 18),
               "window_end_lsl": shadow.get("window_end_lsl", onset - 2),
               "sample_count_at_onset": shadow.get("sample_count_at_onset", record.get("sample_count_at_onset")),
               "status": status, "reason": shadow.get("reason", "" if shadow else "no_live_shadow_record"),
               "logistic_mw_score": logistic, "rbf_mw_score": rbf, "logistic_model_id": model_ids.get("logistic"),
               "rbf_model_id": model_ids.get("rbf"), "model_bundle_id": shadow.get("model_bundle_id"),
               **response, "time_comparable": time_comparable,
               "comparable": time_comparable and logistic is not None and rbf is not None,
               "tcn_status": tcn_status, "tcn_reason": tcn.get("reason", "" if tcn else "no_live_tcn_record"),
               "tcn_mw_score": tcn_score, "tcn_comparable": time_comparable and tcn_score is not None}
        row["tcn_paired_comparable"] = row["comparable"] and row["tcn_comparable"]
        row.update({f"tcn_{key}": tcn.get(key) for key in TCN_FIELDS})
        for key in TCN_JSON_FIELDS:
            if key in tcn:
                row[f"tcn_{key}"] = json.dumps(tcn[key], sort_keys=True, separators=(",", ":"))
        for key in ("error_type", "quality_error_type", "windows_total", "windows_usable", "usable_seconds", "physical_qc_status",
                    "strict_windows_usable", "strict_usable_seconds", "strict_qc_status", "baseline_valid",
                    "live_input_window_status", "live_input_window_start_lsl", "live_input_window_end_lsl",
                    "live_input_window_sample_end_index", "live_input_last_packet_age_sec",
                    "live_input_window_received_age_sec"):
            row[key] = shadow.get(key)
        for key in ("physical_qc_reasons", "strict_qc_reasons", "live_input_window_reasons"):
            if key in shadow:
                row[key] = "|".join(shadow[key])
        for key in ("strict_window_rejections", "physical_window_rejections"):
            if key in shadow:
                row[key] = json.dumps(shadow[key], sort_keys=True, separators=(",", ":"))
        rows.append(row)
    csv_write(output / "probe_comparison.csv", FIELDS, rows)
    return {"file": "probe_comparison.csv", "schema": "ningsi-shadow-probe-comparison-v1",
            "probes_total": len(rows), "scored": sum(r["status"] == "scored" for r in rows),
            "not_recorded": sum(r["status"] == "not_recorded" for r in rows),
            "unavailable": sum(r["status"] == "unavailable" for r in rows),
            "time_comparable": sum(r["time_comparable"] for r in rows),
            "comparable": sum(r["comparable"] for r in rows), "lookback_sec": [-18, -2],
            "tcn": {"scored": sum(r["tcn_status"] == "scored" for r in rows),
                    "unavailable": sum(r["tcn_status"] == "unavailable" for r in rows),
                    "not_recorded": sum(r["tcn_status"] == "not_recorded" for r in rows),
                    "comparable": sum(r["tcn_comparable"] for r in rows),
                    "paired_comparable": sum(r["tcn_paired_comparable"] for r in rows)},
            "model_input_rate_hz": 250, "score_target": "internal mind wandering; uncalibrated ranking scores",
            "ordinal_definition": "one-based across formal SART probes within this session; practice excluded",
            "task_elapsed_definition": "monotonic seconds since first formal SART phase start, including later breaks",
            "protocol_elapsed_definition": "monotonic seconds since protocol_start",
            "quality_definition": "physical and strict refer to the same 16-second probe interval. "
                                  "Live input QC describes the latest aligned standard 4-second window available at onset.",
            "comparison_definition": "time_comparable requires a binary answer, timing and ordinal on a formal real "
                                     "research probe. comparable additionally requires both frozen EEG scores. Compare "
                                     "EEG and time on identical comparable rows grouped by participant; use cohort and "
                                     "manifest state to select the intended population."}
