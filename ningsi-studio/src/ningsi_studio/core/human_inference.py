"""Stream inference using the exact capture window/aggregation contract."""

from __future__ import annotations

from collections import deque
from dataclasses import replace
import math
import re

import numpy as np

from ningsi.acquisition.eeg import unit_conflict
from ningsi.acquisition.timing import (BIOMULTI_CONFIRM_SAMPLES, BIOMULTI_LSL_TIMING, BIOMULTI_REC_TIMING, BIOMULTI_WARM_SAMPLES,
                                      STRICT_TIMING, analysis_timestamps)
from ningsi.models.human import explain, validate_model
from ningsi.models.logistic import FEATURE_NAMES
from ningsi.models.research_features import aggregate_probe, engine_fingerprint
from ningsi.signal import conditioning
from ningsi.signal.artifacts import Verdict
from ningsi.signal.blinks import process_blinks


def veto_live_blinks(row, snap, profile, end):
    """Reject raw dual-channel transients without changing frozen model features.

    This is an extra live eligibility check, outside snapshot_window's training
    contract. A personal correction used by the relative-index path must never
    be silently substituted into a previously trained classifier's input.
    """
    if not row["window"].usable:
        return row
    values = np.asarray(snap["samples"], dtype=float)
    if values.ndim != 2 or values.shape[0] != 2:
        return row
    rate = float(profile["srate"])
    offset = snap["total_samples"] - values.shape[1]
    first = end - round(4 * rate) - offset
    last = end - offset
    if first < 0 or last > values.shape[1]:
        return row
    # The generic detector handles per-channel DC and never subtracts one EEG
    # from the other. It sees only the samples belonging to this model window.
    result = process_blinks(values[:, first:last] * float(profile["scale_to_uv"]), rate,
                            channel_labels=tuple(profile.get("channel_labels") or ("Ch1", "Ch2")))
    if not result.reasons:
        return row
    quality = row["window"].quality
    reasons = tuple(dict.fromkeys((*quality.reasons, *result.reasons)))
    verdict = Verdict(False, reasons, {**quality.metrics, "live_blink_veto": result.as_dict()})
    return {**row, "window": replace(row["window"], quality=verdict), "features": {}}


def snapshot_window(snap, profile, end, *, traces=False):
    rate = float(profile["srate"])
    policy = profile.get("timestamp_policy")
    confirm = BIOMULTI_CONFIRM_SAMPLES if policy == BIOMULTI_LSL_TIMING else 0
    stamps = np.asarray(snap["timestamps"], dtype=float)
    offset = snap["total_samples"] - len(stamps)
    start = max(0, end - round(6 * rate))
    warm = max(0, start - BIOMULTI_WARM_SAMPLES) if confirm else start
    if end < round(4 * rate) or warm < offset or end + confirm > snap["total_samples"]:
        return None
    timing, audit = analysis_timestamps(stamps[warm-offset:end+confirm-offset], rate, policy, first_index=warm)
    timing = timing[start-warm:end-warm]
    if confirm:
        valid = np.isfinite(timing)
        audit["applied"] = bool(valid.all())
        delta = timing[valid] - stamps[start-offset:end-offset][valid]
        audit["max_shift_sec"] = float(np.max(np.abs(delta))) if delta.size else 0.
    values = np.asarray(snap["samples"], dtype=float)[:, start-offset:end-offset]
    row = conditioning.condition(values, timing, rate, profile["scale_to_uv"],
                                 profile.get("preprocessing_policy"), traces=traces)
    row["timestamp_adjustment"] = audit
    if traces:
        row["analysis_timestamps"] = timing[-round(4 * rate):]
    return row


class HumanAttentionPredictor:
    def __init__(self, bundle):
        self.bundle = validate_model(bundle)
        if bundle["engine_fingerprint"] != engine_fingerprint():
            raise ValueError("feature_engine_mismatch")
        policy = next((p.get("preprocessing_policy") for p in bundle["source_profiles"]), None)
        if bundle.get("implementation_fingerprint") != conditioning.implementation(policy):
            raise ValueError("feature_implementation_mismatch")
        self.windows = deque(maxlen=24)
        self.processed = 0
        self.buffer_id = None
        self.connection_generation = None
        self.phase = None
        self.phase_start = math.inf

    def unavailable(self, reason):
        return {"available": False, "model_id": self.bundle["model_id"], "reason": reason,
                "mind_wandering_probability": None, "on_task_probability": None}

    def match_profile(self, descriptor):
        text = (str(descriptor.get("name", "")) + " " + str(descriptor.get("source_id", ""))).lower()
        if re.search(r"(^|[^a-z])(sim|simulation|synthetic|demo)([^a-z]|$)", text):
            raise ValueError("simulation_source")
        matched = []
        for profile in self.bundle["source_profiles"]:
            if (profile.get("name") == descriptor.get("name")
                    and profile.get("channels") == descriptor.get("channel_count")
                    and abs(float(profile.get("srate") or 0) - float(descriptor.get("nominal_srate") or 0)) < .01
                    and profile.get("channel_labels") == descriptor.get("channel_labels")
                    and (not profile.get("source_id") or profile["source_id"] == descriptor.get("source_id"))):
                matched.append(profile)
        if any(p.get("timestamp_policy") == BIOMULTI_REC_TIMING for p in matched):
            raise ValueError("native_rec_clock_required")
        factors = {p.get("scale_to_uv") for p in matched}
        policies = {p.get("timestamp_policy") or STRICT_TIMING for p in matched}
        if len(factors) != 1 or None in factors or len(policies) != 1:
            raise ValueError("device_channels_units_mismatch")
        if any(unit_conflict(p) or unit_conflict({**p, "declared_units": descriptor.get("declared_units")})
               for p in matched):
            raise ValueError("device_channels_units_mismatch")
        return matched[0]

    def update(self, source, phase, *, baseline_valid):
        if not hasattr(source, "model_snapshot"):
            return self.unavailable("real_timestamped_eeg_required")
        snap = source.model_snapshot()
        if not snap.get("research_timing"):
            return self.unavailable("timestamp_processing_mismatch")
        if snap.get("age") is None or snap["age"] > 3:
            self.windows.clear()
            return self.unavailable("signal_stale")
        try:
            profile = self.match_profile(snap["descriptor"])
        except ValueError as exc:
            return self.unavailable(str(exc))
        rate = float(profile["srate"])
        stamps = np.asarray(snap["timestamps"], dtype=float)
        values = np.asarray(snap["samples"], dtype=float)
        if stamps.size < round(4 * rate) or values.ndim != 2 or not np.isfinite(stamps[-1]):
            return self.unavailable("insufficient_raw_history")
        onset = float(stamps[-1] + 1 / rate)
        if (self.buffer_id != snap["buffer_id"]
                or self.connection_generation != snap.get("connection_generation")
                or snap["total_samples"] < self.processed):
            self.windows.clear()
            self.processed = 0
            self.phase = None
            self.buffer_id = snap["buffer_id"]
            self.connection_generation = snap.get("connection_generation")
        if self.phase != phase:
            self.windows.clear()
            self.phase = phase
            self.phase_start = onset
        size, hop = round(4 * rate), round(2 * rate)
        count = snap["total_samples"]
        first = max(size, ((self.processed - size) // hop + 1) * hop + size)
        policy = profile.get("timestamp_policy")
        confirm = BIOMULTI_CONFIRM_SAMPLES if policy == BIOMULTI_LSL_TIMING else 0
        for end in range(first, count - confirm + 1, hop):
            self.processed = end
            row = snapshot_window(snap, profile, end)
            if row is not None:
                self.windows.append(veto_live_blinks(row, snap, profile, end))
        summary = aggregate_probe(self.windows, onset, baseline_valid=baseline_valid, phase_start=self.phase_start)
        if summary["reasons"]:
            return {**self.unavailable("|".join(summary["reasons"])), **{k: v for k, v in summary.items() if k != "features"}}
        result = explain(self.bundle, [summary["features"][name] for name in FEATURE_NAMES])
        result.update({k: v for k, v in summary.items() if k != "features"})
        result["timing"] = "past-only 6s conditioning; 4s windows/2s hop; mean fully inside [now-12,now-2]"
        result["domain_warning"] = "" if phase == "sart" else "当前阶段超出 SART 探针验证范围，结果仅供探索性观察。"
        result["reference_from_capture"] = profile.get("reference")
        return result
