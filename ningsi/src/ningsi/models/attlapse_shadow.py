"""Blind 250 Hz ATTLAPSE comparison using the shared Fp1−Fp2 input."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from .human import logits, sigmoid
from .human_data import digest
from .logistic import FEATURE_NAMES, features_from_window
from .research_features import CONTRACT, condition_window, union_seconds


MODEL_FILE = "attlapse_shadow_250.json"
BUNDLE_ID = "dac6ee0057ddb8f952ea4a42f5f9287453caffeb05bf2518e2ae8e5d96db580e"
RATE = 250
LOOKBACK = (-18, -2)
FAMILIES = ("logistic", "rbf")
PROXY_FLAGS = {"eog", "emg"}
# Sub-sample tolerance for adding the sample period to a large LSL clock value.
CLOCK_EPS = 1e-8


def load_candidates(path=None):
    """Both models come from the same frozen public-data experiment."""
    path = Path(path) if path is not None else Path(__file__).with_name(MODEL_FILE)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != BUNDLE_ID:
        raise ValueError("250 Hz ATTLAPSE candidate bundle checksum mismatch")
    payload = json.loads(data)
    if (payload["input_rate_hz"] != RATE or tuple(payload["feature_names"]) != FEATURE_NAMES
            or tuple(payload["lookback_sec"]) != LOOKBACK or payload["quality"]["policy"] != "physical"):
        raise ValueError("Unexpected ATTLAPSE candidate feature contract")
    params = {family: payload["models"][family]["fitted_parameters"] for family in FAMILIES}
    return {"bundle_id": BUNDLE_ID, "model_ids": {k: digest(v) for k, v in params.items()},
            "parameters": params}


def score_features(model, features):
    vector = np.asarray([features[name] for name in FEATURE_NAMES], dtype=float)
    if vector.shape != (len(FEATURE_NAMES),) or not np.isfinite(vector).all():
        raise ValueError("Invalid probe features")
    linear = model["parameters"]["logistic"]
    rbf = model["parameters"]["rbf"]
    z = (vector - rbf["mean"]) / rbf["scale"]
    distance = np.sum(np.square(z - np.asarray(rbf["train"])), axis=1)
    kernel = np.exp(-distance / len(vector))
    return {"logistic_score": float(sigmoid(logits(linear, vector))),
            "rbf_score": float(sigmoid(kernel @ rbf["dual"] + rbf["bias"]))}


def compute_window(values, timestamps, rate, factor, indices):
    """Condition one recording-aligned context; retain strict and physical QC independently."""
    if rate != RATE:
        raise ValueError("Expected a 250 Hz stream")
    pair = np.asarray(values, dtype=float)[list(indices)]
    stamps = np.asarray(timestamps, dtype=float)
    if pair.ndim != 2 or pair.shape != (2, len(stamps)) or len(stamps) > RATE * CONTRACT["context_sec"]:
        raise ValueError("Invalid two-channel window context")
    inputs = condition_window(pair, stamps, rate, factor)
    diff = condition_window(pair[0:1] - pair[1:2], stamps, rate, factor)
    # Timing and missing-data errors occur before the shared conditioner can
    # compute spectra. Preserve those codes for the acquisition operator.
    strict, physical = [], []
    for prefix, conditioned in (("input", inputs["window"]), ("bipolar", diff["window"])):
        for reason in conditioned.quality.reasons:
            name = f"{prefix}_{reason}" if conditioned.rel else reason
            if name not in strict:
                strict.append(name)
            if reason not in PROXY_FLAGS and name not in physical:
                physical.append(name)
    return {"start": inputs["start"], "end": inputs["end"], "physical_reasons": physical,
            "strict_reasons": strict, "features": {} if physical else features_from_window(diff["window"])}


def _quality(rows, total, gate, phase_start, left):
    reason_key = f"{gate}_reasons"
    accepted = [row for row in rows if not row[reason_key]]
    rejected = Counter(reason for row in rows for reason in row[reason_key])
    if total > len(rows):
        rejected["missing_window"] = total - len(rows)
    seconds = union_seconds((row["start"], row["end"]) for row in accepted)
    reasons = []
    if phase_start is None or phase_start > left + CLOCK_EPS:
        reasons.append("lookback_crosses_phase")
    if (len(accepted) < CONTRACT["min_windows"] or seconds + CLOCK_EPS < CONTRACT["min_seconds"]
            or len(accepted) < CONTRACT["min_valid_ratio"] * total):
        reasons.append("insufficient_usable_eeg")
    return accepted, seconds, reasons, dict(rejected)


def aggregate_and_score(model, rows, onset, phase_start, *, expected_windows=None):
    """Aggregate fully contained analysis windows; count absent windows against coverage.

    The filtering context precedes each analysis window. No response, task
    progress feature or calibration is used in the two frozen classifiers.
    """
    left, right = onset + LOOKBACK[0], onset + LOOKBACK[1]
    candidates = [row for row in rows if row["start"] is not None and row["end"] is not None
                  and left - CLOCK_EPS <= row["start"] and row["end"] <= right + CLOCK_EPS]
    # With a complete 2-s grid, 6 or 7 full 4-s windows fall in this 16-s interval.
    total = max(len(candidates), 6, expected_windows or 0)
    usable, seconds, physical, p_counts = _quality(candidates, total, "physical", phase_start, left)
    strict, s_seconds, s_reasons, s_counts = _quality(candidates, total, "strict", phase_start, left)
    result = {"window_start_lsl": left, "window_end_lsl": right, "windows_total": total,
              "windows_usable": len(usable), "usable_seconds": seconds,
              "physical_qc_status": "failed" if physical else "passed", "physical_qc_reasons": physical,
              "physical_window_rejections": p_counts, "strict_windows_usable": len(strict),
              "strict_usable_seconds": s_seconds, "strict_qc_status": "failed" if s_reasons else "passed",
              "strict_qc_reasons": s_reasons, "strict_window_rejections": s_counts,
              "features": {}, "logistic_score": None, "rbf_score": None}
    if physical:
        result["reason"] = "|".join(physical)
    else:
        features = {name: float(np.mean([row["features"][name] for row in usable])) for name in FEATURE_NAMES}
        result.update(features=features, **score_features(model, features))
    return result
