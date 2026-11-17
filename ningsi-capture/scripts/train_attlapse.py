"""Fit the external shadow model from the versioned ATTLAPSE healthy-control cache."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ningsi" / "src"))
from ningsi.models import attlapse, human
from ningsi.models.human_data import digest
from ningsi.models.research_features import CONTRACT, aggregate_probe, engine_fingerprint, implementation_fingerprint


EOG_CHANNELS = ("eog_up", "eog_down", "eog_left", "eog_right")


def remove_source_eog(probe, values, timestamps):
    """Regress the source's bipolar ocular channels only up to two seconds before the probe."""
    names = probe["aux_channels"]
    if any(names.count(name) != 1 or name in probe["bad_aux_channels"] for name in EOG_CHANNELS):
        return None, "source_eog_missing_or_bad"
    indices = [names.index(name) for name in EOG_CHANNELS]
    if any(probe["aux_units"][index] != "µV" for index in indices):
        return None, "source_eog_units_unknown"
    vertical_up, vertical_down, left, right = np.asarray(probe["aux_values"], dtype=float)[indices]
    regressors = np.stack([vertical_up - vertical_down, left - right])
    available = (timestamps >= probe["onset_sec"] - 18) & (timestamps <= probe["onset_sec"] - 2)
    if (not np.isfinite(regressors[:, available]).all() or not np.isfinite(values[:, available]).all()
            or available.sum() < 4 * probe["rate"]):
        return None, "source_eog_context_invalid"
    centered_eog = regressors[:, available] - regressors[:, available].mean(axis=1, keepdims=True)
    centered_eeg = values[:, available] - values[:, available].mean(axis=1, keepdims=True)
    coefficients, _, rank, _ = np.linalg.lstsq(centered_eog.T, centered_eeg.T, rcond=None)
    if rank < 2:
        return None, "source_eog_flat_or_dependent"
    cleaned = values.copy()
    cleaned[:, available] -= coefficients.T @ centered_eog
    return cleaned, None


def extract_probe(probe):
    rate = float(probe["rate"])
    onset = float(probe["onset_sec"])
    timestamps = np.asarray(probe["timestamps"], dtype=float)
    values = np.asarray(probe["values_uv"], dtype=float)
    if rate != 512 or values.shape != (2, len(timestamps)) or not len(timestamps):
        raise ValueError("Expected the 512 Hz Fp1/Fp2 ATTLAPSE waveform.")
    start_sample = round(timestamps[0] * rate)
    if not np.allclose(timestamps, (np.arange(len(timestamps)) + start_sample) / rate, atol=1e-7, rtol=0):
        raise ValueError("Inconsistent BIDS sample times.")
    size, hop, context = round(4 * rate), round(2 * rate), round(6 * rate)
    first = max(size, int(np.ceil((onset - 8) * rate / hop)) * hop)
    last = int(np.floor((onset - 2) * rate / hop)) * hop
    row = {"participant": probe["participant"], "probe_id": probe["probe_id"],
           "source_answer": probe["source_answer"], "source_score": probe.get("source_score"),
           "label": probe["label"], "onset_sec": onset}
    source_issue = None
    if probe.get("bad_channels"):
        source_issue = "source_channel_marked_bad:" + ",".join(probe["bad_channels"])
    elif "aux_values" in probe:
        values, source_issue = remove_source_eog(probe, values, timestamps)
    if source_issue:
        return {**row, "windows_total": 0, "windows_usable": 0, "usable_seconds": 0.,
                "reasons": source_issue}
    windows = []
    for end in range(first, last + 1, hop):
        lo, hi = max(0, end - context) - start_sample, end - start_sample
        if lo < 0 or hi > len(timestamps):
            raise ValueError("Probe excerpt is missing causal EEG context.")
        windows.append(attlapse.bipolar_window(values[:, lo:hi], timestamps[lo:hi], rate, 1.0, (0, 1)))
    # ATTLAPSE has no verified Ningsi eyes-open baseline. Our six spectral inputs
    # do not use a baseline value; local Capture gates separately on its own calibration.
    result = aggregate_probe(windows, onset, baseline_valid=True, phase_start=probe["phase_start_sec"])
    return {**row,
            "windows_total": result["windows_total"], "windows_usable": result["windows_usable"],
            "usable_seconds": result["usable_seconds"], "reasons": "|".join(result["reasons"]),
            **result["features"]}


def fit_candidate(rows, raw_rows=None):
    usable = [r for r in rows if not r["reasons"] and r["label"] in (0, 1)]
    x = human.matrix(usable)
    y = np.asarray([r["label"] for r in usable])
    oof = np.zeros(len(y))
    splits = []
    raw_tests, raw_scores = [], []
    raw_usable = [r for r in (raw_rows or []) if not r["reasons"] and r["label"] in (0, 1)]
    # Fix the regularization in advance. Each prediction below uses an unseen person.
    l2 = 0.01
    for train, heldout, split in human.group_folds(usable):
        training = [usable[i] for i in train]
        params = human.fit(x[train], y[train], human.person_weights(training, True), l2)
        oof[heldout] = human.sigmoid(human.logits(params, x[heldout]))
        uncorrected = [r for r in raw_usable if r["participant"] in split["validation_participants"]]
        if uncorrected:
            raw_tests.extend(uncorrected)
            raw_scores.extend(human.sigmoid(human.logits(params, human.matrix(uncorrected))).tolist())
        splits.append(split)
    validation = human.evaluate(usable, oof, 0.5)
    parameters = human.fit(x, y, human.person_weights(usable, True), l2)
    model = {"spec": attlapse.MODEL_SPEC, "source_doi": attlapse.DATA_DOI, "cohort": "healthy_controls",
             "attribution": "Horáková, Weibel, Tissot, Pham, Costache & Bonnefond. ATTLAPSE v2.0.1 (CC BY 4.0).",
             "montage": attlapse.MONTAGE, "derivation": "Fp1 minus Fp2, after quality checks on both source channels",
             "source_preprocessing": "Source only: subtract the least-squares projection onto centered bipolar vertical and horizontal EOG, per EEG channel, using [-18,-2] seconds per probe; then apply the unmodified quality and spectral pipeline. No answer or post-probe data used. Live Capture uses physical QC without EOG channels.",
             "labels": attlapse.SOURCE_LABELS, "threshold": 0.5, "parameters": parameters, "l2": l2,
             "feature_contract": CONTRACT, "source_rate_hz": 512,
             "baseline_policy": "Source training uses absolute spectral features with no verified Ningsi baseline; live scoring requires a valid local calibration.",
             "intended_use": "Blinded research score at scheduled SART probes; target-device performance and calibration unverified.",
             "training_participants": sorted({r["participant"] for r in usable}),
             "training_probes": len(usable), "feature_rows_sha256": digest(rows),
             "montage_fingerprint": attlapse.montage_fingerprint(),
             "engine_fingerprint": engine_fingerprint(), "implementation_fingerprint": implementation_fingerprint()}
    model["model_id"] = attlapse.candidate_id(model)
    ci = human.bootstrap_mean([p["auc"] for p in validation["per_participant"] if p["auc"] is not None],
                              human.TRAINING_CONFIG["seed"])
    report = {"source_doi": attlapse.DATA_DOI, "model_id": model["model_id"], "subjects_seen": sorted({r["participant"] for r in rows}),
              "total_probes": len(rows), "usable_probes": len(usable), "class_counts": dict(Counter(int(i) for i in y)),
              "exclusions": dict(Counter(r["reasons"] or "unmapped_answer" for r in rows
                                         if r["reasons"] or r["label"] not in (0, 1))),
              "evaluation": "Participant-held-out external-data fivefold CV; no local EEG evaluation.",
              "folds": splits, "heldout_metrics": validation, "macro_auc_person_bootstrap_ci95": ci}
    if raw_rows is not None:
        report["without_source_eog_correction"] = {
            "usable_probes_before_subject_matching": len(raw_usable), "heldout_probes": len(raw_tests),
            "description": "Same fitted external folds, scored on original Fp1/Fp2 without ocular regression and with identical strict quality gates.",
            "metrics": human.evaluate(raw_tests, raw_scores, 0.5) if raw_tests else None}
        for row, score in zip(raw_tests, raw_scores):
            row["heldout_score_without_eog_correction"] = float(score)
    for row, prob in zip(usable, oof):
        row["heldout_external_score"] = float(prob)
    return model, report


def main():
    from attlapse_data import iter_probes
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, raw_rows = [], []
    for probe in iter_probes(args.data, with_aux=True):
        rows.append(extract_probe(probe))
        raw_rows.append(extract_probe({k: v for k, v in probe.items() if k != "aux_values"}))
    people = {r["participant"] for r in rows}
    if (len(people) != 28 or set(Counter(r["participant"] for r in rows).values()) != {15}
            or len(set((r["participant"], r["probe_id"]) for r in rows)) != len(rows)):
        raise ValueError(f"Expected all 28 distinct controls with unique probes, got {len(people)} participants.")
    model, report = fit_candidate(rows, raw_rows)
    for row, raw in zip(rows, raw_rows, strict=True):
        row["raw_probe_qc"] = raw["reasons"] or "usable"
        row["heldout_score_without_eog_correction"] = raw.get("heldout_score_without_eog_correction")
    source_checksums = {}
    for person in sorted(people):
        folder = args.data / "cache" / person
        eeg = json.loads((folder / "probes.json").read_text(encoding="utf-8"))
        aux = json.loads((folder / "source_aux.json").read_text(encoding="utf-8"))
        source_checksums[person] = {"archive_eeg_crc32": eeg["eeg_zip_crc32"],
                                    "probe_eeg_sha256": eeg["eeg_npy_sha256"],
                                    "probe_aux_sha256": aux["sha256"]}
    report["source_checksums"] = source_checksums
    args.output.mkdir(parents=True, exist_ok=False)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (args.output / "source_probes.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    human.write_json(args.output / "external_validation.json", report)
    human.write_json(args.output / attlapse.MODEL_FILE, model)
    attlapse.load_candidate(args.output / attlapse.MODEL_FILE)
    print(f"{len(people)} controls, {report['usable_probes']}/{len(rows)} usable probes, model {model['model_id']}.")


if __name__ == "__main__":
    main()
