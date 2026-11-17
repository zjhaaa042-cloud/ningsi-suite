"""Refit two specified same-input baselines for offline research comparison."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from attlapse_two_channel_eval import Candidate, PARAMETERS, SEED, describe, fit, load_features, paired, predict
from attlapse_two_channel_features import FEATURE_SETS


def refit(feature_path, directory):
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    if hashlib.sha256(feature_path.read_bytes()).hexdigest() != report["feature_file_sha256"]:
        raise ValueError("Feature file differs from the nested validation source")
    with feature_path.open(encoding="utf-8") as file:
        rates = {json.loads(line)["rate"] for line in file}
    if len(rates) != 1 or not rates <= {250, 512}:
        raise ValueError(f"Unexpected sampling rates {rates}")
    rate = int(next(iter(rates)))
    keys, groups, labels, matrices = load_features(feature_path, FEATURE_SETS)
    with (directory / "inner_selection.csv").open(encoding="utf-8") as file:
        inner = {r["candidate"]: r for r in csv.DictReader(file) if r["fold"] == "all_for_refit"}
    x = matrices["physical", "pre18_2", "bipolar6"]
    usable = np.isfinite(x).all(axis=1) & (labels >= 0)
    entries = {}
    for family in ("logistic", "rbf"):
        options = [Candidate("physical", "pre18_2", "bipolar6", family, param) for param in PARAMETERS[family]]
        def ranking(c):
            row = inner[c.name]
            return round(float(row["coverage_adjusted_macro_auc"]), 14), int(row["scored_pairs"])
        c = max(options, key=ranking)
        trained = fit(x, labels, groups, c)
        if trained is None or not np.isfinite(predict(trained, x[usable])).all():
            raise ValueError("Invalid refitted candidate")
        cohort = sorted(np.unique(groups[usable]).tolist())
        entries[family] = {"selected_config": c.name, "fitted_parameters": trained,
                           "inner_selection": {option.name: inner[option.name] for option in options},
                           "validation_reference": {"relative_file": "report.json",
                                                    "scope": f"pipeline/physical/pre18_2/bipolar6/{family}"},
                           "participants_in_fit": len(cohort), "participant_ids": cohort,
                           "probes_in_fit": int(usable.sum()), "positive": int((labels[usable] == 1).sum()),
                           "negative": int((labels[usable] == 0).sum())}
    scores = {family: np.full(len(keys), np.nan) for family in entries}
    scope_to_family = {model["validation_reference"]["scope"]: family for family, model in entries.items()}
    positions = {key: idx for idx, key in enumerate(keys)}
    with (directory / "outer_predictions.csv").open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            family = scope_to_family.get(row["evaluation_scope"])
            if family and row["score"]:
                scores[family][positions[row["participant"], row["probe_id"]]] = float(row["score"])
    if any(not np.array_equal(np.isfinite(values), usable) for values in scores.values()):
        raise ValueError("Expected identical held-out probe support for the two prespecified model families")
    npeople = len(np.unique(groups))
    draws = np.random.default_rng(SEED).multinomial(npeople, np.full(npeople, 1 / npeople),
                                                   size=report["bootstrap_replicates"])
    summaries = {family: describe(labels, values, groups, draws) for family, values in scores.items()}
    paired_oof = paired(labels, scores["rbf"], scores["logistic"], groups, draws, summaries["rbf"], summaries["logistic"])
    payload = {
        "spec": "offline-attlapse-two-channel-shadow-comparison-v1", "source_doi": "10.5281/zenodo.23592484",
        "source_only_research": True, "source_healthy_controls_seen": len(np.unique(groups)),
        "feature_file_sha256": report["feature_file_sha256"], "input_rate_hz": rate,
        "rate_context": "Original 512 Hz" if rate == 512 else "Causally antialiased from original 512 Hz",
        "electrodes": ["Fp1", "Fp2"], "units": "microvolts", "derivation": "Fp1 - Fp2",
        "feature_names": list(FEATURE_SETS["bipolar6"]),
        "feature_definition": "Welch PSD. Relative theta(4-8), alpha(8-13), beta(13-30) vs total 0.5-45 Hz. "
                              "Natural-log ratios beta/theta, alpha/beta, theta/alpha. Mean over accepted windows.",
        "quality": {"policy": "physical", "reject": "Source flagged bad electrodes, invalid phase or insufficient coverage; "
                             "both raw channels and bipolar derivation must pass non-EOG/EMG artifact checks. "
                             "EOG/EMG spectral proxy flags do not reject a window.", "min_usable_windows": 3,
                    "min_coverage_seconds": 6, "min_usable_window_fraction": .6},
        "lookback_sec": [-18, -2], "window_sec": 4, "hop_sec": 2, "filter_context_sec": 6,
        "window_grid": "recording start; only fully included analysis windows in the lookback",
        "standardization": "Weighted training mean and scale, stored per model.",
        "labels": {"0": "on_task", "1": "spontaneous_or_deliberate_mind_wandering"},
        "score": "sigmoid of the model margin, higher indicates mind wandering, uncalibrated; exploratory threshold 0.5.",
        "model_definitions": {family: report["models"][family] for family in entries},
        "refit": "Within each fixed family, use the all-participant fourfold inner OOF coverage-adjusted "
                 "criterion (tie: scored pairs then declared parameter order); fit on all eligible records.",
        "research_scope": "Exploratory same-feature comparison after inspecting benchmark results. "
                          "Use untouched new participants for confirmation. Existing collector is unchanged.",
        "reproduce": "attlapse_shadow_refit.py --features <feature JSONL> --evaluation <nested evaluation directory>",
        "models": entries, "heldout_rbf_minus_logistic": paired_oof}
    with tempfile.NamedTemporaryFile(mode="w", dir=directory, encoding="utf-8", delete=False) as tmp:
        json.dump(payload, tmp, indent=2, allow_nan=False)
    Path(tmp.name).replace(directory / "shadow_comparison_candidates.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--evaluation", required=True, type=Path)
    args = parser.parse_args()
    refit(args.features, args.evaluation)
