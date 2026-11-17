"""Held-out artifact-flag-only and time-on-task diagnostics for the two-channel SART benchmark."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import tempfile

import attlapse_two_channel_eval as evaluation
from attlapse_two_channel_features import FEATURE_SETS
import numpy as np


FLAGS = ("input_eog", "input_emg")
WINDOW = "pre18_2"
CONFIG = evaluation.Candidate("physical", WINDOW, "diagnostic", "logistic", 0.1)
RBF_SCOPE = "pipeline/physical/pre18_2/bipolar6/rbf"


def inputs(path):
    rows = {"strict": {}, "physical": {}}
    for line in path.read_text(encoding="utf8").splitlines():
        rec = json.loads(line)
        if rec["window"] == WINDOW and rec["quality"] in rows:
            key = (rec["participant"], rec["probe_id"])
            records = rows[rec["quality"]]
            if key in records:
                raise ValueError(f"Duplicate QC record: {key}")
            records[key] = rec
    keys = sorted(rows["strict"])
    if len(keys) != 420 or set(keys) != set(rows["physical"]):
        raise ValueError("Need all 420 strict/physical row pairs")
    x = np.full((len(keys), len(FLAGS)), np.nan)
    spectra = np.full((len(keys), len(FEATURE_SETS["bipolar6"])), np.nan)
    y = np.empty(len(keys), dtype=int)
    rate = set()
    for i, key in enumerate(keys):
        strict, physical = rows["strict"][key], rows["physical"][key]
        for field in ("label", "rate", "onset_sec", "windows_total"):
            if strict[field] != physical[field]:
                raise ValueError(f"Disagreement in {field}: {key}")
        y[i] = strict["label"]
        rate.add(physical["rate"])
        if physical["reasons"]:
            continue
        n = strict["windows_total"]
        counts = [strict["window_rejections"].get(flag, 0) for flag in FLAGS]
        if n <= 0 or any(not 0 <= count <= n for count in counts):
            raise ValueError(f"Invalid window-rejection counts: {key}")
        x[i] = np.asarray(counts, dtype=float) / n
        spectra[i] = [physical["features"][f] for f in FEATURE_SETS["bipolar6"]]
        if not np.isfinite(spectra[i]).all():
            raise ValueError(f"Nonfinite spectrum on accepted probe: {key}")
    if len(rate) != 1 or not np.isin(y, [0, 1]).all():
        raise ValueError("Unexpected rate or missing/non-binary label")
    return keys, np.asarray([k[0] for k in keys]), y, x, spectra, rate.pop()


def summarize(y, score, groups, boot):
    stats, _ = evaluation.describe(y, score, groups, boot)
    fields = ("total_probes", "scored_labelled", "probe_coverage", "participants", "scored_participants",
              "both_classes_scored_participants", "positive", "negative", "scored_within_person_pairs",
              "possible_within_person_pairs", "macro_auc", "coverage_adjusted_macro_auc")
    result = {k: stats[k] for k in fields}
    result["participant_bootstrap_ci95"] = {k: stats["ci95"][k] for k in ("macro_auc", "coverage_adjusted_macro_auc")}
    return result


def cross_fit(x, groups, y, splits):
    visits = np.zeros(len(y), dtype=int)
    oof = np.full(len(y), np.nan)
    fold_log = []
    for split in splits:
        hold = np.isin(groups, split["test"])
        train = np.isin(groups, split["train"])
        if np.any(hold == train) or not np.any(hold):
            raise ValueError("The frozen fold is not a valid disjoint partition")
        model = evaluation.fit(x[train], y[train], groups[train], CONFIG)
        if model is None:
            raise ValueError("No two-class eligible training data in an outer fold")
        oof[hold] = evaluation.predict(model, x[hold])
        visits[hold] += 1
        fold_log.append({"outer_fold": split["outer_fold"], "held_out_people": split["test"],
                         "training_eligible_probes": int(np.isfinite(x[train]).all(axis=1).sum()),
                         "held_out_eligible_probes": int(np.isfinite(oof[hold]).sum())})
    eligible = np.isfinite(x).all(axis=1)
    if not np.all(visits == 1) or not np.array_equal(np.isfinite(oof), eligible):
        raise ValueError("Incomplete or duplicate out-of-fold predictions")
    return oof, fold_log


def frozen_rbf(path, keys, y, splits):
    assigned = {p: s["outer_fold"] for s in splits for p in s["test"]}
    rows = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for r in csv.DictReader(stream):
            if r["evaluation_scope"] != RBF_SCOPE:
                continue
            key = r["participant"], r["probe_id"]
            if key in rows or assigned[key[0]] != int(r["outer_fold"]):
                raise ValueError(f"Unexpected frozen RBF record: {key}")
            rows[key] = (int(r["label"]), float(r["score"]) if r["score"] else np.nan)
    if sorted(rows) != keys or not np.array_equal(y, [rows[k][0] for k in keys]):
        raise ValueError("The main RBF predictions and diagnostic inputs do not match")
    return np.asarray([rows[k][1] for k in keys])


def paired_diff(y, groups, candidate, reference, bootstrap):
    if not np.array_equal(np.isfinite(candidate), np.isfinite(reference)):
        raise ValueError("A paired comparison requires the same predeclared eligible probes")
    new, new_boot = evaluation.describe(y, candidate, groups, bootstrap)
    base, base_boot = evaluation.describe(y, reference, groups, bootstrap)
    return {k: {"difference": new[k] - base[k], "ci95": evaluation.ci(new_boot[k] - base_boot[k])}
            for k in ("macro_auc", "coverage_adjusted_macro_auc")}


def assess(path, splits, expected_rate, predictions):
    keys, groups, y, x, spectra, rate = inputs(path)
    if rate != expected_rate:
        raise ValueError(f"Expected {expected_rate}, got {rate} at {path}")
    subjects = np.unique(groups)
    if len(subjects) != 28 or len(splits) != 5:
        raise ValueError("Incorrect outer group definition")
    oof, fold_log = cross_fit(x, groups, y, splits)
    eligible = np.isfinite(x).all(axis=1)
    ordinal = np.array([int(probe.removeprefix("probe-")) for _, probe in keys], dtype=float)
    if not np.array_equal(np.unique(ordinal), np.arange(1, 16)):
        raise ValueError("Unexpected probe numbering")
    # This comparison prespecifies later ordinal = larger score. It never uses answers.
    time_score = np.where(eligible, ordinal / 15., np.nan)
    time_column = np.where(eligible, ordinal, np.nan)[:, None]
    time_lr, _ = cross_fit(time_column, groups, y, splits)
    spectra_lr, _ = cross_fit(spectra, groups, y, splits)
    combined_lr, _ = cross_fit(np.column_stack([time_column, spectra]), groups, y, splits)
    rbf = frozen_rbf(predictions, keys, y, splits)
    boot = np.random.default_rng(evaluation.SEED).multinomial(len(subjects), np.full(len(subjects), 1/len(subjects)), size=2000)
    return {
        "rate_hz": rate, "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "flag_only_logistic": summarize(y, oof, groups, boot),
        "later_probe_higher": summarize(y, time_score, groups, boot),
        "time_only_logistic": summarize(y, time_lr, groups, boot),
        "bipolar6_only_logistic": summarize(y, spectra_lr, groups, boot),
        "time_and_bipolar6_logistic": summarize(y, combined_lr, groups, boot),
        "time_plus_bipolar6_minus_time_only": paired_diff(y, groups, combined_lr, time_lr, boot),
        "frozen_rbf_minus_untrained_probe_order": paired_diff(y, groups, rbf, time_score, boot),
        "main_predictions_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
        "folds": fold_log,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-features", type=Path, required=True)
    parser.add_argument("--resampled-features", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--native-predictions", type=Path, required=True)
    parser.add_argument("--resampled-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    outer = json.loads(args.splits.read_text(encoding="utf8"))["outer"]
    result = {
        "scope": "Post-hoc diagnostics only; model and processing selection are unchanged.",
        "eligible": "All labeled, physically accepted probes using [-18,-2] s. All 420 probes retained in the adjusted metric denominator.",
        "artifact_features": [flag + " strict window-flag count divided by all candidate windows" for flag in FLAGS],
        "artifact_measurement": "Spectral quality flags inferred from the two raw frontal EEG channels. No ocular or muscle sensors used.",
        "artifact_classifier": "Fixed L2 logistic lambda=0.1, subject and class balanced weights and weighted standardization on outer training people alone.",
        "time_on_task": "Probe ordinal 1..15 increases with elapsed task progression. No fitting, parameters, or data-derived direction.",
        "time_increment": "Three fixed lambda=.1 logit models use ordinal only, six raw bipolar spectral features only, or both. Other settings, validity mask and held-out people match. All scaling and coefficients are estimated on training people. Paired intervals use identical participant draws on frozen OOF scores.",
        "main_rbf_comparison_scope": RBF_SCOPE,
        "reference": "Within-person pair concordance = 0.5 for uninformative rankings, and chance credit for unscored pairs only in the adjusted metric. Percentile intervals resample participants, not folds or windows.",
        "outer_splits_sha256": hashlib.sha256(args.splits.read_bytes()).hexdigest(),
        "results": [assess(args.native_features, outer, 512, args.native_predictions),
                    assess(args.resampled_features, outer, 250, args.resampled_predictions)],
        "interpretation": "An association with these two proxies or probe order identifies a possible alternative explanation. This does not identify causation or prove that frontal EEG power is neural, or determine validity on local hardware.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as tmp:
        staging = Path(tmp) / "diagnostic.json"
        evaluation.dump_json(staging, result)
        staging.replace(args.output)
