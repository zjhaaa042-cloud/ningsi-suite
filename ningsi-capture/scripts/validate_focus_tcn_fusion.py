"""Compare causal TCN with behavior and static EEG on the same ATTLAPSE probes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import benchmark_focus_tcn as benchmark
import numpy as np

from attlapse_focus_optimization import behavior, estimate_residual, logit, residual_score
from attlapse_two_channel_eval import Candidate, describe, ci, fit, predict, sigmoid
from refit_focus_tcn import fit_members, load_frozen_inputs, sha256


MODEL = Candidate(benchmark.GATE, benchmark.WINDOW, "exploratory_fusion", "logistic", .1)


def summarize_windows(x, mask):
    observed = mask[:, :, None]
    count = np.maximum(mask.sum(axis=1, keepdims=True), 1)
    mean = np.where(observed, x, 0.).sum(axis=1) / count
    centered = np.where(observed, x - mean[:, None], 0.)
    std = np.sqrt((centered ** 2).sum(axis=1) / count)
    time = np.arange(x.shape[1]) * 2.
    time_mean = np.sum(mask * time, axis=1, keepdims=True) / count
    centered_time = np.where(mask, time - time_mean, 0.)
    variance = np.maximum((centered_time ** 2).sum(axis=1, keepdims=True), 1.)
    slope = np.sum(centered * centered_time[:, :, None], axis=1) / variance
    return np.concatenate((mean, std, slope), axis=1)


def fold_predictions(x, mask, y, people, inputs, features, train, test):
    train_scored = train & mask.any(axis=1)
    test_scored = mask[test].any(axis=1)
    base = fit(inputs[train], y[train], people[train], MODEL)
    behavior_scores = predict(base, inputs[test])
    if not test_scored.any():
        return behavior_scores, np.full((test.sum(), 2), np.nan)

    static = fit(features[train_scored], y[train_scored], people[train_scored], benchmark.LR)
    mean, scale, members = fit_members(x[train_scored], mask[train_scored], y[train_scored], people[train_scored])
    indexes = np.flatnonzero(test)[test_scored]
    standardized = (x[indexes] - mean) / scale
    tcn_scores = np.mean([member.predict_proba(standardized, mask[indexes]) for member in members], axis=0)
    eeg_scores = np.full((test.sum(), 2), np.nan)
    eeg_scores[test_scored, 0] = predict(static, features[indexes])
    eeg_scores[test_scored, 1] = tcn_scores
    return behavior_scores, eeg_scores


def evaluate(benchmark_dir, data_dir):
    evidence, arrays = load_frozen_inputs(benchmark_dir, data_dir)
    x, mask, y, people, probes = (arrays[k] for k in ("features", "mask", "label", "participants", "probes"))
    eligible = mask.any(axis=1)
    keys = list(zip(people, probes, strict=True))
    inputs, _ = behavior(data_dir.parent, keys)
    if not np.isfinite(inputs).all():
        raise ValueError("The current ATTLAPSE cohort should have behavior for every label")
    static = summarize_windows(x, mask)
    splits_path = data_dir / "evaluation250" / "splits.json"
    split_sha256 = sha256(splits_path)
    if split_sha256 != evidence["run"]["splits_sha256"]:
        raise ValueError("Evaluation splits differ from the frozen TCN benchmark")
    splits = json.loads(splits_path.read_text(encoding="utf-8"))["outer"]
    names = ("behavior", "static_only", "tcn_only", "behavior_plus_static", "behavior_plus_tcn")
    scores = {name: np.full(len(y), np.nan) for name in names}
    visit_count = np.zeros(len(y), dtype=int)
    fitted = []
    for split in splits:
        outer_train = np.isin(people, split["train"])
        outer_test = np.isin(people, split["test"])
        if np.any(outer_train == outer_test):
            raise ValueError("Outer split overlaps or does not cover every participant")
        visit_count += outer_test
        train_behavior = np.full(len(y), np.nan)
        train_eeg = np.full((len(y), 2), np.nan)
        inner_count = np.zeros(len(y), dtype=int)
        for held_out in split["inner_validation_people"]:
            inner_test = np.isin(people, held_out)
            if np.any(inner_test & outer_test):
                raise ValueError("Outer test person leaked into fusion training")
            inner_count += inner_test
            train_behavior[inner_test], train_eeg[inner_test] = fold_predictions(
                x, mask, y, people, inputs, static, outer_train & ~inner_test, inner_test)
        if not np.array_equal(inner_count, outer_train) or not np.array_equal(
                np.isfinite(train_eeg).all(axis=1), outer_train & eligible):
            raise AssertionError("Inner fusion scores must cover each training participant exactly once")

        out_behavior, out_eeg = fold_predictions(x, mask, y, people, inputs, static, outer_train, outer_test)
        scores["behavior"][outer_test] = out_behavior
        usable = outer_train & eligible
        coefficients = {}
        for i, eeg_kind in enumerate(("static", "tcn")):
            train_score = logit(train_eeg[usable, i])[:, None]
            residual = estimate_residual(train_score, logit(train_behavior[usable]), y[usable], people[usable])
            if residual is None:
                raise ValueError("Inner training needs people with both valid probe labels")
            test_score = logit(out_eeg[:, i])[:, None]
            scores[f"{eeg_kind}_only"][outer_test] = out_eeg[:, i]
            scores[f"behavior_plus_{eeg_kind}"][outer_test] = np.where(
                eligible[outer_test], sigmoid(logit(out_behavior) + residual_score(test_score, residual)), out_behavior)
            coefficients[eeg_kind] = {key: float(val[0]) for key, val in zip(("center", "scale", "weight"), residual)}
        fitted.append({"outer_fold": split["outer_fold"], "fusion_coefficients": coefficients})
        print(f"Fold {split['outer_fold']}: {int((outer_test & eligible).sum())} matched probes", flush=True)

    if not np.all(visit_count == 1) or not np.isfinite(scores["behavior"]).all():
        raise AssertionError("Outer predictions must cover each probe once")
    for name in ("static", "tcn"):
        np.testing.assert_array_equal(np.isfinite(scores[f"{name}_only"]), eligible)
        np.testing.assert_array_equal(scores[f"behavior_plus_{name}"][~eligible], scores["behavior"][~eligible])
    subjects = len(np.unique(people))
    boot = np.random.default_rng(20601).multinomial(subjects, np.full(subjects, 1 / subjects), size=5000)
    comparisons = (("behavior_plus_tcn", "behavior"), ("behavior_plus_static", "behavior"),
                   ("behavior_plus_tcn", "behavior_plus_static"), ("tcn_only", "static_only"))
    results = {}
    for scope, valid in (("common_eeg", eligible), ("all_behavior_eeg_fallback", np.ones(len(y), dtype=bool))):
        measured = {name: describe(y, np.where(valid, score, np.nan), people, boot) for name, score in scores.items()
                    if scope == "common_eeg" or not name.endswith("_only")}
        differences = {}
        for a, b in comparisons:
            if a in measured and b in measured:
                aa, bb = measured[a], measured[b]
                differences[f"{a}_minus_{b}"] = {
                    "difference": aa[0]["macro_auc"] - bb[0]["macro_auc"],
                    "paired_ci95": ci(aa[1]["macro_auc"] - bb[1]["macro_auc"]),
                }
        results[scope] = {"scores": {name: data[0] for name, data in measured.items()}, "paired": differences}
    return {"scope": "exploratory nested grouped evaluation; previously used public cohort, no target device data",
            "data_doi": evidence["data_doi"], "input_sha256": evidence["input_sha256"],
            "splits_sha256": split_sha256, "seeds": evidence["tcn_seeds"],
            "fusion": "L2 within-participant rank residual of cross-fitted EEG logit on frozen behavior logit",
            "behavior": "probe progression and median reaction time, RT coefficient of variation, omission and commission rates in previous 18 completed trials",
            "static": "mean, standard deviation and time slope of the same QC-passed spectral windows",
            "outer_fits": fitted, "results": results}, arrays, eligible, scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=benchmark.OUTPUT)
    parser.add_argument("--data", type=Path, default=benchmark.DATA)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results, arrays, eligible, scores = evaluate(args.benchmark, args.data)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "fusion_metrics.json").open("x", encoding="utf-8") as stream:
        json.dump(results, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    with (args.output / "fusion_predictions.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["participant", "probe", "label", "eligible", *scores])
        for i, (person, probe, y) in enumerate(zip(arrays["participants"], arrays["probes"], arrays["label"], strict=True)):
            writer.writerow((person, probe, y, bool(eligible[i]),
                             *(score[i] if np.isfinite(score[i]) else "" for score in scores.values())))
    print(json.dumps({scope: result["paired"] for scope, result in results["results"].items()}, indent=2))


if __name__ == "__main__":
    main()
