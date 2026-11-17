"""Participant-held-out evaluation of causal, raw two-electrode probe features."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import tempfile

# The small fold-specific linear systems are faster without threaded BLAS.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ningsi/src"))
from ningsi.models.human import average_precision, sigmoid


SEED = 20261113
QUALITIES = ("strict", "bipolar_strict", "physical")
WINDOWS = ("pre12_2", "pre18_2", "pre32_2")
FEATURES = ("bipolar6", "bilateral15", "bilateral_variability18")
PARAMETERS = {"logistic": (.01, .1), "lda": (.3, .8), "rbf": (.1, 1.)}


@dataclass(frozen=True)
class Candidate:
    quality: str
    window: str
    features: str
    family: str
    parameter: float

    @property
    def name(self):
        return f"{self.quality}/{self.window}/{self.features}/{self.family}/{self.parameter:g}"

    @property
    def matrix_key(self):
        return self.quality, self.window, self.features


BASELINE = Candidate("strict", "pre12_2", "bipolar6", "logistic", .01)


def candidates():
    # This order also resolves exact inner-CV ties, independent of outer results.
    return [Candidate(q, w, f, family, param) for q, w, f in itertools.product(QUALITIES, WINDOWS, FEATURES)
            for family, params in PARAMETERS.items() for param in params]


def load_features(path, feature_sets):
    grouped, labels = {}, {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        row = json.loads(line)
        key = row["participant"], row["probe_id"]
        label = row["label"]
        if label not in (0, 1, None) or (key in labels and labels[key] != label):
            raise ValueError(f"Invalid or inconsistent label for {key}")
        labels[key] = label
        cell = grouped.setdefault((row["quality"], row["window"]), {})
        if key in cell:
            raise ValueError(f"Duplicate probe {key}")
        cell[key] = row
    keys = sorted(labels)
    expected = set(itertools.product(QUALITIES, WINDOWS))
    if set(grouped) != expected or any(set(cell) != set(keys) for cell in grouped.values()):
        raise ValueError("Each QC/window must account for every original probe, including abstentions")
    groups = np.array([key[0] for key in keys])
    y = np.array([labels[key] if labels[key] is not None else -1 for key in keys], dtype=int)
    matrices = {}
    for (q, w), cell in grouped.items():
        for f in FEATURES:
            names = feature_sets[f]
            x = np.full((len(keys), len(names)), np.nan)
            for i, key in enumerate(keys):
                row = cell[key]
                if not row["reasons"]:
                    x[i] = [row["features"][name] for name in names]
                    if not np.isfinite(x[i]).all():
                        raise ValueError(f"Nonfinite features passed quality gate at {q}/{w}/{f}/{key}")
            matrices[q, w, f] = x
    return keys, groups, y, matrices


def folds(people, n, seed):
    people = np.unique(people)
    if len(people) < n:
        raise ValueError(f"Need {n} distinct participants")
    ordered = np.random.default_rng(seed).permutation(people)
    return [sorted(ordered[i::n].tolist()) for i in range(n)]


def training_weights(y, groups):
    _, ids, counts = np.unique(groups, return_inverse=True, return_counts=True)
    w = 1. / counts[ids]
    w /= np.bincount(y, weights=w, minlength=2)[y]
    return w / w.sum()


def rbf(x, train):
    distance = (x*x).sum(axis=1)[:, None] + (train*train).sum(axis=1) - 2 * (x @ train.T)
    return np.exp(-np.maximum(distance, 0.) / train.shape[1])


def fit(x, y, groups, c):
    usable = np.isfinite(x).all(axis=1) & (y >= 0)
    x, y, groups = x[usable], y[usable], groups[usable]
    if np.unique(y).size < 2:
        return None
    w = training_weights(y, groups)
    mean = w @ x
    scale = np.sqrt(w @ np.square(x - mean))
    scale[scale < 1e-9] = 1.
    z = (x - mean) / scale
    state = {"family": c.family, "parameter": c.parameter, "mean": mean.tolist(), "scale": scale.tolist()}
    if c.family == "logistic":
        # Minimize weighted mean logistic loss + lambda * ||beta||² / 2.
        # The intercept is unpenalized. Newton's method solves the same convex objective as v1.
        design = np.column_stack((np.ones(len(y)), z))
        penalty = np.diag([0., *([c.parameter] * z.shape[1])])
        beta = np.zeros(design.shape[1])
        for _ in range(50):
            prob = sigmoid(design @ beta)
            grad = design.T @ (w * (prob - y)) + penalty @ beta
            hess = (design.T * (w * prob * (1 - prob))) @ design + penalty
            step = np.linalg.solve(hess, grad)
            beta -= step
            if np.max(np.abs(step)) < 1e-9:
                break
        state.update(bias=float(beta[0]), weights=beta[1:].tolist())
    elif c.family == "lda":
        centroids = np.stack([np.average(z[y == k], weights=w[y == k], axis=0) for k in (0, 1)])
        centered = z - centroids[y]
        scatter = (centered.T * w) @ centered
        target = max(float(np.trace(scatter)) / scatter.shape[0], 1e-8)
        covariance = (1 - c.parameter) * scatter + c.parameter * target * np.eye(scatter.shape[0])
        beta = np.linalg.solve(covariance, centroids[1] - centroids[0])
        state.update(bias=float(-.5 * centroids.sum(axis=0) @ beta), weights=beta.tolist())
    elif c.family == "rbf":
        # Weighted mean squared error to {-1,+1} + lambda * ||f||_H²; unpenalized intercept.
        target = 2 * y - 1
        k = rbf(z, z)
        root_w = np.sqrt(w)
        column_mean = w @ k
        center = float(column_mean @ w)
        gram = k - column_mean[:, None] - column_mean[None, :] + center
        rhs = root_w * (target - w @ target)
        alpha = root_w * np.linalg.solve(root_w[:, None] * gram * root_w + c.parameter * np.eye(len(w)), rhs)
        # Centering for a new kernel vector can be absorbed into these coefficients and intercept.
        beta = alpha - w * alpha.sum()
        state.update(train=z.tolist(), dual=beta.tolist(), bias=float(w @ target - column_mean @ beta))
    else:
        raise ValueError(c.family)
    return state


def predict(state, x):
    result = np.full(len(x), np.nan)
    usable = np.isfinite(x).all(axis=1)
    if state is None or not usable.any():
        return result
    z = (x[usable] - state["mean"]) / state["scale"]
    if state["family"] == "rbf":
        margin = rbf(z, np.asarray(state["train"])) @ state["dual"]
    else:
        margin = z @ state["weights"]
    result[usable] = sigmoid(margin + state["bias"])
    return result


def person_stats(y, score, groups):
    people = np.unique(groups)
    stats = []
    for person in people:
        mask = groups == person
        ys, ps = y[mask], score[mask]
        positive, negative = ps[ys == 1], ps[ys == 0]
        total = len(positive) * len(negative)
        a, b = positive[np.isfinite(positive)], negative[np.isfinite(negative)]
        comparisons = a[:, None] - b
        n = comparisons.size
        wins = (comparisons > 0).sum() + .5 * (comparisons == 0).sum()
        stats.append({"participant": str(person), "positive": len(a), "negative": len(b), "pairs": n,
                      "possible_pairs": total, "concordant_pairs": float(wins),
                      "auc": float(wins / n) if n else None,
                      "coverage_adjusted_auc": float((wins + .5 * (total - n)) / total) if total else None})
    return stats


def mean(values):
    valid = [v for v in values if v is not None]
    return float(np.mean(valid)) if valid else None


def selection_score(y, score, groups):
    stats = person_stats(y, score, groups)
    value = mean(p["coverage_adjusted_auc"] for p in stats)
    return (value if value is not None else .5, sum(p["pairs"] for p in stats))


def pipeline_scopes(c):
    return (f"selected/{c.quality}", f"family/{c.quality}/{c.family}",
            f"pipeline/{c.quality}/{c.window}/{c.features}/{c.family}")


def select(train, groups, y, matrices, configs, seed):
    selections, choices, inner_log = {}, {}, []
    inner = folds(groups[train], 4, seed)
    splits = [(train[~np.isin(groups[train], hold)], np.isin(groups[train], hold)) for hold in inner]
    for c in configs:
        x = matrices[c.matrix_key]
        scores = np.full(len(train), np.nan)
        for dev, val in splits:
            scores[val] = predict(fit(x[dev], y[dev], groups[dev], c), x[train[val]])
        value, pairs = selection_score(y[train], scores, groups[train])
        # Coverage-adjusted score is compared first; coverage only breaks a numerical tie.
        ranking = (round(value, 14), pairs)
        inner_log.append({"candidate": c.name, "coverage_adjusted_macro_auc": value, "scored_pairs": pairs,
                          "scored_labelled_probes": int((np.isfinite(scores) & (y[train] >= 0)).sum())})
        for scope in pipeline_scopes(c):
            if scope not in choices or ranking > choices[scope]:
                choices[scope], selections[scope] = ranking, c
    return selections, inner_log, inner


def ratio(num, den):
    return np.divide(num, den, out=np.full(np.shape(num), np.nan, dtype=float), where=np.asarray(den) > 0)


def ci(values):
    finite = values[np.isfinite(values)]
    if not len(finite):
        return None
    return np.percentile(finite, [2.5, 97.5]).tolist()


def describe(y, score, groups, counts):
    """Participant cluster resampling; all probes of a drawn participant have the same multiplicity."""
    people, gids = np.unique(groups, return_inverse=True)
    stats = person_stats(y, score, groups)
    within = np.array([p["auc"] if p["auc"] is not None else np.nan for p in stats])
    adj = np.array([p["coverage_adjusted_auc"] if p["possible_pairs"] else np.nan for p in stats])
    valid = np.isfinite(score) & (y >= 0)
    pred = score >= .5
    pos, neg = valid & (y == 1), valid & (y == 0)
    def by_group(mask):
        return np.bincount(gids[mask], minlength=len(people)).astype(float)
    positives, negatives = by_group(pos), by_group(neg)
    tp, tn = by_group(pos & pred), by_group(neg & ~pred)
    # U[i,j] is the pairwise AUC numerator for positive group i vs. negative group j.
    positive_scores, negative_scores = score[pos], score[neg]
    pairs = positive_scores[:, None] - negative_scores
    indices = gids[pos, None] * len(people) + gids[neg]
    wins = np.bincount(indices.ravel(), weights=((pairs > 0) + .5 * (pairs == 0)).ravel(),
                       minlength=len(people)**2).reshape(len(people), len(people))
    boots = {}
    for name, values in (("macro_auc", within), ("coverage_adjusted_macro_auc", adj)):
        ok = np.isfinite(values)
        boots[name] = ratio(counts[:, ok] @ values[ok], counts[:, ok].sum(axis=1))
    boot_pos, boot_neg = counts @ positives, counts @ negatives
    boots["pooled_auc"] = ratio(((counts @ wins) * counts).sum(axis=1), boot_pos * boot_neg)
    boots["balanced_accuracy"] = .5 * (ratio(counts @ tp, boot_pos) + ratio(counts @ tn, boot_neg))
    possible = sum(p["possible_pairs"] for p in stats)
    scored_pairs = sum(p["pairs"] for p in stats)
    def fraction(a, b):
        return float(a / b) if b else None
    npos, nneg = int(positives.sum()), int(negatives.sum())
    sens, spec = fraction(tp.sum(), npos), fraction(tn.sum(), nneg)
    summary = {"total_probes": len(y), "labelled_probes": int((y >= 0).sum()), "scored_labelled": int(valid.sum()),
               "probe_coverage": fraction(valid.sum(), (y >= 0).sum()), "positive": npos, "negative": nneg,
               "participants": len(people), "scored_participants": int(np.count_nonzero(positives + negatives)),
               "both_classes_scored_participants": int(np.isfinite(within).sum()),
               "both_classes_original_participants": int(np.isfinite(adj).sum()),
               "scored_within_person_pairs": scored_pairs, "possible_within_person_pairs": possible,
               "within_person_pair_coverage": fraction(scored_pairs, possible),
               "macro_auc": mean(p["auc"] for p in stats),
               "coverage_adjusted_macro_auc": mean(p["coverage_adjusted_auc"] for p in stats),
               "pooled_auc": fraction(wins.sum(), npos * nneg),
               "balanced_accuracy": (sens + spec) / 2 if npos and nneg else None,
               "sensitivity": sens, "specificity": spec, "average_precision": average_precision(y[valid], score[valid]),
               "confusion": {"tp": int(tp.sum()), "tn": int(tn.sum()), "fp": int(nneg - tn.sum()),
                             "fn": int(npos - tp.sum())}, "ci95": {}}
    for name, vals in boots.items():
        min_people = summary["both_classes_scored_participants"] if name == "macro_auc" else summary["scored_participants"]
        if name == "coverage_adjusted_macro_auc":
            min_people = summary["both_classes_original_participants"]
        summary["ci95"][name] = ci(vals) if min_people >= 2 else None
    return summary, boots


def paired(y, score, baseline, groups, counts, new_stats, ref_stats):
    overlap = np.isfinite(score) & np.isfinite(baseline)
    new_common, new_boot = describe(y, np.where(overlap, score, np.nan), groups, counts)
    old_common, old_boot = describe(y, np.where(overlap, baseline, np.nan), groups, counts)
    result = {"same_probes": new_common["scored_labelled"], "same_participants": new_common["scored_participants"],
              "same_people_with_both_labels": new_common["both_classes_scored_participants"],
              "common": {}, "native_coverage": {}}
    for name in ("macro_auc", "pooled_auc", "balanced_accuracy"):
        a, b = new_common[name], old_common[name]
        result["common"][name] = {"candidate": a, "baseline": b, "difference": a - b if a is not None and b is not None else None,
                                   "paired_cluster_ci95": ci(new_boot[name] - old_boot[name])
                                   if new_common["ci95"][name] is not None else None}
    a, a_boot = new_stats
    b, b_boot = ref_stats
    name = "coverage_adjusted_macro_auc"
    result["native_coverage"] = {"adjusted_auc_delta": a[name] - b[name] if a[name] is not None and b[name] is not None else None,
                                 "paired_cluster_ci95": ci(a_boot[name] - b_boot[name]) if a["ci95"][name] else None,
                                 "scored_labelled_delta": a["scored_labelled"] - b["scored_labelled"]}
    return result


def dump_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def dump_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def compare_historical(directory, historical, bootstraps=2000):
    """Compare OOF with OOF only: v1 trained on ocular-corrected inputs, tested on original raw EEG."""
    wanted = {f"selected/{quality}" for quality in QUALITIES} | {"fixed/" + BASELINE.name}
    predictions = {scope: {} for scope in wanted}
    with (directory / "outer_predictions.csv").open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            scope = row["evaluation_scope"]
            if scope in wanted:
                key = row["participant"], row["probe_id"]
                predictions[scope][key] = (int(row["label"]) if row["label"] else -1,
                                           float(row["score"]) if row["score"] else np.nan)
    with historical.open(encoding="utf-8-sig", newline="") as file:
        reference = {}
        for row in csv.DictReader(file):
            key = row["participant"], row["probe_id"]
            if key in reference:
                raise ValueError(f"Duplicate historical probe {key}")
            if row["heldout_score_without_eog_correction"] and row["raw_probe_qc"] != "usable":
                raise ValueError(f"Historical score failed raw QC: {key}")
            reference[key] = (int(row["label"]) if row["label"] else -1,
                              float(row["heldout_score_without_eog_correction"])
                              if row["heldout_score_without_eog_correction"] else np.nan)
    keys = sorted(reference)
    groups = np.array([key[0] for key in keys])
    y = np.array([reference[key][0] for key in keys])
    baseline = np.array([reference[key][1] for key in keys])
    people = np.unique(groups)
    draws = np.random.default_rng(SEED).multinomial(len(people), np.full(len(people), 1 / len(people)), size=bootstraps)
    ref_stats = describe(y, baseline, groups, draws)
    results = {}
    for scope, rows in predictions.items():
        if set(rows) != set(keys) or any(rows[k][0] != reference[k][0] for k in keys):
            raise ValueError(f"Historical/current identities or original probe labels differ at {scope}")
        score = np.array([rows[k][1] for k in keys])
        stats = describe(y, score, groups, draws)
        comparison = paired(y, score, baseline, groups, draws, stats, ref_stats)
        # Different training distributions, preprocessing and fold assignments remain after matching.
        results[scope] = {"same_probes": comparison["same_probes"], "same_participants": comparison["same_participants"],
                          "same_people_with_both_labels": comparison["same_people_with_both_labels"],
                          "common": comparison["common"]}
    record = {"description": "Historical raw-input held-out scores from the earlier ocular-corrected training procedure. "
                             "Same probe IDs and labels; different training inputs and outer subject assignments. "
                             "Each reported score excludes its participant at fitting. This is not an independent cohort.",
              "source": str(historical), "sha256": hashlib.sha256(historical.read_bytes()).hexdigest(),
              "baseline_native_metrics": ref_stats[0], "same_probe_comparisons": results}
    dest = directory / "historical_raw_common.json"
    with tempfile.NamedTemporaryFile(mode="w", dir=directory, encoding="utf-8", delete=False) as tmp:
        json.dump(record, tmp, indent=2, allow_nan=False)
    Path(tmp.name).replace(dest)
    return record


def run(source, destination, feature_sets, bootstraps=2000):
    keys, groups, y, matrices = load_features(source, feature_sets)
    if len(np.unique(groups)) != 28 or len(keys) != 420:
        raise ValueError("The registered experiment requires all 28 controls and all 420 probes")
    if destination.exists():
        raise FileExistsError(destination)
    configs = candidates()
    outer = folds(groups, 5, SEED)
    results, chosen, splits, all_inner = {}, {}, [], []
    fold_ids = np.zeros(len(y), dtype=int)
    for i, held in enumerate(outer, 1):
        test = np.flatnonzero(np.isin(groups, held))
        train = np.flatnonzero(~np.isin(groups, held))
        fold_ids[test] = i
        selected, inner, assignments = select(train, groups, y, matrices, configs, SEED + i)
        splits.append({"outer_fold": i, "train": sorted(set(groups[train].tolist())), "test": held,
                       "inner_validation_people": assignments})
        all_inner.extend({"fold": i, **rec} for rec in inner)
        test_predictions = {c: predict(fit(matrices[c.matrix_key][train], y[train], groups[train], c),
                                       matrices[c.matrix_key][test]) for c in configs}
        selected.update({f"fixed/{c.name}": c for c in configs})
        for scope, c in selected.items():
            if scope not in results:
                results[scope] = np.full(len(keys), np.nan)
                chosen[scope] = {}
            results[scope][test] = test_predictions[c]
            chosen[scope][i] = c.name
        print(f"Outer fold {i}/5 complete", flush=True)
    final, final_inner, final_splits = select(np.arange(len(keys)), groups, y, matrices, configs, SEED)
    all_inner.extend({"fold": "all_for_refit", **rec} for rec in final_inner)
    people = np.unique(groups)
    draws = np.random.default_rng(SEED).multinomial(len(people), np.full(len(people), 1 / len(people)), size=bootstraps)
    baseline_id = "fixed/" + BASELINE.name
    baseline = results[baseline_id]
    ref_stats = describe(y, baseline, groups, draws)
    summary = {}
    # Multiple nested scopes can make identical choices. Bootstrap identical vectors once.
    memo = {}
    for scope, score in results.items():
        key = score.tobytes()
        if key not in memo:
            stats = describe(y, score, groups, draws)
            memo[key] = {"metrics": stats[0],
                         "paired_vs_raw_baseline": paired(y, score, baseline, groups, draws, stats, ref_stats)}
        summary[scope] = {**memo[key], "selected_by_outer_fold": chosen[scope]}
    frozen = {}
    for quality in QUALITIES:
        scope = "selected/" + quality
        c = final[scope]
        x = matrices[c.matrix_key]
        frozen[quality] = {"configuration": c.name, "feature_names": list(feature_sets[c.features]),
                           "quality": c.quality, "window": c.window, "features": c.features,
                           "state": fit(x, y, groups, c), "training_rows": int((np.isfinite(x).all(axis=1) & (y >= 0)).sum()),
                           "research_only": True}
    report = {"data": str(source), "feature_file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "seed": SEED, "bootstrap_replicates": bootstraps, "baseline": baseline_id,
              "interpretation": "Participant-held-out nested CV in one public healthy-control cohort. No target-device validation.",
              "selection": "Within each quality policy, pool inner OOF validation predictions across four fixed subject folds. "
                           "For each original dual-label participant, score unobserved positive-negative pairs at 0.5, "
                           "then equally average participants. Break ties by the number of scored pairs then declared candidate order.",
              "abstention": "Ineligible features or no two-class training set yields no score. Predictions use threshold 0.5; uncalibrated.",
              "uncertainty": "Percentile 95% intervals resample all people as clusters from frozen OOF scores, without model refits.",
              "pooled_comparison": "All common-probe metrics use the exact scored probe intersection. Native coverage effects are also reported.",
              "models": {"logistic": "Weighted normalized logistic loss + lambda/2 times squared standardized weights.",
                         "lda": "Equal-prior weighted LDA; shrink within-class covariance toward a sphere of equal average variance.",
                         "rbf": "Gaussian kernel exp(-squared distance/p); weighted mean squared error to {-1,+1} plus lambda times squared RKHS norm; free bias; sigmoid margin."},
              "parameter_grid": PARAMETERS, "results": summary}
    def rows():
        for scope, scores in results.items():
            for idx, (person, probe) in enumerate(keys):
                value = scores[idx]
                yield {"participant": person, "probe_id": probe, "label": int(y[idx]) if y[idx] >= 0 else "",
                       "outer_fold": fold_ids[idx], "evaluation_scope": scope, "chosen": chosen[scope][int(fold_ids[idx])],
                       "score": float(value) if np.isfinite(value) else "",
                       "prediction": int(value >= .5) if np.isfinite(value) else ""}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".nested-eval-") as temp:
        stage = Path(temp) / "result"
        stage.mkdir()
        dump_json(stage / "report.json", report)
        dump_json(stage / "splits.json", {"outer": splits, "all_for_refit_inner_validation_people": final_splits})
        dump_json(stage / "deployment_candidates.json", frozen)
        dump_csv(stage / "outer_predictions.csv",
                 ("participant", "probe_id", "label", "outer_fold", "evaluation_scope", "chosen", "score", "prediction"), rows())
        dump_csv(stage / "inner_selection.csv", ("fold", "candidate", "coverage_adjusted_macro_auc",
                                                 "scored_pairs", "scored_labelled_probes"), all_inner)
        stage.rename(destination)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--historical", type=Path, help="Also compare published v1 raw-input held-out predictions")
    parser.add_argument("--compare-existing", action="store_true", help="Pair historical scores with existing OOF outputs")
    args = parser.parse_args()
    if not args.compare_existing:
        if not args.features:
            parser.error("--features is required for evaluation")
        from attlapse_two_channel_features import FEATURE_SETS
        run(args.features, args.output, FEATURE_SETS)
    elif not args.historical:
        parser.error("--compare-existing requires --historical")
    if args.historical:
        compare_historical(args.output, args.historical)
