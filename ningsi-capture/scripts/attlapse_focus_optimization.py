"""Compare pre-answer EEG residuals with a matched behavioral baseline on held-out participants."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from attlapse_artifact_diagnostic import CONFIG, inputs
from attlapse_data import PROBE_TRIALS
from attlapse_two_channel_eval import SEED, ci, describe, fit, person_stats, predict, sigmoid
from validate_attention_fusion import behavior_before_probe


def behavior(root, keys):
    standard, enhanced = [], []
    tables = {}
    for person, probe in keys:
        if person not in tables:
            path = root / "labels" / person / f"{person}_task-sart_beh.tsv"
            with path.open(encoding="utf-8-sig", newline="") as stream:
                tables[person] = {int(row["trial_number"]): row for row in csv.DictReader(stream, delimiter="\t")}
        trials = tables[person]
        ordinal = int(probe.split("-")[1])
        stop = PROBE_TRIALS[ordinal - 1]
        recent = np.asarray(behavior_before_probe(trials, stop))
        # Unlabelled completed responses, including the period preceding this probe only.
        previous = np.stack([behavior_before_probe(trials, stop - offset) for offset in (18, 36, 54)])
        usable = previous[np.isfinite(previous).all(axis=1)]
        habitual = np.median(usable, axis=0) if len(usable) else recent
        standard.append([ordinal, *recent])
        enhanced.append([ordinal, *recent, recent[0] - habitual[0], recent[1] - habitual[1],
                         recent[2] - habitual[2], recent[3] - habitual[3]])
    return np.asarray(standard), np.asarray(enhanced)


def extract_matrix(rows, sensor, key):
    return np.asarray([r[sensor].get(key) or [np.nan] * 6 for r in rows], dtype=float)


def personal_delta(x, groups, valid):
    """Baseline is made solely from valid earlier measurements. Zero denotes no prior observation."""
    output = x.copy()
    for person in np.unique(groups):
        earlier = []
        for i in np.where(groups == person)[0]:
            if np.isfinite(x[i]).all():
                output[i] = x[i] - np.median(earlier, axis=0) if earlier else 0.
                if valid[i]:
                    earlier.append(x[i])
    return output


def ranked_score(y, s, g):
    auc = [p["auc"] for p in person_stats(y, s, g) if p["auc"] is not None]
    return np.mean(auc) if auc else .5


def logit(p):
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return np.log(p / (1 - p))


def estimate_residual(x, offsets, y, group):
    good = np.isfinite(x).all(axis=1) & np.isfinite(offsets)
    x, offsets, y, group = x[good], offsets[good], y[good], group[good]
    if len(x) == 0:
        return None
    center = np.median(x, axis=0)
    scale = np.quantile(x, .75, axis=0) - np.quantile(x, .25, axis=0)
    scale[scale < 1e-8] = 1.
    z = np.clip((x - center) / scale, -4, 4)
    differences, offset_differences, weights = [], [], []
    for person in np.unique(group):
        pos = np.flatnonzero((group == person) & (y == 1))
        neg = np.flatnonzero((group == person) & (y == 0))
        if pos.size and neg.size:
            pi, ni = np.repeat(pos, len(neg)), np.tile(neg, len(pos))
            differences.append(z[pi] - z[ni])
            offset_differences.append(offsets[pi] - offsets[ni])
            weights.append(np.full(len(pi), 1 / len(pi)))
    if not differences:
        return None
    design = np.vstack(differences)
    t = np.concatenate(offset_differences)
    w = np.concatenate(weights)
    w /= w.sum()
    beta = np.zeros(design.shape[1])
    penalty = .1 * np.eye(len(beta))
    for _ in range(30):
        p = sigmoid(t + design @ beta)
        step = np.linalg.solve((design.T * (w * p * (1 - p))) @ design + penalty,
                               design.T @ (w * (p - 1)) + penalty @ beta)
        beta -= step
        if max(abs(step)) < 1e-8:
            break
    return center, scale, beta


def residual_score(x, params):
    result = np.zeros(len(x))
    good = np.isfinite(x).all(axis=1)
    if params is not None:
        center, scale, beta = params
        result[good] = np.clip((x[good] - center) / scale, -4, 4) @ beta
    return result


def predict_fold(train, test, designs, blocks, group, y):
    stage = {}
    for name, x in designs.items():
        state = fit(x[train], y[train], group[train], CONFIG)
        base_train = logit(predict(state, x[train]))
        base_test = logit(predict(state, x[test]))
        scores = {("none", 0): sigmoid(base_test)}
        for key, matrix in blocks.items():
            params = estimate_residual(matrix[train], base_train, y[train], group[train])
            extra = residual_score(matrix[test], params)
            for a in (.5, 1.):
                scores[key, a] = sigmoid(base_test + a * extra)
        stage[name] = scores
    return stage


def comparison(data, feature_path, out):
    root = data / "two-channel-benchmark"
    path = root / "features250.jsonl"
    keys, groups, y, _, spectra, rate = inputs(path)
    assert rate == 250 and len(keys) == 420
    sources = json.loads(feature_path.read_text(encoding="utf8"))
    rows = sorted(sources, key=lambda r: (r["participant"], r["probe"]))
    assert [(r["participant"], r["probe"]) for r in rows] == keys
    assert np.array_equal([r["label"] for r in rows], y)
    standard, enhanced = behavior(data, keys)
    valid_behavior = np.isfinite(standard).all(axis=1)
    valid = np.isfinite(spectra).all(axis=1) & valid_behavior
    if not np.all(np.isfinite(enhanced[valid_behavior])):
        raise ValueError("Behavior extension changes the comparison population")
    split_file = root / "evaluation250" / "splits.json"
    folds = json.loads(split_file.read_text(encoding="utf8"))["outer"]
    front = extract_matrix(rows, "front", "spec")
    blocks = {"original_spectra": spectra}
    for sensor in ("front", "eyes", "clean", "posterior"):
        x = extract_matrix(rows, sensor, "spec")
        e = extract_matrix(rows, sensor, "erp")
        relative = personal_delta(x, groups, valid)
        blocks[sensor + "_power"] = np.column_stack((x, relative))
        blocks[sensor + "_power_erp"] = np.column_stack((x, relative, e))
    # Preserve physical rejection on all routes; unavailable EEG yields zero residual.
    blocks = {k: np.where(valid[:, None], x, np.nan) for k, x in blocks.items()}
    front_options = [("none", 0)] + [(s, a) for s in ("original_spectra", "front_power", "front_power_erp")
                                     for a in (.5, 1.)]
    people = np.unique(groups)
    boot = np.random.default_rng(SEED).multinomial(len(people), np.full(len(people), 1 / len(people)), size=4000)
    records, all_scores = {}, {}
    for scope, scope_mask in (("common", valid), ("all_behavior", valid_behavior)):
        designs = {"standard": standard, "enhanced": enhanced}
        scores = {k: np.full(len(y), np.nan) for k in (
            "old_behavior", "old_fusion", "old_eeg", "time", "behavior", "optimized",
            "required_eeg", "front_eeg", "eyes_eeg", "clean_eeg", "posterior_eeg",
            *blocks.keys())}
        choices = []
        for fold in folds:
            train = np.flatnonzero(np.isin(groups, fold["train"]) & scope_mask)
            hold = np.flatnonzero(np.isin(groups, fold["test"]) & scope_mask)
            if np.intersect1d(groups[train], groups[hold]).size:
                raise ValueError("Outer participant leak")
            inner = fold["inner_validation_people"]
            if sorted(p for g in inner for p in g) != sorted(fold["train"]):
                raise ValueError("Inner participant leak")
            possible = [(d, k) for d in designs for k in front_options]
            inner_oof = {p: np.full(len(y), np.nan) for p in possible}
            for people_in in inner:
                val = train[np.isin(groups[train], people_in)]
                dev = train[~np.isin(groups[train], people_in)]
                result = predict_fold(dev, val, designs,
                                      {k: blocks[k] for k in ("original_spectra", "front_power", "front_power_erp")},
                                      groups, y)
                for d, k in possible:
                    inner_oof[d, k][val] = result[d][k]
            rank = {p: ranked_score(y[train], v[train], groups[train]) for p, v in inner_oof.items()}
            behavior_key = max(designs, key=lambda d: rank[d, ("none", 0)])
            key = max(front_options, key=lambda k: rank[behavior_key, k])
            required_key = max(front_options[1:], key=lambda k: rank[behavior_key, k])
            outer = predict_fold(train, hold, designs, blocks, groups, y)
            chosen = outer[behavior_key]
            scores["behavior"][hold] = chosen["none", 0]
            scores["optimized"][hold] = chosen[key]
            scores["required_eeg"][hold] = chosen[required_key]
            scores["old_behavior"][hold] = outer["standard"]["none", 0]
            for label, x in (("time", standard[:, :1]), ("old_fusion", np.column_stack((standard, spectra))),
                             ("old_eeg", spectra), *[(s + "_eeg", extract_matrix(rows, s, "spec"))
                                                     for s in ("front", "eyes", "clean", "posterior")]):
                train_eeg = train if label == "time" else train[valid[train]]
                pred = predict(fit(x[train_eeg], y[train_eeg], groups[train_eeg], CONFIG), x[hold])
                if label != "time":
                    pred[~valid[hold]] = np.nan
                scores[label][hold] = pred
            for block in blocks:
                scores[block][hold] = chosen[block, 1.]
            choices.append({"fold": fold["outer_fold"], "train": fold["train"], "hold": fold["test"],
                            "inner": inner, "behavior": behavior_key, "eeg": key, "required_eeg": required_key,
                            "development_scores": {f"{d}/{k[0]}/{k[1]}": float(v) for (d, k), v in rank.items()}})
        summaries = {}
        for k, pred in scores.items():
            summ, _ = describe(y, pred, groups, boot)
            summaries[k] = {n: summ[n] for n in ("scored_labelled", "scored_participants",
                            "both_classes_scored_participants", "macro_auc", "pooled_auc", "balanced_accuracy",
                            "coverage_adjusted_macro_auc", "ci95", "sensitivity", "specificity")}
        diffs = {}
        for new, ref in (("optimized", "behavior"), ("required_eeg", "behavior"), ("behavior", "old_behavior"),
                         ("optimized", "old_fusion"), ("front_power", "behavior"), ("front_power_erp", "behavior"),
                         ("eyes_power", "behavior"), ("clean_power", "behavior"), ("posterior_power", "behavior"),
                         ("eyes_power_erp", "behavior"), ("clean_power_erp", "behavior"), ("posterior_power_erp", "behavior"),
                         ("clean_power", "old_fusion"), ("clean_eeg", "front_eeg"), ("clean_eeg", "old_eeg")):
            a = scores[new]
            b = scores[ref]
            paired = np.isfinite(a) & np.isfinite(b)
            left, l = describe(y, np.where(paired, a, np.nan), groups, boot)
            right, r = describe(y, np.where(paired, b, np.nan), groups, boot)
            diffs[new + "_minus_" + ref] = {
                "matched": int(paired.sum()), "auroc_delta": left["macro_auc"] - right["macro_auc"],
                "left_auc": left["macro_auc"], "right_auc": right["macro_auc"],
                "ci95": ci(l["macro_auc"] - r["macro_auc"]),
                "balanced_accuracy_delta": left["balanced_accuracy"] - right["balanced_accuracy"],
                "balanced_accuracy_ci95": ci(l["balanced_accuracy"] - r["balanced_accuracy"])}
        records[scope] = {"models": summaries, "paired": diffs, "selection": choices}
        all_scores[scope] = scores
    usable_power = np.isfinite(front).all(axis=1) & valid
    participant_power = {p: np.median(front[(groups == p) & usable_power], axis=0)
                         for p in people if np.any((groups == p) & usable_power)}
    deviations = np.stack([front[i] - participant_power[g] for i, g in enumerate(groups)
                           if valid[i] and np.isfinite(front[i]).all()])
    beta = np.array([(v["raw_coupling"], v["clean_coupling"]) for v in rows if "raw_coupling" in v])
    ordinal = standard[:, 0].astype(int)
    diagnostics = {
        "time_thirds": [{"negative": int(((y == 0) & mask).sum()), "positive": int(((y == 1) & mask).sum())}
                        for mask in ((ordinal <= 5), (ordinal > 5) & (ordinal <= 10), ordinal > 10)],
        "overall_behavior_eligible": int(valid_behavior.sum()), "old_common": int(valid.sum()),
        "both_labels_all": sum(len(np.unique(y[groups == p])) == 2 for p in people),
        "both_labels_common": sum(len(np.unique(y[(groups == p) & valid])) == 2 for p in people),
        "between_subject_sd_of_median_powers": np.std(list(participant_power.values()), axis=0).tolist(),
        "within_subject_sd_powers": np.std(deviations, axis=0).tolist(),
        "reference_coupling_median_all_front_leads_raw_and_clean": np.median(beta.transpose(1, 0, 2).reshape(2, -1), axis=1).tolist(),
        "common_features_available": {name: int(np.isfinite(m[valid]).all(axis=1).sum()) for name, m in blocks.items()},
        "total": len(y), "common_positive": int(y[valid].sum()), "total_positive": int(y.sum())}
    first_scores = all_scores["common"]
    reference = {"old_behavior": .700138265614456, "old_fusion": .698610481229529}
    for k, expected in reference.items():
        if not np.isclose(ranked_score(y, first_scores[k], groups), expected, rtol=0, atol=1e-10):
            raise ValueError(f"Original baseline mismatch for {k}")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "predictions.csv").open("w", newline="", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerow(["scope", "participant", "probe", "label", "old_gate", "outer_fold", *first_scores])
        for scope, scores in all_scores.items():
            lookup = {p: fold["outer_fold"] for fold in folds for p in fold["test"]}
            for i, (p, k) in enumerate(keys):
                writer.writerow([scope, p, k, y[i], int(valid[i]), lookup[p],
                                 *[format(a[i], ".15g") if np.isfinite(a[i]) else "" for a in scores.values()]])
    result = {"source": "https://zenodo.org/records/23592484", "cohort": "28 ATTLAPSE healthy controls",
              "primary": "common/optimized minus common/behavior, participant-macro within-person AUROC",
              "historical": "Same 5 person-held-out folds previously inspected; new choices nested in 4 person-held-out inner folds.",
              "confidence": "Paired bootstrap (4000 draws) on held-out people and their frozen OOF scores. Excludes refitting variability.",
              "split_sha256": hashlib.sha256(split_file.read_bytes()).hexdigest(),
              "source_features_sha256": hashlib.sha256(feature_path.read_bytes()).hexdigest(),
              "diagnostics": diagnostics, "results": records}
    (out / "evaluation.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attlapse", type=Path)
    parser.add_argument("features", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    comparison(args.attlapse, args.features, args.output)
