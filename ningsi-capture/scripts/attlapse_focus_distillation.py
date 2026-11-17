"""Exploratory EOG-privileged training for inference from two forehead EEG leads only."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from attlapse_artifact_diagnostic import CONFIG, inputs
from attlapse_focus_optimization import behavior, estimate_residual, extract_matrix, logit, personal_delta, residual_score
from attlapse_two_channel_eval import SEED, ci, describe, fit, predict, sigmoid


def fit_map(raw, clean, people):
    eligible = np.isfinite(clean).all(axis=1)
    x, y, g = raw[eligible], clean[eligible], people[eligible]
    mu = np.nanmedian(x, axis=0)
    sd = np.nanquantile(x, .75, axis=0) - np.nanquantile(x, .25, axis=0)
    sd[sd < 1e-8] = 1.
    z = map_design(x, mu, sd)
    _, idx, count = np.unique(g, return_inverse=True, return_counts=True)
    weights = 1 / count[idx]
    weights /= weights.sum()
    penalty = .1 * np.eye(z.shape[1])
    penalty[0, 0] = 0.
    beta = np.linalg.solve((z.T * weights) @ z + penalty, (z.T * weights) @ y)
    return {"median": mu.tolist(), "iqr": sd.tolist(), "coef": beta.tolist()}


def map_design(x, mu, sd):
    standardized = np.clip((x - mu) / sd, -4, 4)
    return np.column_stack((np.ones(len(x)), np.where(np.isfinite(x), standardized, 0),
                            (~np.isfinite(x).all(axis=1)).astype(float)))


def map_spectra(state, x):
    return map_design(x, np.asarray(state["median"]), np.asarray(state["iqr"])) @ np.asarray(state["coef"])


def fit_student(train, inner, raw, teacher, behavior_x, y, g, good):
    x, group, label = raw[train], g[train], y[train]
    estimated = np.full((len(train), teacher.shape[1]), np.nan)
    for people in inner:
        val = np.isin(group, people)
        if not val.any() or not (~val).any():
            raise ValueError("Invalid inner person split")
        state = fit_map(x[~val], teacher[train[~val]], group[~val])
        estimated[val] = map_spectra(state, x[val])
    if not np.isfinite(estimated).all():
        raise ValueError("Teacher target reconstruction did not cover every development row")
    features = np.column_stack((estimated, personal_delta(estimated, group, good[train])))
    behavior_state = fit(behavior_x[train], label, group, CONFIG)
    base = logit(predict(behavior_state, behavior_x[train]))
    residual = estimate_residual(features, base, label, group)
    return {"behavior": behavior_state, "mapping": fit_map(x, teacher[train], group),
            "residual": [np.asarray(part).tolist() for part in residual]}


def predict_student(state, raw, behavior_x, people):
    # Each person's records must be supplied in time order; prior probes require no outcomes.
    estimate = map_spectra(state["mapping"], raw)
    change = personal_delta(estimate, people, np.ones(len(raw), dtype=bool))
    behavioral = logit(predict(state["behavior"], behavior_x))
    residual = residual_score(np.column_stack((estimate, change)), tuple(np.asarray(v) for v in state["residual"]))
    return sigmoid(behavioral + residual)


def assess(root, feats, out):
    bench = root / "two-channel-benchmark"
    keys, g, y, _, spectra, rate = inputs(bench / "features250.jsonl")
    rows = sorted(json.loads(feats.read_text(encoding="utf8")), key=lambda r: (r["participant"], r["probe"]))
    assert keys == [(r["participant"], r["probe"]) for r in rows] and rate == 250
    folds = json.loads((bench / "evaluation250/splits.json").read_text(encoding="utf8"))["outer"]
    standard, _ = behavior(root, keys)
    front = extract_matrix(rows, "front", "spec")
    corrected = extract_matrix(rows, "clean", "spec")
    raw = np.column_stack((spectra, front))
    good = np.isfinite(standard).all(axis=1) & np.isfinite(spectra).all(axis=1)
    overlap = good & np.isfinite(front).all(axis=1) & np.isfinite(corrected).all(axis=1)
    sensors = {"front_shared": front, "corrected_shared": corrected}
    r = {n: np.full(len(keys), np.nan) for n in
         ("student", "behavior", "original_fusion", "front_shared", "corrected_shared",
          "front_only_shared", "corrected_only_shared", "behavior_shared")}
    representation_scores = []
    for fold in folds:
        tr = np.flatnonzero(np.isin(g, fold["train"]) & good)
        te = np.flatnonzero(np.isin(g, fold["test"]) & good)
        state = fit_student(tr, fold["inner_validation_people"], raw, corrected, standard, y, g, good)
        r["student"][te] = predict_student(state, raw[te], standard[te], g[te])
        estimated = map_spectra(state["mapping"], raw[te])
        for row, measured, person in zip(estimated, corrected[te], g[te], strict=True):
            if np.isfinite(measured).all():
                representation_scores.append([str(person), row.tolist(), measured.tolist()])
        r["behavior"][te] = predict(state["behavior"], standard[te])
        fusion = np.column_stack((standard, spectra))
        r["original_fusion"][te] = predict(fit(fusion[tr], y[tr], g[tr], CONFIG), fusion[te])
        tr = tr[overlap[tr]]
        te = te[overlap[te]]
        backbone = fit(standard[tr], y[tr], g[tr], CONFIG)
        bt = logit(predict(backbone, standard[tr]))
        bh = logit(predict(backbone, standard[te]))
        r["behavior_shared"][te] = sigmoid(bh)
        for name, sensor in sensors.items():
            # Reference and raw models have identical training windows, historical windows and test windows.
            d = np.column_stack((sensor, personal_delta(sensor, g, overlap)))
            residual = estimate_residual(d[tr], bt, y[tr], g[tr])
            r[name][te] = sigmoid(bh + residual_score(d[te], residual))
            key = name.replace("_shared", "_only_shared")
            r[key][te] = predict(fit(sensor[tr], y[tr], g[tr], CONFIG), sensor[te])
    n = len(np.unique(g))
    boot = np.random.default_rng(SEED).multinomial(n, np.full(n, 1 / n), size=4000)
    results, draws = {}, {}
    for name, pred in r.items():
        metrics, draws[name] = describe(y, pred, g, boot)
        results[name] = {k: metrics[k] for k in ("macro_auc", "balanced_accuracy", "scored_labelled",
                         "both_classes_scored_participants", "ci95")}
    paired = {}
    for a, b in (("student", "behavior"), ("student", "original_fusion"),
                 ("corrected_shared", "front_shared"), ("corrected_shared", "behavior_shared"),
                 ("corrected_only_shared", "front_only_shared")):
        assert np.array_equal(np.isfinite(r[a]), np.isfinite(r[b]))
        paired[a + "_minus_" + b] = {"delta": results[a]["macro_auc"] - results[b]["macro_auc"],
                                     "ci95": ci(draws[a]["macro_auc"] - draws[b]["macro_auc"])}
    subjects = np.array([v[0] for v in representation_scores])
    estimated = np.array([v[1] for v in representation_scores])
    measured = np.array([v[2] for v in representation_scores])
    corrs = [np.array([np.corrcoef(estimated[subjects == p, j], measured[subjects == p, j])[0, 1]
                       for j in range(6)]) for p in np.unique(subjects) if np.sum(subjects == p) >= 3]
    evaluation = {"target_features": "log(theta, alpha, beta) for channel mean log power and left-minus-right lead",
                  "median_within_subject_teacher_reconstruction_r": np.median(corrs, axis=0).tolist(),
                  "fixed_map_penalty": .1, "outcome_informed_secondary_analysis": True,
                  "metrics": results, "paired": paired}
    out.mkdir(parents=True, exist_ok=True)
    (out / "distillation.json").write_text(json.dumps(evaluation, indent=2, allow_nan=False), encoding="utf8")
    with (out / "distillation_predictions.csv").open("w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["participant", "probe", "label", *r])
        for i, (person, probe) in enumerate(keys):
            w.writerow([person, probe, y[i], *[v[i] if np.isfinite(v[i]) else "" for v in r.values()]])
    # This refit is an experimental replay candidate, distinct from all performance estimates above.
    dev = np.flatnonzero(good)
    final = fit_student(dev, [f["test"] for f in folds], raw, corrected, standard, y, g, good)
    final["scope"] = "Offline ATTLAPSE healthy controls, 250 Hz, physical pre18_2 gate; not activated."
    path = out / "student_research_weights.json"
    path.write_text(json.dumps(final, indent=2, allow_nan=False), encoding="utf8")
    restored = json.loads(path.read_text(encoding="utf8"))
    assert np.array_equal(predict_student(final, raw[dev], standard[dev], g[dev]),
                          predict_student(restored, raw[dev], standard[dev], g[dev]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attlapse", type=Path)
    parser.add_argument("features", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    assess(args.attlapse, args.features, args.output)
