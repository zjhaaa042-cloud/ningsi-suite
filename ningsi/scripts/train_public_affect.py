"""Fit fixed, participant-held-out classifiers for SAM40 stress and EmoEEG-MC valence.

Training-only dependencies: scipy, xlrd. Runtime uses NumPy. Pass the extracted
SAM40 Data directory, the fetch_emoeeg_affect.py cache and the dataset README.md.
Original waveforms and behavioural tables are not placed in the application.
This reproduces the original frozen label protocol. For new valence research,
use resolve_emoeeg_video_order.py and validate_valence_mechanisms.py: earlier
OpenNeuro event tables resolve additional exact video ratings.
"""
from __future__ import annotations

import argparse
import ast
import csv
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path
import re

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly

from ningsi.models.affect import FEATURES, SPEC, feature_vector
from ningsi.models.human import fit, logits, sigmoid, write_json
from ningsi.models.logistic import roc_auc
from ningsi.models.research_features import condition_window


SEED = 20261115
RATE = 250
STRESS = "SAM40 (Ghosh et al., doi:10.6084/m9.figshare.14562090.v1, CC BY 4.0)"
VALENCE = "EmoEEG-MC (Xu et al., OpenNeuro ds005540, CC0)"
LABELS = {"stress": ["较低自评压力倾向", "较高自评压力倾向"],
          "valence": ["负效价倾向", "正效价倾向"]}
SORTED_CLIPS = [f"{family}{num}" for family in ("sad", "dis", "fear", "neu", "joy", "ten", "ins")
                for num in (4, 5, 8)]


def trial_features(values, rate):
    # Polyphase frequency conversion is offline on public source recordings.
    # Every actual classifier input then uses the same nominal rate as deployment.
    gcd = math.gcd(RATE, rate)
    signal = resample_poly(values, RATE // gcd, rate // gcd, axis=1)
    features, rejected = [], Counter()
    # Every trial has five evenly spaced decisions using six seconds of context
    # and a four second decision window. Training gives trials equal weight.
    for end in np.linspace(6 * RATE, signal.shape[1], 5, dtype=int):
        segment = signal[:, end - 6 * RATE:end]
        win = condition_window(segment, np.arange(segment.shape[1]) / RATE, RATE, 1.)["window"]
        if win.usable:
            features.append(feature_vector(win).tolist())
        else:
            rejected.update(win.quality.reasons)
    return features, dict(rejected)


def sam_tasks(root):
    import xlrd
    sheet = xlrd.open_workbook(str(root / "scales.xls")).sheet_by_index(0)
    channels = [line.split()[-1] for line in (root / "Coordinates.locs").read_text().splitlines()]
    assert [channels[2], channels[31]] == ["Fp1", "Fp2"]
    tasks = []
    for i in range(2, sheet.nrows):
        row = sheet.row_values(i)
        person = int(row[0])
        for trial in range(1, 4):
            for j, task in enumerate(("Arithmetic", "Mirror_image", "Stroop")):
                rating = float(row[1 + (trial - 1) * 3 + j])
                assert 1 <= rating <= 10
                file = root / "raw_data" / f"{task}_sub_{person}_trial{trial}.mat"
                tasks.append((file, person, f"{task}-{trial}", rating, int(rating >= 6)))
    return tasks


def stress_trial(task):
    file, person, trial, rating, label = task
    values = loadmat(file, variable_names=["Data"])["Data"]
    if values.shape != (32, 3200):
        raise ValueError(f"Unexpected SAM40 shape: {file}")
    features, reasons = trial_features(values[[2, 31]], 128)
    return dict(person=person, trial=trial, rating=rating, label=label, windows=features, rejected=reasons)


def video_orders(readme):
    first = readme.index("### Full Trial Participants")
    block = readme[first:readme.index("### Participants with Missing", first)]
    orders = {int(num): ast.literal_eval(seq) for num, seq in
              re.findall(r"(?m)^(\d+)\. `(\[.*?\])`", block)}
    # Only subjects 1–53 have identical within-context clip orders in both
    # modalities, which is necessary for the label-identification rule below.
    assert set(orders) == set(range(1, 54)) and all(sorted(s) == sorted(SORTED_CLIPS) for s in orders.values())
    return orders


def valence_subject(task):
    path, person, order = task
    output = []
    with np.load(path, allow_pickle=False) as source:
        pairs = source["channels"]
        behavior = list(csv.DictReader(io.StringIO(str(source["ratings"])), delimiter="\t"))
    if len(behavior) != 42 or pairs.shape != (2, 21, 6000):
        raise ValueError(f"Incomplete video trials: {path}")
    for position, clip in enumerate(order):
        # Each consecutive pair of three-trial blocks contains video and imagery
        # in randomized block order. TSV does not identify the block context.
        # Video labels are identifiable only when BOTH possible video ratings
        # imply the same class. Never guess the context or average its labels.
        first = 6 * (position // 3) + position % 3
        idx = (first, first + 3)
        if any(float(behavior[i]["trial_number"].strip("[]")) != i + 1 for i in idx):
            raise ValueError(f"Non-sequential behaviour record: {path}")
        rating = [float(behavior[i]["score_8"]) for i in idx]
        if not all(0 <= value <= 7 for value in rating):
            raise ValueError(f"Invalid valence: {path}")
        label = 1 if min(rating) > 3.5 else (0 if max(rating) < 3.5 else None)
        features, reasons = trial_features(pairs[:, SORTED_CLIPS.index(clip)], 200)
        output.append(dict(person=person, trial=clip, rating=rating, label=label, windows=features, rejected=reasons))
    return output


def validate(rows, model):
    details = []
    window_y, window_p = [], []
    for row in rows:
        if row["label"] is None or not row["windows"]:
            continue
        scores = sigmoid(logits(model, np.array(row["windows"])))
        window_p.extend(scores)
        window_y.extend([row["label"]] * len(scores))
        details.append({"person": row["person"], "trial": row["trial"], "label": row["label"],
                        "score": float(np.mean(scores)), "usable_windows": len(scores)})
    people = sorted({r["person"] for r in details})
    macro_aucs, confusion_by_person = [], []
    for person in people:
        selected = [r for r in details if r["person"] == person]
        auc = roc_auc([r["label"] for r in selected], [r["score"] for r in selected])
        if auc is not None:
            macro_aucs.append(auc)
        observed = np.array([r["label"] for r in selected])
        inferred = np.array([r["score"] >= .5 for r in selected])
        confusion_by_person.append([np.sum(inferred & (observed == 1)), np.sum(observed == 1),
                                    np.sum(~inferred & (observed == 0)), np.sum(observed == 0)])
    labels = np.array([r["label"] for r in details])
    scores = np.array([r["score"] for r in details])
    sensitivity = float(np.mean(scores[labels == 1] >= .5)) if (labels == 1).any() else None
    specificity = float(np.mean(scores[labels == 0] < .5)) if (labels == 0).any() else None
    rng = np.random.default_rng(SEED)
    ci = (np.percentile(rng.choice(macro_aucs, size=(2000, len(macro_aucs))).mean(axis=1), [2.5, 97.5]).tolist()
          if len(macro_aucs) > 1 else None)
    sampled = np.asarray(confusion_by_person)[rng.integers(len(people), size=(2000, len(people)))].sum(axis=1)
    valid = sampled[(sampled[:, 1] > 0) & (sampled[:, 3] > 0)]
    ba_ci = np.percentile(.5 * (valid[:, 0] / valid[:, 1] + valid[:, 2] / valid[:, 3]), [2.5, 97.5]).tolist()
    labelled = sum(row["label"] is not None for row in rows)
    return {"trials_total": len(rows), "trials_with_binary_rating": labelled, "trials_scored": len(details),
            "people_with_both_scored_labels": len(macro_aucs), "scored_people": len(people),
            "labelled_trial_coverage": len(details) / labelled,
            "usable_labelled_windows": len(window_y), "labelled_windows_attempted": labelled * 5,
            "eligible_window_coverage": len(window_y) / (labelled * 5),
            "trial_positive": int(labels.sum()), "trial_negative": int(len(labels) - labels.sum()),
            "trial_sensitivity": sensitivity, "trial_specificity": specificity,
            "trial_balanced_accuracy": (sensitivity + specificity) / 2 if None not in (sensitivity, specificity) else None,
            "trial_balanced_accuracy_ci95": ba_ci, "all_source_trial_coverage": len(details) / len(rows),
            "trial_accuracy": float(np.mean((scores >= .5) == labels)), "trial_pooled_auc": roc_auc(labels, scores),
            "trial_within_person_macro_auc": float(np.mean(macro_aucs)) if macro_aucs else None,
            "within_person_macro_auc_ci95": ci, "single_window_pooled_auc": roc_auc(window_y, window_p),
            "constant_score_auc_and_balanced_accuracy": .5}, details


def train(rows, name, out):
    individuals = sorted({r["person"] for r in rows})
    # Fixed random split over all source participants before QC or model fitting.
    held = set(np.random.default_rng(SEED).permutation(individuals)[:math.ceil(len(individuals) / 4)].tolist())
    development = [r for r in rows if r["person"] not in held and r["label"] is not None and r["windows"]]
    counts = Counter(r["person"] for r in development)
    x, y, weights = [], [], []
    for r in development:
        n = len(r["windows"])
        x.extend(r["windows"])
        y.extend([r["label"]] * n)
        weights.extend([1 / (counts[r["person"]] * n)] * n)
    x, y, weights = np.array(x), np.array(y), np.array(weights)
    for label in (0, 1):
        mask = y == label
        weights[mask] /= 2 * weights[mask].sum()
    # Single, fixed regularization. No test-informed model or threshold selection.
    params = fit(x, y, weights, l2=.05)
    held_rows = [r for r in rows if r["person"] in held]
    metrics, predictions = validate(held_rows, params)
    classes = Counter(row["label"] for row in development)
    majority = int(classes[1] > classes[0])
    observed = np.array([row["label"] for row in predictions])
    metrics["development_majority_label"] = majority
    metrics["majority_trial_accuracy"] = float(np.mean(observed == majority))
    if name == "stress":
        tasks = sorted({r["trial"].split("-")[0] for r in development})
        task_mean = {task: float(np.mean([r["label"] for r in development if r["trial"].split("-")[0] == task]))
                     for task in tasks}
        baseline = np.asarray([task_mean[r["trial"].split("-")[0]] for r in predictions])
        metrics["task_only_trial_pooled_auc"] = roc_auc(observed, baseline)
        metrics["task_only_trial_balanced_accuracy"] = .5 * float(
            np.mean(baseline[observed == 1] >= .5) + np.mean(baseline[observed == 0] < .5))
        metrics["task_only_development_positive_fraction"] = task_mean
        metrics["within_task_eeg_aucs"] = {}
        for task in tasks:
            subset = [r for r in predictions if r["trial"].split("-")[0] == task]
            metrics["within_task_eeg_aucs"][task] = roc_auc([r["label"] for r in subset], [r["score"] for r in subset]) if subset else None
    model_id = hashlib.sha256(json.dumps(params, sort_keys=True, allow_nan=False).encode()).hexdigest()
    result = {"dataset": STRESS if name == "stress" else VALENCE, "parameters": params,
              "model_id": model_id, "labels": LABELS[name], "holdout": metrics,
              "target": ("Post-task SAM40 self-rated stress 6–10 vs 1–5 on original 1–10 scale; only three rated tasks."
                         if name == "stress" else "Video-trial self-rated valence above vs below 3.5 on original 0–7 scale; "
                         "retain only identifiable same-polarity paired-context ratings for complete participants 1–53."),
              "source_people": len(individuals), "source_trials": len(rows),
              "source_label_counts": {str(key): n for key, n in Counter(r["label"] for r in rows).items()},
              "target_device_validated": False,
              "trained_people": sorted(counts), "held_out_people": sorted(held)}
    with (out / f"{name}_heldout.csv").open("w", encoding="utf-8", newline="") as dest:
        writer = csv.DictWriter(dest, fieldnames=predictions[0].keys())
        writer.writeheader()
        writer.writerows(predictions)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam", type=Path, required=True)
    parser.add_argument("--emo", type=Path, required=True)
    parser.add_argument("--readme", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        sam = list(pool.map(stress_trial, sam_tasks(args.sam), chunksize=4))
        order = video_orders(args.readme.read_text(encoding="utf-8"))
        emo = [row for subject in pool.map(valence_subject, [(args.emo / f"sub-{p:02d}.npz", p, sequence)
                   for p, sequence in order.items()]) for row in subject]
    write_json(args.output / "trials.json", {"stress": sam, "valence": emo})
    bundle = {"spec": SPEC, "electrodes": ["Fp1", "Fp2"], "features": list(FEATURES), "input_rate_hz": RATE,
              "models": {name: train(rows, name, args.output) for name, rows in (("stress", sam), ("valence", emo))}}
    write_json(args.output / "frontal_affect.json", bundle)
    print(json.dumps({k: m["holdout"] for k, m in bundle["models"].items()}, indent=2))
