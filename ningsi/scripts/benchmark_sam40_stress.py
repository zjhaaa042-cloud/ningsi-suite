"""Subject-independent Fp1/Fp2 stress optimization on SAM40 self-ratings.

Run with ningsi/src on PYTHONPATH. Requires NumPy, SciPy and xlrd.
Refits old training in each comparison. No validation subject supplies training labels.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from compare_frontal_affect import filtered_windows, paired_metrics
from train_public_affect import RATE, sam_tasks
from ningsi.models.affect import feature_vector
from ningsi.models.human import fit, logits, sigmoid, write_json
from ningsi.models.logistic import roc_auc
from ningsi.signal.window import analyze_window


EEG = ("legacy", "dense", "rest")
HISTORY = ("history", "history_eeg")
DIAGNOSTICS = ("dense_old_bands", "sparse_median", "task_only", "first_rating_only")
SEED = 676125
FIRST = np.linspace(6 * RATE, 25 * RATE, 5, dtype=int)
ENDS = np.union1d(FIRST, np.arange(6 * RATE, 25 * RATE + 1, RATE))
FEATURES = ("log_rel_theta", "log_rel_alpha", "log_rel_beta", "log_abs_theta",
            "log_abs_alpha", "log_abs_beta")


def features(path):
    data = loadmat(path, variable_names=["Data"])["Data"]
    if data.shape != (32, 3200):
        raise ValueError(f"Invalid SAM40 shape {path}")
    clean, original, reasons = [], [], Counter()
    for end, pair in zip(ENDS, filtered_windows(data[[2, 31]], 128, ENDS)):
        w = analyze_window(pair, RATE)
        if not w.usable:
            reasons.update(w.quality.reasons)
            continue
        x = feature_vector(w).tolist()
        absolute = [np.log(w.powers[b]) for b in ("theta", "alpha", "beta")]
        clean.append(x + absolute)
        if end in FIRST:
            original.append(x + absolute)
    return original, clean, reasons


def prepare(cache, output):
    source = json.loads((cache / "evaluation/trials.json").read_text(encoding="utf8"))["stress"]
    frozen = {(r["person"], r["trial"]): r for r in source}
    rows, initial_rest = [], {}
    for path, person, trial, rating, label in sam_tasks(cache / "Data"):
        if person not in initial_rest:
            rest = cache / "Data/raw_data" / f"Relax_sub_{person}_trial1.mat"
            _, clean, _ = features(rest)
            initial_rest[person] = np.median(clean, axis=0).tolist() if clean else None
        row = frozen[person, trial]
        assert row["rating"] == rating and row["label"] == label
        old, clean, reasons = features(path)
        old_three = [x[:3] for x in old]
        assert len(old) == len(row["windows"]) and np.allclose(old_three, row["windows"], atol=1e-8, rtol=1e-8)
        task, visit = trial.rsplit("-", 1)
        median = np.median(clean, axis=0).tolist() if clean else []
        rows.append(dict(person=person, trial=trial, task=task, visit=int(visit),
                         rating=rating, label=label, old=row["windows"], median=median,
                         sparse_median=np.median(old, axis=0).tolist() if old else [],
                         dense_three=[x[:3] for x in clean],
                         rest=initial_rest[person], dense_good=len(clean), rejected=reasons,
                         prior=frozen[person, task+"-1"]["rating"] if int(visit)>1 else None))
    write_json(output / "features.json", rows)
    return rows


def design(row, arm):
    if arm == "legacy":
        return row["old"]
    if arm in ("history", "first_rating_only"):
        return [[row["prior"]]] if row["prior"] is not None else []
    if arm == "task_only":
        tasks=("Arithmetic", "Mirror_image", "Stroop")
        return [[float(row["task"] == task) for task in tasks]] if row["prior"] is not None else []
    if arm == "dense_old_bands":
        return row["dense_three"]
    if arm == "sparse_median":
        return [row["sparse_median"]] if row["sparse_median"] else []
    x = row["median"]
    if not x:
        return []
    if arm == "dense":
        return [x]
    # Missing independent resting calibration gets zero deviation, no extra
    # indicator. Absolute features remain available for every eligible trial.
    delta = np.subtract(x, row["rest"]).tolist() if row["rest"] else [0.] * len(x)
    if arm == "rest":
        return [x + delta]
    return [[row["prior"]] + x + delta] if row["prior"] is not None else []


def train_and_predict(train, test, arm):
    train = [r for r in train if design(r, arm)]
    eligible = [r for r in test if design(r, arm)]
    counts = Counter(r["person"] for r in train)
    x, y, w = [], [], []
    for row in train:
        xx = design(row, arm)
        x.extend(xx)
        y.extend([row["label"]] * len(xx))
        w.extend([1. / (counts[row["person"]] * len(xx))] * len(xx))
    y, w = np.array(y), np.array(w)
    for label in (0, 1):
        mask = y == label
        w[mask] /= 2*w[mask].sum()
    params = fit(np.array(x), y, w, l2=.05)
    predictions = {(r["person"], r["trial"]):
                   float(np.mean(sigmoid(logits(params, np.array(design(r, arm)))))) for r in eligible}
    return params, predictions


def run_fold(train, test, arm_names):
    assert not {r["person"] for r in train} & {r["person"] for r in test}
    out, weights = {}, {}
    for arm in arm_names:
        # Identical training opportunities for the paired history ablation.
        arm_train = [r for r in train if r["median"]] if arm in HISTORY else train
        weights[arm], out[arm] = train_and_predict(arm_train, test, arm)
    return [dict(person=r["person"], trial=r["trial"], label=r["label"], rating=r["rating"],
                 **{arm: pred.get((r["person"],r["trial"])) for arm,pred in out.items()})
            for r in test], weights


def folds(people, splits):
    order = np.random.default_rng(SEED).permutation(sorted(people)).tolist()
    return [sorted(order[i::splits]) for i in range(splits)]


def cross_validate(rows, arms, splits=5):
    people = {r["person"] for r in rows}
    oof, assignment = [], folds(people, splits)
    for val in assignment:
        train = [r for r in rows if r["person"] not in val]
        test = [r for r in rows if r["person"] in val]
        predictions, _ = run_fold(train, test, arms)
        oof.extend(predictions)
    assert len({(r["person"], r["trial"]) for r in oof}) == len(rows)
    return oof, assignment


def paired(rows, a, b):
    data = [dict(person=r["person"], label=r["label"], base=r[a], expanded=r[b])
            for r in rows if r[a] is not None and r[b] is not None]
    return paired_metrics(data)


def choose(rows):
    # Fixed primary outcome mirrors the original binary self-rating claim.
    # Compare candidate arms on the same old-eligible trial set only.
    return max(EEG, key=lambda arm: paired(rows, "legacy", arm)["expanded"]["balanced_accuracy"])


def diagnostics(rows):
    per_person = {p: [r for r in rows if r["person"] == p] for p in {r["person"] for r in rows}}
    y = np.array([r["rating"] for r in rows])
    person_mean = np.array([np.mean([t["rating"] for t in per_person[r["person"]]]) for r in rows])
    task_means = {task: float(np.mean([r["rating"] for r in rows if r["task"]==task]))
                  for task in {r["task"] for r in rows}}
    qc = Counter()
    for r in rows:
        qc.update(r["rejected"])
    paired_ratings = [(r["prior"],r["rating"]) for r in rows if r["visit"] > 1]
    return dict(people=len(per_person), trials=len(rows), original_clean=sum(bool(r["old"]) for r in rows),
                dense_clean=sum(bool(r["median"]) for r in rows),
                dense_usable_windows=sum(r["dense_good"] for r in rows),
                dense_attempted_windows=len(rows)*len(ENDS), dense_rejection_counts=qc,
                participants_with_rest=sum(bool(g[0]["rest"]) for g in per_person.values()),
                only_one_label_people=sum(len({r["label"] for r in g})==1 for g in per_person.values()),
                only_one_label_among_original_scorable=sum(len({r["label"] for r in g if r["old"]})==1
                                                           for g in per_person.values()),
                middle_ratings_5_or_6=int(np.isin(y,[5,6]).sum()),
                between_person_fraction_rating_variance=float(np.sum((person_mean-y.mean())**2)/np.sum((y-y.mean())**2)),
                same_task_first_rating_correlation=float(np.corrcoef(np.asarray(paired_ratings).T)[0,1]),
                task_mean_ratings=task_means)


def run(cache, output):
    output.mkdir(parents=True,exist_ok=True)
    bundle=json.loads((cache/"evaluation/frontal_affect.json").read_text(encoding="utf8"))["models"]["stress"]
    rows = prepare(cache, output)
    held = set(bundle["held_out_people"])
    dev = [r for r in rows if r["person"] not in held]
    test = [r for r in rows if r["person"] in held]
    arms = EEG + HISTORY + DIAGNOSTICS
    d_preds, assignment = cross_validate(dev, arms)
    # Nested subject-held-out estimate of the actual EEG selection policy.
    nested, chosen = [], []
    for val in assignment:
        inner = [r for r in dev if r["person"] not in val]
        compare, inner_groups = cross_validate(inner, EEG, splits=4)
        choice = choose(compare)
        chosen.append(dict(test=val, inner=inner_groups, selected=choice))
        for r in d_preds:
            if r["person"] in val:
                nested.append({**r,"selected":r[choice]})
    selection = choose(d_preds)
    h_preds, weights = run_fold(dev,test,arms)
    for r in h_preds:
        r["selected"] = r[selection]
    old_scores = {(int(r["person"]),r["trial"]):float(r["score"]) for r in
                  csv.DictReader((cache/"evaluation/stress_heldout.csv").open(encoding="utf8"))}
    new_scores = {(r["person"],r["trial"]):r["legacy"] for r in h_preds if r["legacy"] is not None}
    assert set(old_scores) == set(new_scores)
    max_error = max(abs(v-old_scores[k]) for k,v in new_scores.items())
    assert max_error < 1e-8
    def summarize(pred, phase):
        return dict(phase=phase, assigned_trials=len(pred), comparisons={
            name: paired(pred,a,b) for name,a,b in [
                ("selection_vs_original","legacy","selected"),
                ("dense_vs_original","legacy","dense"),("rest_vs_original","legacy","rest"),
                ("history_plus_eeg_vs_history","history","history_eeg"),
                ("history_plus_eeg_vs_original","legacy","history_eeg"),
                ("history_vs_original","legacy","history"),
                ("density_only_vs_original","legacy","dense_old_bands"),
                ("sparse_median_absolute_vs_original","legacy","sparse_median"),
                ("first_rating_vs_task_only","task_only","first_rating_only")]},
            native_eeg={a:paired(pred,a,a) for a in EEG}, coverage={arm:sum(r[arm] is not None for r in pred)
                                                                  for arm in arms})
    write_json(output/"results.json",dict(seed=SEED, old_test_previously_examined=True, eeg_selection=selection,
               source="https://doi.org/10.6084/m9.figshare.14562090.v1", model_sha=bundle["model_id"],
               reproduced_old_test_max_error=max_error, feature_names=FEATURES,
               diagnostics={key:diagnostics(part) for key,part in (("all",rows),("development",dev),("previous_holdout",test))},
               nested_selection=chosen, development=summarize(nested,"nested development, participant-disjoint predictions"),
               test=summarize(h_preds,"old test participants, previously reported in original study"),
               weights=weights))
    with (output/"predictions.csv").open("w",encoding="utf8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["split","person","trial","rating","label",*arms,"selected"])
        writer.writeheader()
        for name,part in (("nested_development",nested),("previous_holdout",h_preds)):
            writer.writerows(dict(split=name,**r) for r in part)
    return selection


if __name__ == "__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cache",type=Path)
    ap.add_argument("output",type=Path)
    args=ap.parse_args()
    print("Selected EEG arm:",run(args.cache,args.output))
