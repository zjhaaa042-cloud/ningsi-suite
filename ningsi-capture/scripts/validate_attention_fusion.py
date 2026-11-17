"""Compare prospective EEG and SART behavior on the same held-out ATTLAPSE probes."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from attlapse_artifact_diagnostic import cross_fit, inputs, paired_diff
from attlapse_data import PROBE_TRIALS
from attlapse_two_channel_eval import SEED, describe


def behavior_before_probe(trials, next_trial):
    # The questionnaire precedes the stimulus associated with its behavior row.
    # Eighteen *completed* trials give two full 1:9 No-Go cycles in the source task.
    past = [trials[i] for i in range(next_trial - 18, next_trial)]
    go = [row for row in past if row["trial_type"] == "1"]
    nogo = [row for row in past if row["trial_type"] == "2"]
    rt = [float(r["reaction_time"]) for r in go if r["is_response"] == "1"]
    if len(go) + len(nogo) != 18 or not go or not nogo or len(rt) < 3:
        return np.full(4, np.nan)
    if not np.isfinite(rt).all() or min(rt) <= 0:
        return np.full(4, np.nan)
    return [np.median(rt), np.std(rt, ddof=1) / np.mean(rt),
            sum(r["is_response"] == "0" for r in go) / len(go),
            sum(r["is_response"] == "1" for r in nogo) / len(nogo)]


def evaluate(root):
    bench = root / "two-channel-benchmark"
    feature_file = bench / "features250.jsonl"
    split_file = bench / "evaluation250" / "splits.json"
    splits = json.loads(split_file.read_text(encoding="utf8"))["outer"]
    keys, groups, y, _, spectra, rate = inputs(feature_file)
    if rate != 250:
        raise ValueError("Expected 250 Hz EEG")
    tables = {}
    inputs_sha = {}
    behavior = []
    time = []
    for person, probe_id in keys:
        if person not in tables:
            path = root / "labels" / person / f"{person}_task-sart_beh.tsv"
            with path.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream, delimiter="\t"))
            tables[person] = {int(row["trial_number"]): row for row in rows}
            if len(tables[person]) != len(rows):
                raise ValueError(f"Duplicate trials in {person}")
            inputs_sha[person] = hashlib.sha256(path.read_bytes()).hexdigest()
        ordinal = int(probe_id.removeprefix("probe-"))
        behavior.append(behavior_before_probe(tables[person], PROBE_TRIALS[ordinal-1]))
        time.append(ordinal)
    # Restrict every fit and score to identical rows, including the time-only baseline.
    time = np.asarray(time, dtype=float)[:, None]
    behavior = np.asarray(behavior)
    eligible = np.isfinite(behavior).all(axis=1) & np.isfinite(spectra).all(axis=1)
    designs = {"time": time, "behavior": behavior, "eeg": spectra,
               "time_behavior": np.column_stack((time, behavior)),
               "time_behavior_eeg": np.column_stack((time, behavior, spectra))}
    scores, report = {}, {}
    people = np.unique(groups)
    bootstrap = np.random.default_rng(SEED).multinomial(
        len(people), np.full(len(people), 1 / len(people)), size=2000)
    for name, x in designs.items():
        features = np.where(eligible[:, None], x, np.nan)
        scores[name], _ = cross_fit(features, groups, y, splits)
        metrics, _ = describe(y, scores[name], groups, bootstrap)
        fields = ("macro_auc", "balanced_accuracy", "coverage_adjusted_macro_auc",
                  "sensitivity", "specificity", "both_classes_scored_participants", "confusion")
        report[name] = {field: metrics[field] for field in fields}
        report[name]["macro_auc_ci95"] = metrics["ci95"]["macro_auc"]
    return {
        "source": "ATTLAPSE v2.0.1 healthy controls, DOI 10.5281/zenodo.23592484",
        "design": "Exploratory reanalysis of one existing cohort; same five participant-held-out folds. "
                  "Fixed lambda=0.1 logistic model, balanced training weights and train-only normalization. "
                  "Threshold 0.5. No tuning on newly computed results.",
        "target": "on-task vs deliberate/spontaneous mind wandering at a thought probe",
        "behavior": "previous 18 completed trials: median correct Go RT, Go RT CV, Go omissions, No-Go commissions",
        "time": "formal probe ordinal; EEG from pre-probe [-18,-2] s; no probe answer is an input",
        "participants": len(people), "total_probes": len(keys), "common_probes": int(eligible.sum()),
        "coverage": float(eligible.mean()), "results": report,
        "added_eeg_over_time_and_behavior": paired_diff(
            y, groups, scores["time_behavior_eeg"], scores["time_behavior"], bootstrap),
        "split_sha256": hashlib.sha256(split_file.read_bytes()).hexdigest(),
        "feature_sha256": hashlib.sha256(feature_file.read_bytes()).hexdigest(),
        "behavior_sha256": inputs_sha,
        "scope": "Source SART protocol only; transfer to shorter local trials or target hardware unverified.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf8")
