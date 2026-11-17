"""Common-probe paired AUROC differences for five fixed pipeline contrasts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from attlapse_two_channel_eval import SEED, ci, describe
import numpy as np


BASE = "pipeline/physical/"
TARGET = BASE + "pre18_2/bipolar6/rbf"
ALTERNATIVES = [
    BASE + "pre18_2/bipolar6/logistic",
    BASE + "pre18_2/bilateral15/rbf",
    BASE + "pre18_2/bilateral_variability18/rbf",
    BASE + "pre12_2/bipolar6/rbf",
    BASE + "pre32_2/bipolar6/rbf",
]


def compare(path, replicates):
    scopes = {name: {} for name in [TARGET, *ALTERNATIVES]}
    with path.open(encoding="utf8", newline="") as handle:
        for row in csv.DictReader(handle):
            name = row["evaluation_scope"]
            if name not in scopes:
                continue
            key = (row["participant"], row["probe_id"])
            if key in scopes[name] or not row["chosen"].startswith(name.removeprefix("pipeline/") + "/"):
                raise ValueError(f"Duplicate or mismatched pipeline at {key}")
            scopes[name][key] = (int(row["label"]), int(row["outer_fold"]), float(row["score"]) if row["score"] else np.nan)
    reference = scopes[TARGET]
    keys = sorted(reference)
    y = np.asarray([reference[key][0] for key in keys])
    people = np.asarray([key[0] for key in keys])
    subjects = np.unique(people)
    if len(keys) != 420 or len(subjects) != 28:
        raise ValueError("Missing participants or probes in outer predictions")
    rng = np.random.default_rng(SEED)
    counts = rng.multinomial(len(subjects), np.full(len(subjects), 1 / len(subjects)), size=replicates)
    target_scores = np.asarray([reference[key][2] for key in keys])
    comparisons = []
    for name in ALTERNATIVES:
        rows = scopes[name]
        if rows.keys() != reference.keys() or any(rows[key][:2] != reference[key][:2] for key in keys):
            raise ValueError("Probe, label or outer subject fold does not match")
        other = np.asarray([rows[key][2] for key in keys])
        common = np.isfinite(target_scores) & np.isfinite(other)
        a, a_boot = describe(y, np.where(common, target_scores, np.nan), people, counts)
        b, b_boot = describe(y, np.where(common, other, np.nan), people, counts)
        comparisons.append({"comparator": name, "common_count": a["scored_labelled"],
            "common_fraction": a["probe_coverage"], "common_scored_people": a["scored_participants"],
            "common_people_with_both_classes": a["both_classes_scored_participants"],
            "common_positive": a["positive"], "common_negative": a["negative"],
            "common_within_person_pairs": a["scored_within_person_pairs"],
            "target_auc": a["macro_auc"], "comparator_auc": b["macro_auc"],
            "target_minus_comparator": a["macro_auc"] - b["macro_auc"],
            "paired_ci95": ci(a_boot["macro_auc"] - b_boot["macro_auc"])})
    return {"prediction_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "people": len(subjects), "total_probes": len(keys), "comparisons": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    report = {
        "target_pipeline": TARGET, "direction": "target minus comparator",
        "contrast_selection": "Post-hoc exploratory comparisons. Fixed pipeline scopes, each with its own inner-fold parameter selection.",
        "method": "Restrict to probes scored by both pipelines. Compute unweighted within-person AUROC for each person with both labels, then average people. Paired percentile 95% intervals resample all 28 people as clusters, recompute both means on the same draws of frozen outer predictions, and subtract. No model refitting or multiplicity correction.",
        "rates": {str(rate): compare(args.benchmark / folder / "outer_predictions.csv", args.bootstrap_replicates)
                  for rate, folder in ((512, "evaluation"), (250, "evaluation250"))},
        "seed": SEED, "bootstrap_replicates": args.bootstrap_replicates,
    }
    path = args.benchmark / "paired_candidate_comparisons.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
