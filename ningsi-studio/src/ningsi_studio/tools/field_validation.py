"""Audit paired, prospective research scores and participant check-ins without modifying the database."""
from collections import defaultdict
import json
import math
from pathlib import Path
import sqlite3

import numpy as np

from ningsi.models.human import bootstrap_mean
from ningsi.models.logistic import roc_auc


def _metrics(observations):
    by_subject = defaultdict(list)
    for subject, label, score in observations:
        by_subject[subject].append((label, score))
    aucs = []
    for pairs in by_subject.values():
        labels, scores = zip(*pairs)
        if len(set(labels)) == 2:
            aucs.append(roc_auc(labels, scores))
    return {"scored_ratings": len(observations), "scored_people": len(by_subject),
            "people_with_both_labels": len(aucs),
            "within_person_macro_auc": float(np.mean(aucs)) if aucs else None,
            "participant_bootstrap_ci95": bootstrap_mean(aucs, seed=20261115)}


def evaluate(database):
    database = Path(database).resolve(strict=True)
    result = {"scope": "Completed, unaccelerated LSL sessions, monitoring or training phase. "
                       "Operator-verified source and genuine voluntary self-reports are required. "
                       "Matched model scores precede responses by at most four seconds. "
                       "Voluntary check-ins alone cannot establish population-level accuracy.",
              "database_schema_has_checkins": False, "dimensions": {}}
    labels = {"stress": "0–5 lower; 6–10 higher on local 0–10 self-rating",
              "valence": "0–4 unpleasant; 6–10 pleasant; neutral 5 excluded"}
    outcomes = {name: defaultdict(list) for name in labels}
    counts = {name: {"rated": 0, "eligible_binary": 0, "repeated_window": 0,
                     "with_prediction": 0} for name in labels}
    used = {name: set() for name in labels}
    conn = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        result["database_schema_has_checkins"] = bool(conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='state_checkins'").fetchone())
        if result["database_schema_has_checkins"]:
            rows = conn.execute("""
                SELECT s.subject_id, s.id AS session_id, c.stress, c.valence, c.phase, c.t_sec, c.eeg_context
                FROM state_checkins c JOIN sessions s ON c.session_id=s.id
                WHERE c.source_kind='lsl' AND s.status='done' AND s.time_scale=1
                    AND c.phase IN ('monitor','training') ORDER BY c.id
            """)
            for row in rows:
                context = json.loads(row["eeg_context"]) if row["eeg_context"] else {}
                age = row["t_sec"] - context.get("elapsed_sec", -1e9)
                paired = context.get("quality_ok") and context.get("phase") == row["phase"] and 0 <= age <= 4
                for name in labels:
                    value = row[name]
                    if value is None:
                        continue
                    counts[name]["rated"] += 1
                    if name == "valence" and value == 5:
                        continue
                    counts[name]["eligible_binary"] += 1
                    item = context.get("prediction", {}).get(name) or {}
                    score, model_id = item.get("score"), item.get("model_id")
                    if (paired and item.get("available") and model_id and isinstance(score, (float, int))
                            and math.isfinite(score) and 0 <= score <= 1):
                        window = (model_id, row["session_id"], row["phase"], context["elapsed_sec"])
                        if window in used[name]:
                            counts[name]["repeated_window"] += 1
                            continue
                        used[name].add(window)
                        counts[name]["with_prediction"] += 1
                        outcomes[name][model_id].append((row["subject_id"], int(value > 5), score))
    finally:
        conn.close()
    for name in labels:
        counts_for_task = counts[name]
        n = counts_for_task["eligible_binary"] - counts_for_task["repeated_window"]
        result["dimensions"][name] = {
            "target": labels[name], **counts_for_task,
            "coverage": counts_for_task["with_prediction"] / n if n else None,
            "by_frozen_model": {key: _metrics(values) for key, values in outcomes[name].items()}}
    return result
