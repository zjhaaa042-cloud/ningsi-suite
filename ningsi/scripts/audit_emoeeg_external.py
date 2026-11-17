"""Audit six unseen EmoEEG-MC subject folders and quarantine trials with unresolved identity."""

import argparse
import ast
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from compare_frontal_affect import filtered_windows
from ningsi.models.affect import feature_vector
from ningsi.models.human import logits, sigmoid
from ningsi.models.logistic import roc_auc
from ningsi.signal.window import analyze_window
from train_public_affect import SORTED_CLIPS


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ratings(rows):
    return tuple(tuple(row[f"score_{j}"] for j in range(1, 11)) for row in rows)


def waves(pre, sorted_eeg):
    if pre.shape[0] != 2 or pre.shape[2] != 6000:
        raise ValueError("Unexpected waveform shape")
    mapped = []
    for epoch in pre.transpose(1, 0, 2):
        match = [i for i, candidate in enumerate(sorted_eeg.transpose(1, 0, 2))
                 if np.array_equal(epoch, candidate) and np.ptp(candidate) > 0]
        mapped.append(match[0] if len(match) == 1 else None)
    return mapped


def run(source, cache, artifact, output):
    readme = (cache / "EmoEEG-readme.md").read_text(encoding="utf8")
    versions = json.loads((source / "trees.json").read_text(encoding="utf8"))
    original = json.loads((cache / "evaluation/frontal_affect.json").read_text(encoding="utf8"))["models"]["valence"]
    new = json.loads(artifact.read_text(encoding="utf8"))
    assert max(new["trained_people"]) < 55 and max(original["held_out_people"]) < 55
    known = {}
    for path in (cache / "emoeeg").glob("*.npz"):
        with np.load(path, allow_pickle=False) as data:
            known[path.stem] = ratings(csv.DictReader(io.StringIO(str(data["ratings"])), delimiter="\t"))
    audit, examples = {}, []
    for person in range(55, 61):
        sub = f"sub-{person}"
        folder = source / sub
        events = folder / "events.tsv"
        payload = events.read_bytes()
        entry = f"{sub}/{sub}_events.tsv"
        if hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest() != versions["1.0.3"][entry]:
            raise ValueError("Event hash mismatch")
        bname = f"{sub}/beh/{sub}_task-emotion_beh.tsv"
        rel = next(k for k in versions["1.0.7"] if f"derivatives/{sub}/ses-vid/" in k and k.endswith("_reorder.npy"))
        assert versions["1.0.3"][rel] == versions["1.0.7"][rel]
        assert versions["1.0.3"][bname] == versions["1.0.7"][bname]
        with (folder / "behavior.tsv").open(encoding="utf-8-sig") as f:
            beh = list(csv.DictReader(f, delimiter="\t"))
        ev = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig")), delimiter="\t"))
        clips = readme.split(f"- **sub{person}**:")[-1].split("- **sub")[0].split("## Participants' Behaviour")[0]
        seq = ast.literal_eval(re.findall(r"`(\[.*?\])`", clips)[-1])
        duplicate = [other for other, signature in known.items() if ratings(beh) == signature]
        row = {"all_ten_ratings_equal_to": duplicate, "behavior_rows": len(beh),
               "unique_video_ids": sorted(set(r["video_name"] for r in beh)),
               "video_markers": sum(r["trial_type"] == "vid" for r in ev),
               "expected_video_count":len(seq), "rating_sha256": digest(folder / "behavior.tsv"),
               "event_git_sha1": versions["1.0.3"][entry], "event_sha256": digest(events),
               "waveform_pointer_git_sha1_unchanged":versions["1.0.3"][rel]}
        audit[sub] = row
        if duplicate:
            row["reason"] = "Every one of the ten ratings at every trial matches a previously analyzed person."
            continue
        sorted_eeg, pre = np.load(folder / "fp.npy"), np.load(folder / "fp_ica.npy")
        ix = waves(pre, sorted_eeg)
        identity = [SORTED_CLIPS[i] if i is not None else None for i in ix]
        matches = [i for i, (a,b) in enumerate(zip(identity,seq)) if a == b]
        row.update(reordered_shape=list(sorted_eeg.shape), chronological_shape=list(pre.shape),
                   zero_reordered_trials_1based=[j+1 for j in range(sorted_eeg.shape[1]) if np.ptp(sorted_eeg[:,j])==0],
                   chronological_clips_exact_fingerprint=identity, expected_sequence=seq,
                   matching_documented_1based_positions=[i+1 for i in matches],
                   frontal_file_sha256=digest(folder/"fp.npy"), chronological_file_sha256=digest(folder/"fp_ica.npy"))
        if person == 56:
            row["reason"] = "All 21 chronological clip identities disagree with this person's documented sequence."
            continue
        row["reason"] = "Two EEG sequences agree on 16 of 18 published clip positions, but there is an extra EEG trial, a constant stimulus ID, and no documented video markers."
        if not (len(beh) == 36 and len(seq) == 18 and len(matches) == 16):
            raise ValueError("The conditional sub-55 scenario no longer fits published inputs")
        for pos in matches:
            # Hypothesis only: behavioral rows follow the six-row context pairing from complete sessions.
            start = pos // 3 * 6 + pos % 3
            pair = [float(beh[j]["score_8"]) for j in (start, start+3)]
            if not (min(pair) > 3.5 or max(pair) < 3.5):
                continue
            x = []
            for value in filtered_windows(sorted_eeg[:, ix[pos]], 200):
                win = analyze_window(value,250)
                if win.usable:
                    x.append(feature_vector(win))
            if x:
                scores = [float(np.mean(sigmoid(logits(model["parameters"], np.asarray(x))))) for model in (original, new)]
                examples.append({"position":pos+1,"clip":seq[pos],"behavior_candidates_1based":[start+1,start+4],
                                 "ratings":pair,"conditional_label":int(pair[0]>3.5),"usable_windows":len(x),
                                 "old":scores[0],"refit":scores[1]})
    conditional = {}
    y = np.array([r["conditional_label"] for r in examples])
    for name in ("old","refit"):
        pred = np.array([r[name] for r in examples])
        conditional[name] = {"auc":roc_auc(y,pred), "balanced_accuracy":float(
            .5 * (np.mean(pred[y==1] >= .5) + np.mean(pred[y==0] < .5))) if set(y)=={0,1} else None}
    result = {"official_dataset":"https://openneuro.org/datasets/ds005540/versions/1.0.7",
              "versioned_events":"https://github.com/OpenNeuroDatasets/ds005540/tree/1.0.3",
              "audited_people":list(range(55,61)), "strict_test_people":[], "strict_scored_trials":0,
              "independent_auc":None,"independent_balanced_accuracy":None,"people":audit,
              "sub55_conditional_only":{"assumption":"EEG and ratings came from the same person and six-row paired context structure survived omissions. Context identity is not verifiable.",
                                       "use":"Diagnostic illustration only; excluded from evaluation.", "trials":examples,
                                       "counts":[int(sum(y==c)) for c in (0,1)],"descriptive":conditional},
              "original_file_sha256":digest(cache / "evaluation/frontal_affect.json"),
              "corrected_file_sha256":digest(artifact)}
    (output / "external_audit.json").write_text(json.dumps(result,indent=2,allow_nan=False), encoding="utf8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "cache", "refit_model", "output"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    run(a.source, a.cache, a.refit_model, a.output)
