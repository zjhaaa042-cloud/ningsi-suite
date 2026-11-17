"""SAM40 raw versus author-filtered EEG under the same stress feature contract."""
import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from benchmark_sam40_stress import folds, paired, train_and_predict
from compare_frontal_affect import filtered_windows
from train_public_affect import RATE, sam_tasks
from ningsi.models.affect import feature_vector
from ningsi.models.human import logits, sigmoid, write_json
from ningsi.signal.window import analyze_window


def spectrum(path, field):
    source = loadmat(path, variable_names=[field])[field]
    assert source.shape == (32, 3200) and np.isfinite(source).all()
    channels = source[[2, 31]]
    samples = []
    for pair in filtered_windows(channels, 128):
        win = analyze_window(pair, RATE)
        samples.append(feature_vector(win).tolist() if win.usable else None)
    fingerprint = sha256(np.ascontiguousarray(channels, dtype="<f8").tobytes()).hexdigest()
    return samples, fingerprint


def load_domains(cache):
    raw_reference = json.loads((cache / "evaluation/trials.json").read_text(encoding="utf8"))["stress"]
    expected = {(r["person"], r["trial"]): r for r in raw_reference}
    output = []
    for path, person, trial, rating, label in sam_tasks(cache / "Data"):
        raw, raw_hash = spectrum(path, "Data")
        clean, clean_hash = spectrum(path.parent.parent / "filtered_data" / path.name, "Clean_data")
        ref = expected[person, trial]
        assert ref["label"] == label and ref["rating"] == rating
        raw_valid = [x for x in raw if x is not None]
        assert len(raw_valid) == len(ref["windows"]) and np.allclose(raw_valid, ref["windows"], atol=1e-8)
        common = [(a, b) for a, b in zip(raw, clean) if a is not None and b is not None]
        output.append(dict(person=person, trial=trial, label=label, raw=raw_valid,
                           filtered=[x for x in clean if x is not None],
                           raw_common=[a for a, _ in common], filtered_common=[b for _, b in common],
                           raw_hash=raw_hash, filtered_hash=clean_hash))
    return output


def as_domain(rows, domain):
    return [dict(person=r["person"], trial=r["trial"], label=r["label"], old=r[domain]) for r in rows]


def predict(params, features):
    if features:
        return float(np.mean(sigmoid(logits(params, np.array(features)))))
    return None


CONDITIONS = {
    "raw": ("raw", "raw"),
    "raw_to_filtered": ("raw", "filtered"),
    "filtered": ("filtered", "filtered"),
    "raw_common": ("raw_common", "raw_common"),
    "raw_to_filtered_common": ("raw_common", "filtered_common"),
    "filtered_common": ("filtered_common", "filtered_common"),
}


def score_fold(train, test):
    assert not {r["person"] for r in train} & {r["person"] for r in test}
    models = {domain: train_and_predict(as_domain(train, domain), [], "legacy")[0]
              for domain in ("raw", "filtered", "raw_common", "filtered_common")}
    scores = []
    for r in test:
        pred = dict(person=r["person"], trial=r["trial"], label=r["label"])
        for name, (train_on, evaluate_on) in CONDITIONS.items():
            pred[name] = predict(models[train_on], r[evaluate_on])
        scores.append(pred)
    return scores


def summarize(rows, preds):
    overview = {}
    for name in CONDITIONS:
        stats = paired(preds, name, name)
        overview[name] = dict(scored_people=stats["people_scored"], both_class_people=stats["both_class_people"],
                              trials=stats["scored_trials"], **stats["base"])
    return dict(trials=len(rows), native=overview,
                windows={k:sum(len(r[k]) for r in rows) for k in ("raw", "filtered", "raw_common")},
                paired={key:paired(preds, a, b) for key,a,b in (
                    ("raw_weights_across_domains","raw","raw_to_filtered"),
                    ("domain_specific_refit","raw","filtered"),
                    ("identical_train_test_windows_frozen","raw_common","raw_to_filtered_common"),
                    ("identical_train_test_windows_refit","raw_common","filtered_common"))})


def run(cache, out):
    out.mkdir(parents=True, exist_ok=True)
    rows = load_domains(cache)
    expected = json.loads((cache / "evaluation/frontal_affect.json").read_text(encoding="utf8"))["models"]["stress"]
    holdout = set(expected["held_out_people"])
    dev = [r for r in rows if r["person"] not in holdout]
    test = [r for r in rows if r["person"] in holdout]
    assignments = folds({r["person"] for r in dev}, 5)
    cv = []
    for val in assignments:
        train = [r for r in dev if r["person"] not in val]
        here = [r for r in dev if r["person"] in val]
        cv.extend(score_fold(train, here))
    test_pred = score_fold(dev, test)
    frozen_error = max(abs(p["raw"] - predict(expected["parameters"], r["raw"]))
                       for r,p in zip(test,test_pred) if r["raw"])
    assert frozen_error < 1e-8
    duplicates = {}
    for key in ("raw_hash", "filtered_hash"):
        frequencies = Counter(r[key] for r in rows)
        duplicates[key] = sum(count - 1 for count in frequencies.values())
    write_json(out / "author_filter_results.json", dict(
        primary_source="https://doi.org/10.6084/m9.figshare.14562090.v1",
        expected_source_md5="318b8b0c4e5a22c6e903534a842ce683",
        control="Authors' 25 s nonlinear filtering changes spectra as well as artifact rejection.",
        folds=assignments, previous_holdout=sorted(holdout), frozen_raw_max_error=frozen_error,
        identical_25_second_two_channel_waveforms_beyond_first=duplicates,
        development=summarize(dev, cv), test=summarize(test,test_pred)))
    with (out / "author_filter_predictions.csv").open("w",encoding="utf8",newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split","person","trial","label",*CONDITIONS])
        writer.writeheader()
        for split, preds in (("development_oof",cv),("previous_holdout",test_pred)):
            writer.writerows(dict(split=split,**p) for p in preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.cache,args.output)
