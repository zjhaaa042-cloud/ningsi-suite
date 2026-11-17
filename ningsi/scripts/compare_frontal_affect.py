"""Fixed, paired five-fold comparison on the original development participants only."""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import lfilter, resample_poly

from ningsi.models.affect import BANDS, feature_vector
from ningsi.models.human import fit, logits, sigmoid, write_json
from ningsi.models.logistic import roc_auc
from ningsi.signal.bands import band_powers, relative_band_powers
from ningsi.signal.preprocess import _biquad
from ningsi.signal.spectrum import welch_psd
from ningsi.signal.window import analyze_window
from train_public_affect import RATE, SEED, SORTED_CLIPS


def filtered_windows(values, rate, ends=None):
    gcd = math.gcd(RATE, rate)
    x = resample_poly(values, RATE // gcd, rate // gcd, axis=1)
    if ends is None:
        ends = np.linspace(6 * RATE, x.shape[1], 5, dtype=int)
    x = np.stack([x[:, end - 6 * RATE:end] for end in ends])
    x -= x.mean(axis=-1, keepdims=True)
    # Same zero-initial-state mirror-padded passes as the runtime conditioner.
    # Verify every usable window's three deployed features against its saved
    # production result before including either candidate.
    for kind, frequency, q in (("highpass", .5, .7071), ("notch", 50., 30.),
                               ("notch", 60., 30.), ("lowpass", 45., .7071)):
        b, a = _biquad(kind, frequency, RATE, q)
        pad = 32
        padded = np.concatenate((2*x[..., :1] - x[..., 1:pad+1][..., ::-1], x,
                                 2*x[..., -1:] - x[..., -pad-1:-1][..., ::-1]), axis=-1)
        forward = lfilter(b, a, padded, axis=-1)
        x = lfilter(b, a, forward[..., ::-1], axis=-1)[..., ::-1][..., pad:-pad]
    return x[..., -4*RATE:]


def expanded_features(pair):
    """Paired, ordered Fp1/Fp2 microvolt inputs after the live 250 Hz filter."""
    window = analyze_window(pair, RATE)
    if not window.usable:
        return None
    features = list(feature_vector(window))
    alpha = []
    for channel in pair:
        spectrum = welch_psd(channel, RATE)
        rel = relative_band_powers(spectrum.freqs, spectrum.psd)
        features.extend(math.log(rel[b]) for b in BANDS)
        alpha.append(band_powers(spectrum.freqs, spectrum.psd)["alpha"])
    features.append(math.log(alpha[1]) - math.log(alpha[0]))
    return features


def load_development(source, dimension, bundle, root):
    held = set(bundle["models"][dimension]["held_out_people"])
    rows = []
    last_emo, last_person = None, None
    for row in source:
        if row["person"] in held or row["label"] is None or not row["windows"]:
            continue
        person, trial = row["person"], row["trial"]
        if dimension == "stress":
            task, visit = trial.rsplit("-", 1)
            file = root / "Data/raw_data" / f"{task}_sub_{person}_trial{visit}.mat"
            values = loadmat(file, variable_names=["Data"])["Data"][[2, 31]]
            rate = 128
        else:
            if last_person != person:
                with np.load(root / "emoeeg" / f"sub-{person:02d}.npz", allow_pickle=False) as src:
                    last_emo = src["channels"]
                last_person = person
            values, rate = last_emo[:, SORTED_CLIPS.index(trial)], 200
        expanded = [v for pair in filtered_windows(values, rate) if (v := expanded_features(pair)) is not None]
        frozen = np.array(row["windows"])
        if len(expanded) != len(frozen) or not np.allclose(np.array(expanded)[:, :3], frozen, rtol=1e-8, atol=1e-8):
            raise ValueError(f"Window/feature parity failed for {person}/{trial}")
        rows.append({**row, "x": np.array(expanded)})
    return rows


def predict_fold(train, val, n_features):
    counts = Counter(r["person"] for r in train)
    x = np.concatenate([r["x"][:, :n_features] for r in train])
    y = np.concatenate([np.repeat(r["label"], len(r["x"])) for r in train])
    w = np.concatenate([np.full(len(r["x"]), 1./(counts[r["person"]]*len(r["x"]))) for r in train])
    for value in (0, 1):
        mask = y == value
        w[mask] /= 2 * w[mask].sum()
    model = fit(x, y, w, l2=.05)
    return [float(np.mean(sigmoid(logits(model, r["x"][:, :n_features])))) for r in val]


def paired_metrics(oof):
    people = sorted({r["person"] for r in oof})
    per_person = []
    for person in people:
        part = [r for r in oof if r["person"] == person]
        labels = [r["label"] for r in part]
        if len(set(labels)) == 2:
            per_person.append([roc_auc(labels, [r[k] for r in part]) for k in ("base", "expanded")])
    auc = np.asarray(per_person)
    rng = np.random.default_rng(SEED)
    boot = auc[rng.integers(len(auc), size=(2000,len(auc)))].mean(axis=1)
    delta_samples = boot[:, 1] - boot[:, 0]
    y = np.asarray([r["label"] for r in oof])
    result = {"people_scored":len(people), "both_class_people": len(auc), "scored_trials":len(oof)}
    counts = []
    for j, key in enumerate(("base", "expanded")):
        p = np.array([r[key] for r in oof])
        pred = p >= .5
        result[key] = {"macro_auc":float(auc[:,j].mean()), "macro_auc_ci95":np.percentile(boot[:,j],[2.5,97.5]).tolist(),
                       "pooled_auc":roc_auc(y,p), "balanced_accuracy":float(.5*(pred[y==1].mean()+(~pred[y==0]).mean()))}
        by_subject = []
        for person in people:
            mask = np.array([r["person"] == person for r in oof])
            by_subject.append([np.sum(mask & (y == 1) & pred), np.sum(mask & (y == 1)),
                               np.sum(mask & (y == 0) & ~pred), np.sum(mask & (y == 0))])
        counts.append(by_subject)
    draw = rng.integers(len(people), size=(2000,len(people)))
    cnt = np.asarray(counts)[:,draw].sum(axis=2)
    bacc = .5*(cnt[:,:,0]/cnt[:,:,1]+cnt[:,:,2]/cnt[:,:,3])
    result["paired_delta"] = {"macro_auc":float((auc[:,1]-auc[:,0]).mean()),
                              "macro_auc_ci95":np.percentile(delta_samples,[2.5,97.5]).tolist(),
                              "balanced_accuracy":result["expanded"]["balanced_accuracy"]-result["base"]["balanced_accuracy"],
                              "balanced_accuracy_ci95":np.percentile(bacc[1]-bacc[0],[2.5,97.5]).tolist()}
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args=parser.parse_args()
    evaluation = args.root / "evaluation"
    bundle=json.loads((evaluation/"frontal_affect.json").read_text(encoding="utf-8"))
    source=json.loads((evaluation/"trials.json").read_text(encoding="utf-8"))
    report = {"protocol":"Fixed, participant-separated five-fold development CV; participant-paired percentile 95% CIs. "
              "Both use identical strictly eligible windows, held-out participants of frozen public test omitted. "
              "Regularization and .5 threshold unchanged. Right minus left log absolute alpha power is a candidate feature.",
              "features_added":["Fp1 log relative theta, alpha, beta","Fp2 log relative theta, alpha, beta",
                                "log absolute alpha Fp2 - log absolute alpha Fp1"]}
    for name in ("stress","valence"):
        rows = load_development(source.pop(name), name, bundle, args.root)
        people = sorted({r["person"] for r in rows})
        folds=np.array_split(np.random.default_rng(SEED).permutation(people),5)
        oof=[]
        for k,ids in enumerate(folds):
            train=[r for r in rows if r["person"] not in ids]
            val=[r for r in rows if r["person"] in ids]
            base = predict_fold(train,val,3)
            new = predict_fold(train,val,10)
            for r, a, b in zip(val,base,new):
                oof.append({"fold":k+1,"person":r["person"],"trial":r["trial"],"label":r["label"],"base":a,"expanded":b})
        report[name] = {**paired_metrics(oof), "eligible_windows":sum(len(r["x"]) for r in rows),
                        "development_fold_ids":[part.tolist() for part in folds]}
        with (evaluation/f"{name}_development_comparison.csv").open("w",newline="",encoding="utf-8") as dest:
            writer=csv.DictWriter(dest,fieldnames=oof[0].keys())
            writer.writeheader()
            writer.writerows(oof)
    write_json(evaluation/"development_comparison.json",report)
    print(json.dumps({k:report[k]["paired_delta"] for k in ("stress","valence")},indent=2))
