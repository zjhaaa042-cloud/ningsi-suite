"""Subject-independent SAM40 montage and task-identity controls, for offline diagnosis."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from scipy.signal import butter, sosfiltfilt, welch
from scipy.special import logsumexp
from scipy.stats import rankdata


def descriptors(samples):
    # One row per 25 s trial, four-second windows entirely within that trial.
    # No region-specific re-reference: both montages use original sensor reference.
    segments = np.stack([samples[:, i:i+512] for i in range(128, 3200-512+1, 256)])
    f, psd = welch(segments, fs=128, axis=-1, nperseg=256, noverlap=128, detrend='linear')
    bands = ((4, 8), (8, 13), (13, 20), (20, 30))
    total = psd[:, :, (f >= 2) & (f < 40)].sum(axis=-1)
    power = np.stack([psd[:, :, (f >= lo) & (f < hi)].sum(axis=-1) for lo, hi in bands], axis=-1)
    relative = np.log(np.maximum(power, 1e-15) / np.maximum(total[:, :, None], 1e-15))
    filtered = sosfiltfilt(butter(3, (4, 30), btype='bandpass', fs=128, output='sos'), segments)
    filtered -= filtered.mean(axis=-1, keepdims=True)
    filtered /= np.maximum(np.linalg.norm(filtered, axis=-1, keepdims=True), 1e-10)
    corr = filtered @ filtered.transpose(0, 2, 1)
    out = []
    for channels in ([2, 31], list(range(32))):
        i, j = np.triu_indices(len(channels), k=1)
        spectral = np.median(relative[:, channels], axis=0).ravel()
        spatial = np.median(corr[:, np.array(channels)[i], np.array(channels)[j]], axis=0)
        out.append(np.r_[spectral, spatial])
    return out


def model_predict(x, y, train, test):
    # Use the same class-balanced multinomial loss and fixed C on both montages.
    mean = x[train].mean(axis=0)
    scale = x[train].std(axis=0)
    scale[scale < 1e-10] = 1.
    z = (x[train] - mean) / scale
    classes, codes, counts = np.unique(y[train], return_inverse=True, return_counts=True)
    weights = len(z) / (len(classes) * counts[codes])
    truth = np.eye(len(classes))[codes]
    shape = (z.shape[1] + 1, len(classes))
    def loss(theta):
        w = theta.reshape(shape)
        logits = z @ w[1:] + w[0]
        logp = logits - logsumexp(logits, axis=1, keepdims=True)
        value = -(logp * truth * weights[:, None]).sum() + .5 / .02 * np.square(w[1:]).sum()
        residual = (np.exp(logp) - truth) * weights[:, None]
        grad = np.vstack((residual.sum(axis=0), z.T @ residual + w[1:] / .02))
        return value, grad.ravel()
    opt = minimize(loss, np.zeros(np.prod(shape)), method='L-BFGS-B', jac=True, options={'maxiter':500})
    if not opt.success:
        raise ValueError(opt.message)
    w = opt.x.reshape(shape)
    logits = (x[test] - mean) / scale @ w[1:] + w[0]
    return np.exp(logits - logsumexp(logits, axis=1, keepdims=True)), classes


def binary_auc(y, p):
    positives = y == 1
    n = positives.sum()
    return (rankdata(p)[positives].sum() - n*(n+1)/2) / (n*(len(y)-n))


def estimate(y, p, groups, classes):
    pred = classes[np.argmax(p, axis=1)]
    one_true = y[None, :] == classes[:, None]
    by_person, aucs = [], []
    for person in np.unique(groups):
        use = groups == person
        counts = one_true[:, use].sum(axis=1)
        hits = (one_true[:, use] & (y[use] == pred[use])).sum(axis=1)
        by_person.append(np.c_[hits, counts])
        if len(classes) == 2 and len(np.unique(y[use])) == 2:
            aucs.append(binary_auc(y[use], p[use, 1]))
    rng = np.random.default_rng(11521)
    sampled = np.array(by_person)[rng.integers(len(by_person), size=(4000, len(by_person)))].sum(axis=1)
    sampled = sampled[(sampled[:, :, 1] > 0).all(axis=1)]
    ba = (sampled[:, :, 0] / sampled[:, :, 1]).mean(axis=1)
    all_counts = np.sum(by_person, axis=0)
    result = dict(people=len(by_person), trials=len(y), balanced_accuracy=np.mean(all_counts[:, 0] / all_counts[:, 1]),
                  ba_ci95=np.quantile(ba, [.025,.975]).tolist())
    if len(classes) == 2:
        result.update(pooled_auc=binary_auc(y,p[:,1]), macro_auc=np.mean(aucs), both_label_people=len(aucs))
    return result, ba


def evaluate(root, output):
    raw = root / 'Data' / 'raw_data'
    els = [line.split()[-1] for line in (root/'Data/Coordinates.locs').read_text().splitlines()]
    if len(els) != 32 or (els[2], els[31]) != ('Fp1', 'Fp2'):
        raise ValueError('Unexpected SAM40 electrodes')
    source = json.loads((root/'evaluation/trials.json').read_text(encoding='utf8'))['stress']
    frozen = json.loads((root/'evaluation/frontal_affect.json').read_text(encoding='utf8'))['models']['stress']
    rated = {(r['person'], r['trial']):r for r in source}
    records, front, scalp = [], [], []
    for p in sorted(set(r['person'] for r in source)):
        for task in ('Arithmetic', 'Mirror_image', 'Stroop', 'Relax'):
            for session in range(1,4):
                trial = f'{task}-{session}'
                path = raw / f'{task}_sub_{p}_trial{session}.mat'
                samples = loadmat(path, variable_names=['Data'])['Data']
                if samples.shape != (32, 3200) or not np.isfinite(samples).all():
                    raise ValueError(f'Invalid recording {path}')
                frontal, full = descriptors(samples)
                front.append(frontal)
                scalp.append(full)
                r = rated.get((p, trial))
                records.append((p, trial, task, int(r['label']) if r else -1, bool(r and r['windows'])))
    groups = np.array([r[0] for r in records])
    task = np.array([r[2] for r in records])
    stress = np.array([r[3] for r in records])
    eligible = np.array([r[4] for r in records])
    held = np.isin(groups, frozen['held_out_people'])
    dev = np.unique(groups[~held])
    perm = np.random.default_rng(11521).permutation(dev)
    folds = np.array_split(perm, 5)
    fronts = {'frontal':np.array(front), 'scalp':np.array(scalp)}
    results = {}
    predictions = []
    designs = {'four_tasks': (task, np.ones(len(task), dtype=bool)), 'three_active_tasks': (task, stress>=0),
               'stress_all':(stress,stress>=0), 'stress_old_eligible':(stress,eligible)}
    for name,(y,mask) in designs.items():
        results[name] = {}
        for mode in ('development_oof','original_holdout'):
            use = mask & (~held if mode == 'development_oof' else held)
            result = {}
            bas = {}
            for montage,x in fronts.items():
                scores = np.full((len(y), len(np.unique(y[mask]))), np.nan)
                for people in (folds if mode == 'development_oof' else [frozen['held_out_people']]):
                    test = np.isin(groups,people) & use
                    train = ~np.isin(groups,people) & ~held & mask
                    if set(groups[test]) & set(groups[train]):
                        raise ValueError('Person leakage')
                    scores[test], classes = model_predict(x,y,train,test)
                if not np.isfinite(scores[use]).all():
                    raise ValueError('Missing predictions')
                result[montage], bas[montage] = estimate(y[use],scores[use],groups[use],classes)
                for i in np.where(use)[0]:
                    predictions.append([name,mode,montage,groups[i],records[i][1],y[i],*scores[i]])
            delta = bas['scalp'] - bas['frontal']
            result['scalp_minus_frontal_ba'] = {'estimate':result['scalp']['balanced_accuracy'] - result['frontal']['balanced_accuracy'],
                                                'ci95':np.quantile(delta,[.025,.975]).tolist()}
            results[name][mode] = result
    output.mkdir(exist_ok=True, parents=True)
    with (output/'spatial_control_scores.csv').open('w',encoding='utf8',newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['target','evaluation','montage','subject','trial','label','p_class0','p_class1','p_class2','p_class3'])
        writer.writerows(predictions)
    report = dict(source='SAM40 v1 doi:10.6084/m9.figshare.14562090.v1', held_out=frozen['held_out_people'],
                  development_folds=[p.tolist() for p in folds], results=results,
                  method='One trial per row. Fixed C=.02 balanced logistic regression on median relative log bandpowers and pairwise 4–30 Hz correlation. Class ordering alphabetical/numeric. Strict subset uses trials with at least one old accepted window, but descriptors span the trial. Offline diagnostic; includes artifacts; task identity can follow setting, order or movement. Paired cluster bootstrap over subjects. No parameter selection.')
    (output/'spatial_control.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf8')
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.root,args.out),indent=2))
