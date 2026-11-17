"""One prediction per person for Nencki-Symfonia's pre-session subjective stress."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from benchmark_lemon_bipolar import SEED, auc, fit, load_models, logits, metrics, segment_features, sigmoid


def predict(x, rating, train, test):
    use = train & (rating!=3)
    label = (rating[use]>3).astype(int)
    weights = 1/(2*np.bincount(label)[label])
    model = fit(x[use],label,weights,l2=.1)
    return sigmoid(logits(model,x[test]))


def evaluate(scores, rating, folds):
    folds_out = []
    binary = rating!=3
    for fold in np.unique(folds):
        test = (folds==fold)&binary
        folds_out.append(metrics((rating[test]>3).astype(int),scores[test]))
    return dict(mean_fold_auc_ba=np.mean(folds_out,axis=0).tolist(),
                fold_auc_ba=np.asarray(folds_out).tolist(),
                pooled_auc_ba=metrics((rating[binary]>3).astype(int),scores[binary]).tolist(),
                continuous_spearman=float(spearmanr(rating,scores).statistic))


def cv(x,rating,folds):
    result = np.empty(len(x))
    for fold in np.unique(folds):
        test = folds==fold
        result[test] = predict(x,rating,~test,test)
    return result


def main(out):
    audit = json.loads((out/'fetch_audit.json').read_text())
    label = json.loads((out/'stress.json').read_text())
    old = load_models()['stress']['parameters']
    participants, matrix, frozen = [], [], []
    qc = {}
    for person, status in sorted(audit.items()):
        qc[person] = dict(status=status)
        if status!='ok':
            continue
        with np.load(out/'raw_pairs_dc_corrected'/f'{person}.npz',allow_pickle=False) as data:
            epochs = [segment_features(pair) for pair in data['eeg_uv']]
            qc[person]['provenance'] = json.loads(str(data['provenance']))
        qc[person]['usable_by_epoch'] = [len(e) for e in epochs]
        usable = [e for e in epochs if len(e)]
        if len(usable)<3:
            qc[person]['status'] = 'insufficient_clean_epochs'
            continue
        participants.append(person)
        matrix.append(np.median([np.median(e,axis=0) for e in usable],axis=0))
        frozen.append(np.mean(sigmoid(logits(old,np.concatenate(usable)[:,:3]))))
    x = np.asarray(matrix)
    rating = np.array([label[p] for p in participants])
    group = np.sign(rating-3)
    rng = np.random.default_rng(SEED)
    folds = np.full(len(rating),-1)
    nfolds = min(5,int((group==-1).sum()),int((group==1).sum()))
    if nfolds<2:
        raise ValueError('Insufficient usable participants across stress levels')
    for g in np.unique(group):
        ids = rng.permutation(np.flatnonzero(group==g))
        folds[ids] = np.arange(len(ids))%nfolds
    scores = dict(frozen=np.array(frozen),three=cv(x[:,:3],rating,folds),expanded=cv(x,rating,folds))
    counts = dict(all=len(audit),has_eeg=sum(v=='ok' for v in audit.values()),eligible=len(x),
                  low=int((group<0).sum()),middle=int((group==0).sum()),high=int((group>0).sum()))
    summary = {k:evaluate(v,rating,folds) for k,v in scores.items()}
    # Preserve each test fold's class composition, shuffle participants, repeat the full training.
    columns = {'three':x[:,:3], 'expanded':x}
    hits = dict.fromkeys(columns,0)
    permutations = 499
    binary = rating!=3
    for _ in range(permutations):
        shuffled = rating.copy()
        for f in range(nfolds):
            ix = np.flatnonzero((folds==f)&binary)
            shuffled[ix] = rng.permutation(rating[ix])
        masks = [(folds==f)&binary for f in range(nfolds)]
        for key,features in columns.items():
            prob = cv(features,shuffled,folds)
            score = np.mean([auc(shuffled[mask]>3,prob[mask]) for mask in masks])
            if score>=summary[key]['mean_fold_auc_ba'][0]-1e-12:
                hits[key] += 1
    probabilities = {k:(1+h)/(1+permutations) for k,h in hits.items()}
    result = dict(source='OpenNeuro ds004621 v1.0.4; pre-session stress 1-5; eyes-open first run',
                  n=counts,fold_count=nfolds,models=summary,permutations=permutations,
                  expanded_one_sided_fold_auc_p=probabilities['expanded'],
                  one_sided_fold_auc_p=probabilities)
    for name,body in [('source_qc',qc),('evaluation',result)]:
        (out/f'{name}.json').write_text(json.dumps(body,indent=2,allow_nan=False),encoding='utf8')
    with (out/'predictions.csv').open('w',encoding='utf8',newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['person','self_rating','outer_fold',*scores])
        writer.writerows(zip(participants,rating,folds,*scores.values()))
    print(json.dumps(result))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    main(parser.parse_args().out)
