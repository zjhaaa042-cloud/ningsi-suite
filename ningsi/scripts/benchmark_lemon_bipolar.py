"""Paired raw Fp1/Fp2 validation against the LEMON two-year perceived-stress questionnaire."""
import argparse
import csv
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from compare_frontal_affect import filtered_windows
from ningsi.models.affect import feature_vector, load_models
from ningsi.models.human import fit, logits, sigmoid
from ningsi.signal.bands import band_powers, relative_band_powers
from ningsi.signal.spectrum import welch_psd
from ningsi.signal.window import analyze_window


SEED = 151126
BANDS = ('theta','alpha','beta')


def segment_features(condition):
    current = []
    for pair in filtered_windows(condition,250,ends=[1500,2000,2500,3000]):
        window = analyze_window(pair,250)
        if not window.usable:
            continue
        old = feature_vector(window)
        extra = []
        alpha = []
        for lead in (pair[0],pair[1],pair[0]-pair[1],(pair[0]+pair[1])/2):
            psd = welch_psd(lead,250)
            rel = relative_band_powers(psd.freqs,psd.psd)
            absolute = band_powers(psd.freqs,psd.psd)
            if min(absolute[b] for b in BANDS)<=0:
                raise ValueError('Invalid PSD on a quality-approved window')
            extra.extend(np.log([rel[b] for b in BANDS]))
            alpha.append(np.log(absolute['alpha']))
        current.append(np.r_[old,extra,alpha[0]-alpha[1],alpha[2],alpha[3]])
    return np.asarray(current)


def subject_features(eeg):
    segments = [segment_features(condition) for condition in eeg]
    counts = [len(rows) for rows in segments]
    if min(counts)==0:
        return counts,None
    old_model = load_models()['stress']['parameters']
    frozen = float(np.mean(sigmoid(logits(old_model,np.concatenate(segments)[:,:3]))))
    eo,ec = [np.median(rows,axis=0) for rows in segments]
    return counts, dict(frozen=frozen,three=((eo[:3]+ec[:3])/2).tolist(),expanded=np.r_[(eo+ec)/2,ec-eo].tolist(),
                        eo_alpha=eo[1],ec_alpha=ec[1])


def auc(y,p):
    positive = y==1
    n = positive.sum()
    if 0<n<len(y):
        return float((rankdata(p)[positive].sum()-n*(n+1)/2)/(n*(len(y)-n)))
    return np.nan


def metrics(y,p):
    return np.array([auc(y,p), ((p[y==1]>=.5).mean()+(p[y==0]<.5).mean())/2])


def score_rows(train,test,keys):
    y = np.array([int(r['psq']>=50) for r in train])
    w = 1/(2*np.bincount(y)[y])
    outputs = [dict(person=r['person'],label=int(r['psq']>=50),psq=r['psq'],frozen=r['frozen']) for r in test]
    models = {}
    for key in keys:
        x = np.array([r[key] for r in train])
        model = fit(x,y,w,l2=.1)
        score = sigmoid(logits(model,np.array([r[key] for r in test])))
        for r,s in zip(outputs,score):
            r[key] = float(s)
        models[key] = model
    return outputs,models


def evaluate(rows):
    names = ('frozen','three','expanded','demographic')
    y = np.array([r['label'] for r in rows]); stress = np.array([r['psq'] for r in rows])
    scores = {k:np.array([r[k] for r in rows]) for k in names}
    # One row per participant, resample people together for every method.
    rng = np.random.default_rng(SEED)
    draws = rng.integers(len(y),size=(2500,len(y)))
    draws = [idx for idx in draws if np.unique(y[idx]).size==2]
    bootstrap = {k:np.array([metrics(y[idx],s[idx]) for idx in draws]) for k,s in scores.items()}
    point = {k:metrics(y,s) for k,s in scores.items()}
    out = dict(n=len(y),positive=int(y.sum()),models={},deltas={})
    for k,s in scores.items():
        out['models'][k] = dict(auroc=point[k][0],balanced_accuracy=point[k][1],
                                 ci95_auroc_ba=np.percentile(bootstrap[k],[2.5,97.5],axis=0).T.tolist(),
                                 spearman_to_continuous_psq=float(spearmanr(stress,s).statistic))
    for other in ('frozen','three','demographic'):
        out['deltas']['expanded_minus_'+other] = {'auroc_ba':(point['expanded']-point[other]).tolist(),
            'ci95_auroc_ba':np.percentile(bootstrap['expanded']-bootstrap[other],[2.5,97.5],axis=0).T.tolist()}
    return out


def main(out):
    audit = json.loads((out/'fetch_audit.json').read_text(encoding='utf8'))
    with (out/'PSQ.csv').open(encoding='utf-8-sig',newline='') as f:
        psq = {r['']:float(r['PSQ_OverallScore']) for r in csv.DictReader(f) if r['PSQ_OverallScore'].strip() not in ('','NA','NaN')}
    # Pandas-style metadata has two unnamed columns. Read only the ID, age bin and sex fields.
    with (out/'demography.csv').open(encoding='utf-8-sig',newline='') as f:
        tab = csv.reader(f)
        hdr = next(tab)
        demo = {r[0]:[float(np.mean([int(x) for x in re.findall(r'\d+',r[hdr.index('Age')])])),float(r[hdr.index('Gender_ 1=female_2=male')])]
                for r in tab if r[0] in psq}
    shuffled = np.random.default_rng(SEED).permutation(sorted(psq))
    hold = set(shuffled[:int(np.ceil(len(shuffled)/4))]); fold_ids = np.array_split(shuffled[int(np.ceil(len(shuffled)/4)):],5)
    eligible,log = [],{}
    for person,reason in sorted(audit.items()):
        if reason!='ok':
            log[person] = dict(status=reason)
            continue
        path=out/'raw_pairs_dc_corrected'/f'{person}.npz'
        with np.load(path,allow_pickle=False) as data:
            counts,record = subject_features(data['eeg_uv'])
            prov = json.loads(str(data['provenance']))
        log[person] = dict(status='scored' if record else 'qc',accepted_eo_ec=counts,source=prov)
        if record:
            eligible.append(dict(person=person,psq=psq[person],demographic=demo[person],**record))
    dev = [r for r in eligible if r['person'] not in hold]
    test = [r for r in eligible if r['person'] in hold]
    keys = ['three','expanded','demographic']
    predictions,per_fold = [],[]
    for i,fold in enumerate(fold_ids,1):
        val = [r for r in dev if r['person'] in fold]
        scores,_ = score_rows([r for r in dev if r['person'] not in fold],val,keys)
        predictions.extend(dict(**r,split=f'dev-{i}') for r in scores)
        y = np.array([r['label'] for r in scores])
        per_fold.append(dict(n=len(scores),positive=int(y.sum()),auroc=auc(y,np.array([r['expanded'] for r in scores]))))
    locked,weights = score_rows(dev,test,keys)
    result = dict(source='LEMON original Fp1/Fp2 at FCz reference, doi:10.1038/sdata.2018.308',
                  principal_target='PSQ overall >=50/100, last 2 years; one label per person; no contemporaneous acute-stress labels',
                  source_psq_count=len(psq),raw_available=sum(x=='ok' for x in audit.values()),eligible=len(eligible),
                  source_sha256=hashlib.sha256((out/'PSQ.csv').read_bytes()).hexdigest(),
                  dev_folds=[s.tolist() for s in fold_ids],test_ids=sorted(hold),eligible_test=[r['person'] for r in test],
                  positive_control_fraction_ec_alpha_greater=float(np.mean([r['ec_alpha']>r['eo_alpha'] for r in eligible])),
                  dev=evaluate(predictions),test=evaluate(locked),expanded_per_dev_fold=per_fold,weights=weights)
    predictions.extend(dict(**r,split='locked') for r in locked)
    labels = np.array([r['label'] for r in locked],dtype=bool)
    ranks = rankdata([r['expanded'] for r in locked])
    observed = ranks[labels].sum()
    null = [sum(ranks[i] for i in indexes) for indexes in combinations(range(len(ranks)),sum(labels))]
    result['test']['expanded_exact_one_sided_rank_p'] = sum(x>=observed for x in null)/len(null)
    if len(set(r['person'] for r in predictions))!=len(eligible):
        raise ValueError('Participant leakage or missing out-of-sample scores')
    for name,obj in [('evaluation',result),('source_qc',log)]:
        (out/(name+'.json')).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf8')
    with (out/'predictions.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(predictions[0]));writer.writeheader();writer.writerows(predictions)
    print({k:result[k] for k in ('source_psq_count','raw_available','eligible')},result['test']['models']['expanded'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    main(parser.parse_args().out)
