"""Replay the frozen holdout scores and check raw-signal and temporal invariants."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter,resample_poly,sosfiltfilt
from scipy.special import expit
from validate_wanderim_frontal import behavior, clean_windows, spectrum, predict, margin, train, residual, summarize
from ningsi.models.logistic import FEATURE_NAMES, features_from_window
from ningsi.signal.bands import relative_band_powers
from ningsi.signal.spectrum import welch_psd


def previous_gate(samples):
    x=resample_poly(samples,1,2,axis=1)
    x=sosfiltfilt(butter(4,(.5,40),btype='bandpass',fs=250,output='sos'),x)
    for end in (1250,1500,1750):
        seg=x[:,end-1000:end]
        if np.all((np.ptp(seg,axis=1)>=1)&(np.ptp(seg,axis=1)<=180)) and np.max(np.abs(np.diff(seg,axis=1)))<=65:
            return True
    return False


def project_feature_comparison(out, rows, metadata, predictions):
    common = [r for r in rows if r['behavior'] is not None and r['eeg'] is not None and r['label'] is not None]
    features = []
    for row in common:
        path = out/'source'/('p'+row['person'])/f"{row['probe']-1:02}.npy"
        vectors = []
        for pair in clean_windows(np.load(path, allow_pickle=False), 500):
            psd = welch_psd(pair[0]-pair[1], 250)
            relative = relative_band_powers(psd.freqs, psd.psd)
            values = features_from_window(SimpleNamespace(rel=relative))
            vectors.append([values[name] for name in FEATURE_NAMES])
        assert len(vectors) == row['valid_chunks']
        features.append(np.mean(vectors, axis=0))
    x = np.asarray(features)
    b = np.asarray([r['behavior'] for r in common])
    y = np.asarray([r['label'] for r in common])
    p = np.asarray([r['person'] for r in common])
    scored = []
    original = {(r['person'],int(r['probe'])):r for r in predictions}
    for fold in metadata['folds']:
        tr = np.isin(p, fold['train'])
        te = np.isin(p, fold['test'])
        assert not np.any(tr & te)
        base = train(b[tr], y[tr], p[tr])
        eeg = train(x[tr], y[tr], p[tr])
        added = residual(x[tr], y[tr], p[tr], margin(predict(base, b[tr])))
        z = (x[te]-added['mean']) / added['scale']
        base_scores = predict(base, b[te])
        alone_scores = predict(eeg, x[te])
        fused = expit(margin(base_scores) + .25*(added['bias']+z@added['weights']))
        for k,j in enumerate(np.flatnonzero(te)):
            item = common[j]
            primary = original[item['person'], item['probe']]
            assert abs(float(primary['behavior'])-base_scores[k]) < 1e-9
            scored.append(dict(person=item['person'], probe=item['probe'], label=item['label'], split=fold['split'],
                               scores=dict(project_six_only=alone_scores[k], project_six_quarter=fused[k],
                                           behavior=base_scores[k], candidate=float(primary['candidate']))))
    with (out/'project_six_predictions.csv').open('w',newline='',encoding='utf8') as fh:
        writer = csv.DictWriter(fh,fieldnames=['person','probe','label','split',*scored[0]['scores']])
        writer.writeheader()
        for row in scored:
            writer.writerow({**{k:v for k,v in row.items() if k!='scores'}, **row['scores']})
    result = {}
    for split in ['development','holdout']:
        chosen = [r for r in scored if (r['split']=='holdout')==(split=='holdout')]
        result[split] = summarize(chosen, reference='project_six_quarter')
    return result


def verify(out):
    source=out/'source'
    rows=json.loads((out/'features.json').read_text())
    stats=json.loads((out/'results.json').read_text())
    bundle=json.loads((out/'candidate.json').read_text())
    people=stats['holdout']['people']
    grouped=set()
    c=0
    old_errors=0
    checked=0
    max_delta=0.
    for info in json.loads((out/'synchronization.json').read_text())['included']:
        data=loadmat(source/info['behavior'])
        t=data['test_res']
        probes=data['probe_res']
        grouped.add(hashlib.sha256(t.tobytes()+probes.tobytes()).hexdigest())
        first=np.load(source/('p'+info['person'])/'00.npy').astype(float)
        shifted=first+np.array([[10000.],[-19000.]])
        old_errors += previous_gate(first)!=previous_gate(shifted)
        before=spectrum(first,500)
        after=spectrum(shifted,500)
        assert (before is None)==(after is None)
        if before is not None:
            checked+=1
            assert before[2]==after[2]
            max_delta=max(max_delta,float(np.max(np.abs(before[0]-after[0]))),float(np.max(np.abs(before[1]-after[1]))))
        for i,q in enumerate(probes):
            a=behavior(t,q,i)
            altered=t.copy()
            altered[altered[:,7]>=q[2],9:12]=1729
            other=q.copy()
            other[6:]=998
            b=behavior(altered,other,i)
            assert (a is None and b is None) or np.array_equal(a,b)
            c+=1
    assert max_delta<1e-8 and len(grouped)==26 and c==1560
    table=list(csv.DictReader((out/'predictions.csv').open(encoding='utf8')))
    test={(r['person'],int(r['probe'])):r for r in table if r['split']=='holdout'}
    errors=[]
    for row in rows:
        key=(row['person'],row['probe'])
        if key not in test:
            continue
        b=np.array([row['behavior']])
        value=predict(bundle['weights']['behavior'],b)[0]
        if bundle['chosen']!='behavior':
            name,factor=bundle['chosen'].split('_')
            model=bundle['weights'][name]
            z=(np.array(row[name])-model['mean'])/model['scale']
            value=expit(margin(value)+float(factor)*(model['bias']+z@model['weights']))
        errors.append(abs(value-float(test[key]['candidate'])))
    assert len(errors)==len(test) and max(errors)<1e-10
    eligible=[r for r in rows if r['behavior'] is not None and r['label'] is not None]
    hold=np.isin([r['person'] for r in eligible],bundle['holdout'])
    bx=np.array([r['behavior'] for r in eligible]); y=np.array([r['label'] for r in eligible])
    p=np.array([r['person'] for r in eligible])
    model=train(bx[~hold],y[~hold],p[~hold])
    scores=predict(model,bx[hold])
    fallback=[]
    for row,val in zip([r for r in eligible if r['person'] in bundle['holdout']],scores):
        fallback.append(dict(person=row['person'],label=row['label'],scores={'behavior':float(val)}))
    matched = []
    compared_probes = [r for r in eligible if r['person'] in bundle['holdout']]
    base = predict(bundle['weights']['behavior'], bx[hold])
    for row, score in zip(compared_probes, base):
        key = (row['person'], row['probe'])
        combined = float(test[key]['candidate']) if key in test else float(score)
        matched.append(dict(person=row['person'], label=row['label'],
                            scores={'behavior':float(score), 'fusion_or_fallback':combined}))
    previous = {}
    for split in ['development','holdout']:
        paired = [dict(person=r['person'], label=int(r['label']),
                       scores={'old_quarter':float(r['old_0.25']), 'candidate':float(r['candidate'])})
                  for r in table if (r['split']=='holdout')==(split=='holdout')]
        previous[split] = summarize(paired, reference='old_quarter')['scores']['candidate']
    report=dict(causal_trials_verified=c,distinct_behavior_subject_hashes=len(grouped),
                raw_epoch_sha256_unique=stats['unique_epoch_hashes'],
                DC_offset_test=dict(participants=26,before_correction_changed_gates=old_errors,after_correction_changed_gates=0,
                                    accepted_pairs_tested_for_feature_invariance=checked,max_feature_change=max_delta,
                                    added_microvolts=[10000,-19000]),
                as_exported_test_rows=len(errors),as_exported_max_score_error=max(errors),
                all_behavior_holdout=summarize(fallback),
                all_holdout_using_behavior_on_qc_failure=summarize(matched),
                selected_vs_fixed_old_quarter=previous,
                project_six_matched_window=project_feature_comparison(out, rows, stats, table))
    (out/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    verify(parser.parse_args().output)
