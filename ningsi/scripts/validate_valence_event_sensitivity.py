"""Sensitivity of the three-band model to additionally inferred modality rows in four training people."""
import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from compare_frontal_affect import predict_fold
from ningsi.models.human import write_json
from validate_valence_mechanisms import SEED, auc


def score(rows, pred):
    y=np.array([r['label'] for r in rows]); p=np.asarray(pred)
    people=sorted({r['person'] for r in rows})
    counts, aucs=[], []
    for person in people:
        mask=np.array([r['person']==person for r in rows]); ys,ps=y[mask],p[mask]
        counts.append([sum((ys==1)&(ps>=.5)),sum(ys==1),sum((ys==0)&(ps<.5)),sum(ys==0)])
        aucs.append(auc(ys,ps))
    draws=np.random.default_rng(SEED).integers(len(people),size=(4000,len(people)))
    def ba(x):
        return .5*(x[...,0]/x[...,1]+x[...,2]/x[...,3])
    arr=np.array(counts)
    return ({'ba':float(ba(arr.sum(0))),'macro_auc':float(np.nanmean(aucs)),'pooled_auc':float(auc(y,p))},
            {'ba':ba(arr[draws].sum(1)),'macro_auc':np.nanmean(np.array(aucs)[draws],axis=1)})


def main(cache,out):
    inputs=out/'events/additional_video_rows.json'
    m1={int(k):v for k,v in json.loads((out/'video_rows.json').read_text('utf8')).items()}
    extra={int(k):v for k,v in json.loads(inputs.read_text('utf8')).items()}
    assert set(extra)=={9,10,11,20} and not (set(m1)&set(extra))
    baseline=json.loads((out/'refit_corrected_three_bands.json').read_text('utf8'))
    held=set(baseline['held_out_people'])
    assert not held&set(extra)
    source=json.loads((cache/'evaluation/trials.json').read_text('utf8'))['valence']
    assert all(sum(r['person']==person for r in source)==21 for person in range(1,54))
    groups={}; fitted=[]; test=[]; counts=[]
    for mapping in (m1,{**m1,**extra}):
        by_person={}; pos={}; train=[]; current=[]
        for row in source:
            p=row['person']; index=pos.get(p,0); pos[p]=index+1
            if p>52:
                continue
            label=row['label']
            if p in mapping:
                first=index//3*6+index%3
                bi=mapping[p][index]
                assert bi in (first,first+3)
                rating=row['rating'][(first,first+3).index(bi)]
                label=None if rating==3.5 else int(rating>3.5)
            c=by_person.setdefault(p,dict(identifiable=0,scored=0))
            c['identifiable']+=int(label is not None)
            if label is None or not row['windows']:
                continue
            c['scored']+=1
            entry={**row,'label':label,'x':np.asarray(row['windows'])}
            (current if p in held else train).append(entry)
        if test:
            assert [(r['person'],r['trial'],r['label']) for r in current]==[(r['person'],r['trial'],r['label']) for r in test]
        test=current; fitted.append(predict_fold(train,test,3)); counts.append(len(train))
        groups[len(counts)]={p:by_person[p] for p in extra}
    with (out/'corrected_holdout.csv').open(encoding='utf8') as f:
        previous={(int(k['person']),k['trial']):float(k['old']) for k in csv.DictReader(f)}
    error=max(abs(previous[(r['person'],r['trial'])]-v) for r,v in zip(test,fitted[0]))
    assert error<1e-8
    metrics,draws=zip(*(score(test,pred) for pred in fitted))
    delta={k:{'point':metrics[1][k]-metrics[0][k],'ci95':np.percentile(draws[1][k]-draws[0][k],[2.5,97.5]).tolist()} for k in draws[0]}
    path=out/'additional_event_sensitivity.csv'
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.writer(f);w.writerow(['person','trial','label','primary_fit','additional_four_fit'])
        for r,a,b in zip(test,*fitted):
            w.writerow([r['person'],r['trial'],r['label'],a,b])
    result=dict(secondary=True,additional_training_subjects=sorted(extra),training_trial_counts=counts,
                test_people=sorted(held),test_trials=len(test),subject_coverage=groups,
                primary_score_parity=error,primary_model_id=baseline['model_id'],source_sha256=sha256(inputs.read_bytes()).hexdigest(),
                primary=metrics[0],sensitivity=metrics[1],paired_delta=delta)
    write_json(out/'additional_event_sensitivity.json',result)
    print(json.dumps(result['paired_delta']))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('cache',type=Path); ap.add_argument('out',type=Path)
    args=ap.parse_args();main(args.cache,args.out)
