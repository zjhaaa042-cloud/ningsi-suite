"""Measure two-frontal EEG stress ranking after canceling within-person round effects."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from itertools import combinations, product
from pathlib import Path
import sys

import numpy as np


TASKS = ('Stroop','Mirror_image','Arithmetic')
EDGES = ((4,8),(8,13),(13,30))
METHODS = ('constant','legacy','task_order','optimized','residual_eeg','task_plus_eeg')
SEED = 676125


def read_json(path):
    return json.loads(path.read_text(encoding='utf8'))


def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as fp:
        return list(csv.DictReader(fp))


def write_csv(path,records):
    with path.open('w',encoding='utf8',newline='') as fp:
        writer=csv.DictWriter(fp,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def process(path):
    from scipy.io import loadmat

    data=loadmat(path,variable_names=['Data'])['Data']
    if data.shape!=(32,3200):
        raise ValueError(str(path))
    return segment_features(data[[2,31]].astype(float),128)


def segment_features(x,rate):
    from scipy.signal import resample_poly, welch
    from compare_frontal_affect import filtered_windows
    from ningsi.models.affect import feature_vector
    from ningsi.signal.window import analyze_window
    from math import gcd

    assert x.shape==(2,rate*25)
    common=gcd(rate,250)
    x=resample_poly(x-x.mean(axis=-1,keepdims=True),250//common,rate//common,axis=-1,padtype='line')
    ends=np.linspace(6*250,25*250,5,dtype=int)
    feats=[]
    for pair in filtered_windows(x,250,ends):
        win=analyze_window(pair,250)
        if not win.usable:
            continue
        channels=np.vstack([pair,pair.mean(axis=0),pair[0]-pair[1]])
        freq,psd=welch(channels,fs=250,nperseg=250,noverlap=125,axis=-1)
        power=np.stack([psd[:,(freq>=a)&(freq<b)].sum(axis=1) for a,b in EDGES],axis=1)
        total=psd[:,(freq>=1)&(freq<40)].sum(axis=1,keepdims=True)
        if np.any(power<=0) or np.any(total<=0):
            raise ValueError('Nonpositive power')
        feats.append(np.r_[feature_vector(win),np.log(power/total).ravel(),
                          np.log(power[0]/power[1]),np.corrcoef(pair)[0,1]])
    return (np.median(feats,axis=0).tolist() if feats else []),len(feats)


def prepare(repo,out):
    old_dir=repo/'var/state-optimization-20261115/stress'
    cache=repo/'var/affect-public'
    sys.path.insert(0,str(next(cache.glob('xlrd*.whl'))))
    import xlrd
    sheet=xlrd.open_workbook(str(cache/'Data/scales.xls')).sheet_by_index(0)
    assert sheet.row_values(1)[1:]==['Maths','Symmetry','Stroop']*3
    coords=[line.split()[-1] for line in (cache/'Data/Coordinates.locs').read_text().splitlines()]
    assert coords[2]=='Fp1' and coords[31]=='Fp2'
    frozen=read_json(cache/'evaluation/frontal_affect.json')['models']['stress']
    held=frozen['held_out_people']
    folds=[row['test'] for row in read_json(old_dir/'results.json')['nested_selection']]
    old={(int(r['person']),r['trial']):r for r in read_csv(old_dir/'predictions.csv')}
    records=[]
    for i in range(2,sheet.nrows):
        row=sheet.row_values(i)
        person=int(row[0])
        for rep in (1,2,3):
            for k,task in enumerate(reversed(TASKS)):
                label=int(row[1+3*(rep-1)+k])
                src=cache/'Data/raw_data'/f'{task}_sub_{person}_trial{rep}.mat'
                peer=old[person,f'{task}-{rep}']
                assert int(float(peer['rating']))==label
                feat,n=process(src)
                records.append(dict(person=person,task=task,round=rep,rating=label,
                                    old=float(peer['legacy']) if peer['legacy'] else None,
                                    x=feat,n_clean=n,source=src.name,
                                    sha256=hashlib.sha256(src.read_bytes()).hexdigest()))
    cache_file=dict(rows=records,folds=folds,held=held,legacy_id=frozen['model_id'],
                    scales_sha256=hashlib.sha256((cache/'Data/scales.xls').read_bytes()).hexdigest(),
                    coords_sha256=hashlib.sha256((cache/'Data/Coordinates.locs').read_bytes()).hexdigest())
    dump(out/'features.json',cache_file)
    return cache_file


def make_contrasts(rows):
    lookup={(r['person'],r['round'],r['task']):r for r in rows if r['old'] is not None and r['x']}
    result=[]
    for p in sorted({r['person'] for r in rows}):
        for a,b in combinations(TASKS,2):
            pairs={}
            for v in (1,2,3):
                if (p,v,a) not in lookup or (p,v,b) not in lookup:
                    continue
                l,r=lookup[p,v,a],lookup[p,v,b]
                pairs[v]=(l,r)
                result.append(record(p,[l,r],[1,-1],'within_round'))
            for earlier,later in combinations(pairs,2):
                result.append(record(p,[*pairs[later],*pairs[earlier]],[1,-1,-1,1],'double_difference'))
        for task in TASKS:
            for earlier,later in combinations((1,2,3),2):
                if (p,earlier,task) in lookup and (p,later,task) in lookup:
                    result.append(record(p,[lookup[p,later,task],lookup[p,earlier,task]],[1,-1],'same_task_across_rounds'))
    return result


def record(person,items,sign,kind):
    return dict(person=person,kind=kind,keys=[(r['round'],r['task']) for r in items],
                components=[f"{r['task']}-{r['round']}" for r in items],sign=sign,
                d=float(np.array(sign)@np.array([r['rating'] for r in items])),
                old=float(np.array(sign)@np.array([r['old'] for r in items])),
                x=np.array(sign)@np.stack([r['x'] for r in items]),
                binary_contrast=(len(items)==2 and (items[0]['rating']>=6)!=(items[1]['rating']>=6)))


def regress(X,y,people):
    # No intercept: swapping tasks must reverse the sign of the prediction.
    counts=Counter(people)
    w=np.array([1/(len(counts)*counts[p]) for p in people])
    scale=np.sqrt(w@(X**2))
    scale[scale<1e-8]=1
    z=X/scale
    coef=np.linalg.solve((z.T*w)@z+np.eye(X.shape[1]),(z.T*w)@y)/scale
    return coef


def fold(rows,comparisons,train_ids,test_ids,number):
    assert not set(train_ids)&set(test_ids)
    train_raw=[r for r in rows if r['person'] in train_ids]
    e_means={}
    y_means={}
    for rep,task in product((1,2,3),TASKS):
        cell=[r for r in train_raw if r['round']==rep and r['task']==task]
        y_means[rep,task]=np.mean([r['rating'] for r in cell])
        e_means[rep,task]=np.mean([r['x'] for r in cell if r['x'] and r['old'] is not None],axis=0)
    models={}
    predictions=[]
    for kind in ('within_round','double_difference','same_task_across_rounds'):
        fit_rows=[r for r in comparisons if r['person'] in train_ids and r['kind']==kind]
        tests=[r for r in comparisons if r['person'] in test_ids and r['kind']==kind]
        def transform(r):
            e=np.array(r['sign'])@np.stack([e_means[k] for k in r['keys']])
            y=np.array(r['sign'])@np.array([y_means[k] for k in r['keys']])
            return r['x']-e,y
        # Use continuous differences, including exact ties, during training.
        people=[r['person'] for r in fit_rows]
        X=np.stack([r['x'] for r in fit_rows])
        y=np.array([r['d'] for r in fit_rows])
        residuals=[transform(r) for r in fit_rows]
        beta=regress(X,y,people)
        beta_resid=regress(np.stack([x for x,_ in residuals]),y-np.array([p for _,p in residuals]),people)
        models[kind]=dict(beta=beta.tolist(),beta_residual=beta_resid.tolist())
        for r in tests:
            x,prior=transform(r)
            eeg=float(x@beta_resid)
            predictions.append(dict(person=r['person'],fold=number,kind=kind,
                                    blocks='|'.join(r['components']),rating_delta=r['d'],
                                    binary_contrast=int(r['binary_contrast']),constant=0,legacy=r['old'],
                                    task_order=float(prior),optimized=float(r['x']@beta),
                                    residual_eeg=eeg,task_plus_eeg=float(prior+eeg)))
    return predictions,dict(train=train_ids,test=test_ids,models=models,
                            feature_baselines={str(k):v.tolist() for k,v in e_means.items()},
                            rating_baselines={str(k):float(v) for k,v in y_means.items()})


def cluster_stats(values,other=None):
    v=np.array(values,dtype=float)
    if other is not None:
        v=v-np.array(other)
    boot=np.random.default_rng(SEED).integers(len(v),size=(10000,len(v)))
    return dict(mean=float(v.mean()),ci95=np.percentile(v[boot].mean(axis=1),[2.5,97.5]).tolist())


def score(rows,key):
    delta=np.array([r['rating_delta'] for r in rows])
    pred=np.array([r[key] for r in rows])
    # 0.5 for exact score ties; rows with equal ratings have already been excluded.
    return float(np.mean((np.sign(delta)*np.sign(pred)+1)/2))


def evaluate(rows,locked):
    valid=[r for r in rows if r['rating_delta']!=0]
    groups=sorted({r['person'] for r in valid})
    prob={k:[score([r for r in valid if r['person']==p],k) for p in groups] for k in METHODS}
    intervals={k:cluster_stats(v) for k,v in prob.items()}
    deltas={a+'_minus_'+b:cluster_stats(prob[a],prob[b]) for a,b in (
        ('optimized','legacy'),('residual_eeg','legacy'),('task_plus_eeg','task_order'))}
    for k in METHODS:
        if locked:
            centered=np.array(prob[k])-.5
            null=np.array(list(product((-1,1),repeat=len(centered)))) @ centered
            intervals[k]['sign_flip_p']=float(np.mean(null>=centered.sum()-1e-12))
    return dict(eligible_including_ties=len(rows),rated_pairs=len(valid),people=len(groups),
                participant_scores={k:dict(zip(groups,v)) for k,v in prob.items()},
                metrics=intervals,paired=deltas)


def main(repo,out):
    out.mkdir(exist_ok=True,parents=True)
    cached=out/'features.json'
    data=read_json(cached) if cached.exists() else prepare(repo,out)
    rows=data['rows']
    contrasts=make_contrasts(rows)
    dev=sorted(set(range(1,41))-set(data['held']))
    folds=data['folds']
    assert sorted(p for grp in folds for p in grp)==dev
    preds=[]
    params=[]
    for i,test in enumerate([*folds,data['held']]):
        train=sorted(set(dev)-set(test))
        foldpreds,weights=fold(rows,contrasts,train,test,'holdout' if i==5 else f'dev{i+1}')
        preds.extend(foldpreds)
        params.append(weights)
    write_csv(out/'predictions.csv',preds)
    dump(out/'models.json',params)
    result={}
    for locked,name in ((False,'development'),(True,'holdout')):
        split=[r for r in preds if (r['fold']=='holdout')==locked]
        for key in ('within_round','double_difference','same_task_across_rounds','within_round_binary'):
            part=[r for r in split if r['kind']==key] if key!='within_round_binary' else [
                r for r in split if r['kind']=='within_round' and r['binary_contrast']]
            result[name+'_'+key]=evaluate(part,locked)
    mist_file=out/'mist_design.json'
    if not mist_file.exists():
        legacy_rows=read_json(repo/'var/new-cohorts-20261115/stress/mist_matched_features.json')
        dump(mist_file,[{key:r[key] for key in ('person','task','label')} for r in legacy_rows])
    positive=read_json(mist_file)
    # The publication defines one complete visit per condition for each person.
    sessions=defaultdict(list)
    for r in positive:
        sessions[r['person'],r['label']].append(r)
    result['mist_identification']=dict(people=len({r['person'] for r in positive}),sessions=len(sessions),
        blocks=len(positive),blocks_per_session=sorted({len(group) for group in sessions.values()}),
        groups_with_within_session_condition_change=sum(len({r['label'] for r in group})>1 for group in sessions.values()),
        session_centered_label_sse=float(sum(sum((r['label']-np.mean([a['label'] for a in group]))**2 for r in group)
                                            for group in sessions.values())))
    result['coverage']=dict(assigned_blocks=len(rows),eligible_blocks=sum(bool(r['x']) and r['old'] is not None for r in rows),
                            dev_assigned=dev,holdout_assigned=data['held'],
                            source_feature_hash=hashlib.sha256(cached.read_bytes()).hexdigest())
    dump(out/'evaluation.json',result)
    for key,v in result.items():
        if 'metrics' in v:
            print(key,v['people'],v['rated_pairs'],{k:round(v['metrics'][k]['mean'],4) for k in METHODS})


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('repo',type=Path)
    ap.add_argument('out',type=Path)
    args=ap.parse_args()
    main(args.repo,args.out)
