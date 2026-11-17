"""Independent participant holdout: does raw dual-frontal EEG add to pre-probe SART behavior?"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, find_peaks, resample_poly, sosfiltfilt, welch
from scipy.special import expit
from scipy.stats import rankdata

from attlapse_two_channel_eval import Candidate, fit, predict


BEHAV = ("probe_progress","face_or_digit","go_median_rt","go_cv","go_omissions","nogo_commissions","has_nogo",
         "go_rt_change","go_cv_change","go_omission_change","nogo_commission_change")
BANDS = ((1,4),(4,8),(8,13),(13,30),(30,40))
CONFIG = Candidate("physical","preprobe","frontal","logistic",.1)
SEED = 11478


def performance(rows):
    go = rows[:,4] != rows[:,5]
    correct_rt = rows[go & (rows[:,11]==1),9] - rows[go & (rows[:,11]==1),7]
    correct_rt = correct_rt[(correct_rt>0) & (correct_rt <= 2.)]
    if len(correct_rt)<3 or not go.any():
        return None
    return np.array([np.median(correct_rt),np.std(correct_rt,ddof=1)/np.mean(correct_rt),
                     np.mean(rows[go,11]==0),np.mean(rows[~go,10]==0) if (~go).any() else 0.])


def behavior(trials, probe, idx):
    end = probe[2]
    # Independent hardware/behavior clocks are verified before this function.
    past = trials[(trials[:,0]==probe[3]) & (trials[:,3]<=probe[5]) &
                  (trials[:,7]+trials[:,8]<=end) & (~np.isfinite(trials[:,9]) | (trials[:,9]<end))]
    if len(past)<18:
        return None
    recent = performance(past[-18:])
    if recent is None:
        return None
    # Previous behavior is always available before the score; no test labels.
    previous = performance(past[:-18]) if len(past)>=30 else None
    delta = recent-previous if previous is not None else np.zeros(4)
    return np.r_[idx/59.,int(probe[4]==1),recent,int((past[-18:,4]==past[-18:,5]).any()),delta]


def clean_windows(samples, rate):
    if rate % 250 or samples.shape != (2,7*rate) or not np.isfinite(samples).all():
        raise ValueError('Wrong rate, shape or nonfinite raw samples')
    # BrainVision preserves electrode DC offsets, which can exceed 10,000 µV.
    # Remove that arbitrary constant *before* any zero-padded rate conversion,
    # and extend the observed segment at its endpoints rather than to zero.
    centered = samples.astype(np.float64)
    centered -= np.median(centered,axis=1,keepdims=True)
    x = resample_poly(centered,1,rate//250,axis=1,padtype='line')
    x = sosfiltfilt(butter(4,(.5,40),btype='bandpass',fs=250,output='sos'),x)
    # Padded filtering is confined to the observed pre-answer interval [-8, -1).
    chunks = [x[:,end-1000:end] for end in (1250,1500,1750)]
    for pair in chunks:
        ptp = np.ptp(pair,axis=1)
        if np.any((ptp < 1) | (ptp > 180)) or np.max(np.abs(np.diff(pair,axis=1))) > 65:
            continue
        yield pair


def spectrum(samples, rate):
    old, rich = [], []
    for pair in clean_windows(samples, rate):
        signals = np.stack([pair[0],pair[1],pair[0]-pair[1]])
        f, power = welch(signals,fs=250,nperseg=500,noverlap=250,axis=-1)
        p = np.stack([np.sum(power[:,(f>=lo)&(f<hi)],axis=1) for lo,hi in BANDS],axis=1)
        log = np.log(p+1e-12)
        rel = np.log((p+1e-12)/(p.sum(axis=1,keepdims=True)+5e-12))
        old.append(np.r_[log[2,1:4],rel[2,1:4]])
        slow = sosfiltfilt(butter(3,(.8,4),btype='bandpass',fs=250,output='sos'),signals)
        peaks = [find_peaks(-lead,distance=62,prominence=12)[1]['prominences'] for lead in slow]
        slow_features = [len(pks)/4 for pks in peaks] + [np.max(pks) if len(pks) else 0. for pks in peaks]
        rich.append(np.r_[log[:2].mean(axis=0),log[0]-log[1],log[2],slow_features])
    if not old:
        return None
    return np.median(old,axis=0),np.median(rich,axis=0),len(old)


def features(out):
    plans=json.loads((out/'synchronization.json').read_text(encoding='utf8'))['included']
    rows=[]
    for p in plans:
        person=p['person']
        data=loadmat(out/'source'/p['behavior'])
        probes, trials=data['probe_res'],data['test_res']
        # The acquisition ID and behavioral participant ID must agree.
        person_info=loadmat(out/'source'/p['behavior'],simplify_cells=True)['SubjectInfo']
        if int(person_info['subID']) != int(person):
            raise ValueError('Participant identity mismatch')
        prior=[]
        for i,probe in enumerate(probes):
            path=out/'source'/('p'+person)/f'{i:02}.npy'
            if not path.exists():
                raise ValueError(f'Missing raw input {path}')
            rec=dict(person=person,probe=i+1,label=int(probe[31]==2) if probe[31] in (1,2) else None,
                     response_state=None if not np.isfinite(probe[31]) else int(probe[31]), current_look=bool(probe[30]==1),
                     behavior=None, old=None, eeg=None, valid_chunks=0,
                     sample_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            b=behavior(trials,probe,i)
            if b is not None:
                rec['behavior']=b.tolist()
            result=spectrum(np.load(path,allow_pickle=False),p['rate'])
            if result is not None:
                o,r,qual=result
                # Shrink causal within-person levels toward the current observation
                # when fewer than five prior usable observations are available.
                deviation=(r-np.median(prior,axis=0))*min(len(prior)/5,1.) if prior else np.zeros_like(r)
                rec.update(old=o.tolist(),eeg=np.r_[r,deviation].tolist(),valid_chunks=qual)
                prior.append(r)
            rows.append(rec)
    return rows


def group_auc(y,s,person):
    result={}
    for p in np.unique(person):
        sel=person==p
        yy=y[sel]
        pos=yy==1
        n=pos.sum()
        if n and n<len(yy):
            result[p]=float((rankdata(s[sel])[pos].sum()-n*(n+1)/2)/(n*(len(yy)-n)))
    return result


def macro(y,s,p):
    return float(np.mean(list(group_auc(y,s,p).values())))


def fold_indices(p, seed, n):
    people=np.random.default_rng(seed).permutation(np.unique(p))
    return [(~np.isin(p,part), np.isin(p,part)) for part in np.array_split(people,n)]


def train(x, y, p):
    return fit(x,y,p,CONFIG)


def residual(x,y,people,offset):
    # Fit additional coefficients with a frozen behavioral logit as offset.
    # Balancing by person/class matches the baseline training loss.
    _,inv,freq=np.unique(people,return_inverse=True,return_counts=True)
    sample=1/freq[inv]
    for c in (0,1):
        m=y==c
        sample[m] /= 2*sample[m].sum()
    mu=sample@x
    sd=np.sqrt(sample@((x-mu)**2))
    sd[sd<1e-8]=1
    z=np.column_stack([np.ones(len(x)),(x-mu)/sd])
    theta=np.zeros(z.shape[1])
    penalty=np.diag([0.]+[.1]*(z.shape[1]-1))
    for _ in range(50):
        prob=expit(offset+z@theta)
        grad=z.T@(sample*(prob-y))+penalty@theta
        hess=z.T@((sample*prob*(1-prob))[:,None]*z)+penalty
        step=np.linalg.solve(hess,grad)
        theta-=step
        if np.max(np.abs(step))<1e-8:
            break
    return dict(mean=mu.tolist(),scale=sd.tolist(),bias=float(theta[0]),weights=theta[1:].tolist())


def margin(s):
    return np.log(np.clip(s,1e-9,1-1e-9)/np.clip(1-s,1e-9,1))


def fixed_arms(b, xx, y,p,trainmask,testmask):
    back=train(b[trainmask],y[trainmask],p[trainmask])
    base_train=margin(predict(back,b[trainmask]))
    base=margin(predict(back,b[testmask]))
    scores={'behavior':expit(base)}
    scores['progress_task_only']=predict(train(b[trainmask,:2],y[trainmask],p[trainmask]),b[testmask,:2])
    weights={'behavior':back}
    for key,x in xx.items():
        model=residual(x[trainmask],y[trainmask],p[trainmask],base_train)
        z=(x[testmask]-model['mean'])/model['scale']
        added=model['bias']+z@model['weights']
        for factor in (.25,.65,1.):
            scores[f'{key}_{factor}']=expit(base+factor*added)
        weights[key]=model
        scores[key+'_only']=predict(train(x[trainmask],y[trainmask],p[trainmask]),x[testmask])
    return scores,weights


def select(b,xx,y,p,seed):
    scores={}
    for fit_mask, validation in fold_indices(p,seed,4):
        s,_=fixed_arms(b,xx,y,p,fit_mask,validation)
        for key,v in s.items():
            if key.endswith('only'):
                continue
            scores.setdefault(key,np.zeros(len(y)))[validation]=v
    ranked={k:macro(y,v,p) for k,v in scores.items()}
    chosen=max(ranked,key=ranked.get)
    return chosen,ranked


def summarize(scored,reference='behavior'):
    groups=np.array([r['person'] for r in scored]);y=np.array([r['label'] for r in scored])
    keys=sorted(scored[0]['scores'])
    stats={}
    pairs={}
    for key in keys:
        pred=np.array([r['scores'][key] for r in scored])
        by=group_auc(y,pred,groups)
        pairs[key]=np.array([val for _,val in sorted(by.items())])
        stats[key]=dict(macro_auc=float(pairs[key].mean()),balanced_accuracy=float(.5*((pred[y==1]>=.5).mean()+(pred[y==0]<.5).mean())))
    n=len(pairs[reference]);rng=np.random.default_rng(SEED); idx=rng.integers(0,n,size=(5000,n))
    for key,seq in pairs.items():
        diff=seq-pairs[reference]
        stats[key].update(ci95=np.quantile(seq[idx].mean(axis=1),[.025,.975]).tolist(),
                          delta=float(diff.mean()),delta_ci95=np.quantile(diff[idx].mean(axis=1),[.025,.975]).tolist(),
                          people_improving=int(np.sum(diff>1e-12)),people_tied=int(np.sum(np.abs(diff)<=1e-12)))
    return dict(people=sorted(np.unique(groups).tolist()),both_classes_people=n,probes=len(y),positive=int(y.sum()),scores=stats)


def evaluate(out,rows):
    participants=sorted({r['person'] for r in rows})
    signatures={r['sample_sha256'] for r in rows}
    if len(signatures)!=len(rows):
        raise ValueError("Duplicate complete frontal EEG epoch; investigate before scoring")
    held=sorted(np.random.default_rng(SEED).permutation(participants)[:8].tolist())
    common=[r for r in rows if r['label'] is not None and r['behavior'] is not None and r['eeg'] is not None]
    b=np.array([r['behavior'] for r in common]);xx={k:np.array([r[k] for r in common]) for k in ('old','eeg')}
    y=np.array([r['label'] for r in common]);p=np.array([r['person'] for r in common])
    development=~np.isin(p,held)
    outer=[]
    for k,person in enumerate(np.array_split(np.random.default_rng(SEED+19).permutation(np.unique(p[development])),5)):
        validation=np.isin(p,person)
        outer.append((f'dev{k+1}',development&~validation,validation))
    outer.append(('holdout',development,~development))
    log=[];all_predictions=[];weights={}
    for k,(part,tr,te) in enumerate(outer):
        assert not(set(p[tr])&set(p[te]))
        choice,ranks=select(b[tr],{k:x[tr] for k,x in xx.items()},y[tr],p[tr],SEED+k)
        scores,ws=fixed_arms(b,xx,y,p,tr,te)
        scores['candidate']=scores[choice]
        if part=='holdout':
            weights=dict(weights=ws,chosen=choice,task_features=BEHAV,train_people=sorted(set(p[tr])),holdout=held,
                         note='Frozen research model; requires SART behavioral stream and raw Fp1/Fp2.')
        preds=[dict(person=common[j]['person'],probe=common[j]['probe'],label=int(y[j]),split=part,
                    scores={name:float(s[i]) for name,s in scores.items()}) for i,j in enumerate(np.flatnonzero(te))]
        all_predictions.extend(preds)
        log.append(dict(split=part,train=sorted(set(p[tr])),test=sorted(set(p[te])),choice=choice,inner_macro_auc=ranks,
                        estimation=summarize(preds)))
    with (out/'predictions.csv').open('w',newline='',encoding='utf8') as fp:
        writer=csv.DictWriter(fp,fieldnames=['person','probe','label','split',*all_predictions[0]['scores']])
        writer.writeheader()
        for r in all_predictions:
            writer.writerow({**{k:v for k,v in r.items() if k!='scores'},**r['scores']})
    dev=summarize([r for r in all_predictions if r['split']!='holdout'])
    test=summarize([r for r in all_predictions if r['split']=='holdout'])
    result=dict(source='https://osf.io/ey3ca/', raw_source='https://osf.io/w7gz5/', behavior_source='https://osf.io/6g2fh/',
                subjects=len(participants),all_probes=len(rows),valid_binary=sum(r['label'] is not None for r in rows),
                valid_behavior=sum(r['label'] is not None and r['behavior'] is not None for r in rows),
                unique_epoch_hashes=len(signatures),
                common_probes=len(common),design_sha256=hashlib.sha256((out/'design.md').read_bytes()).hexdigest(),
                feature_names_behavior=BEHAV,development=dev,holdout=test,folds=log)
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8')
    (out/'candidate.json').write_text(json.dumps(weights,indent=2,allow_nan=False),encoding='utf8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    rows=features(args.output)
    (args.output/'features.json').write_text(json.dumps(rows,allow_nan=False),encoding='utf8')
    stats=evaluate(args.output,rows)
    print(json.dumps({name:stats[name]['scores']['candidate'] for name in ('development','holdout')},indent=2))
