"""Participant-independent evaluation on explicit FACED self-rated valence from Fp1 and Fp2."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.signal import welch
from scipy.special import expit
from scipy.stats import rankdata

from compare_frontal_affect import filtered_windows
from fetch_faced_frontal import ORDER
from ningsi.models.affect import feature_vector,load_models
from ningsi.models.human import logits
from ningsi.signal.window import analyze_window


SEED=53141
MODELS=('three_bands','bilateral','history','history_rff')
BANDS=((2,4),(4,8),(8,10),(10,13),(13,20),(20,30),(30,40))


def features(pair):
    signals=np.vstack((pair,pair[1]-pair[0]))
    f,power=welch(signals,fs=250,nperseg=250,noverlap=125)
    band=np.stack([power[:,(f>=lo)&(f<hi)].sum(axis=1) for lo,hi in BANDS],axis=1)
    logp=np.log(np.maximum(band,1e-13))
    relative=logp-np.log(np.maximum(band.sum(axis=1,keepdims=True),1e-13))
    return np.r_[logp[:2].mean(axis=0), relative[:2].mean(axis=0), logp[1]-logp[0], relative[2], np.corrcoef(pair)[0,1]]


def extract_person(args):
    root,person=args
    folder=root/f'sub-{person:03}'
    paths=sorted(folder.glob('clip-*.npz'))
    if len(paths)!=28: raise ValueError(f'Missing FACED short waveforms: {person}, {len(paths)}')
    meta=json.loads((folder/'provenance.json').read_text(encoding='utf8'))
    events=list(csv.DictReader((folder/'events.tsv').read_text(encoding='utf-8-sig').splitlines(),delimiter='\t'))
    events=[e for e in events if e['video_index']!='n/a']
    if any(float(a['onset'])>=float(b['onset']) for a,b in zip(events,events[1:])): raise ValueError('Nonchronological event sequence')
    output=[]
    for path in paths:
        with np.load(path,allow_pickle=False) as source:
            x=source['uv'].astype(float); rating=float(source['rating']); pos=int(source['position']); interval=source['sample_range'].tolist(); source_hash=str(source['sha256_range'])
        clip=int(path.stem.split('-')[1]);ev=events[pos]
        assert int(ev['video_index'])==clip and rating==float(ev['Valence']) and x.shape==(2,3000) and np.isfinite(x).all()
        sr=meta['rate']; end=round((float(ev['onset'])+float(ev['duration'])-1)*sr)
        assert interval==[end-12*sr,end]
        bands=[]; extended=[]; failures=Counter()
        for pair in filtered_windows(x,250,ends=np.array([1500,2000,2500,3000])):
            win=analyze_window(pair,250)
            if win.usable:
                bands.append(feature_vector(win).tolist());extended.append(features(pair))
            else: failures.update(win.quality.reasons)
        output.append(dict(person=person, clip=clip, position=pos, rate=sr, rating=rating,
                           label=int(rating>3.5) if rating!=3.5 else None, legacy=bands,
                           bilat=np.median(extended,axis=0).tolist() if extended else None,
                           qc=dict(failures),source_hash=source_hash))
    history=[]
    for row in sorted(output,key=lambda r:r['position']):
        row['prior_n']=len(history)
        row['prior_sum']=np.sum(history,axis=0).tolist() if history else None
        if row['bilat'] is not None: history.append(row['bilat'])
    return output


def weights(rows):
    counts=Counter(r['person'] for r in rows)
    y=np.array([r['label'] for r in rows]);w=np.array([1/counts[r['person']] for r in rows])
    for c in (0,1): w[y==c]/=(2*w[y==c].sum())
    return w


def matrix(rows,name,prior):
    if name=='three_bands': return np.array([np.median(r['legacy'],axis=0) for r in rows])
    raw=np.array([r['bilat'] for r in rows]);
    if name=='bilateral': return raw
    past=np.array([(r['prior_sum'] if r['prior_sum'] is not None else np.zeros(raw.shape[1])) for r in rows])
    avg=(past+3*prior)/(np.array([r['prior_n'] for r in rows])[:,None]+3)
    return np.column_stack((raw,raw-avg))


def rff(z):
    rng=np.random.default_rng(83047)
    omega=rng.normal(size=(z.shape[1],96))/np.sqrt(z.shape[1]);phase=rng.uniform(0,2*np.pi,size=96)
    return np.column_stack((z, np.sqrt(2)*np.cos(z@omega+phase)))


def train(rows,name):
    w=weights(rows);y=np.array([r['label'] for r in rows]);prior=np.average(np.array([r['bilat'] for r in rows]),axis=0,weights=w)
    x=matrix(rows,name,prior);mean=np.average(x,axis=0,weights=w);sd=np.sqrt(np.average((x-mean)**2,axis=0,weights=w));sd=np.maximum(sd,1e-6)
    z=np.clip((x-mean)/sd,-8,8)
    if name=='history_rff': z=rff(z)
    def objective(coef):
        v=z@coef[:-1]+coef[-1]
        cost=w@(np.logaddexp(0,v)-y*v)+.025*(coef[:-1]@coef[:-1]); e=w*(expit(v)-y)
        return cost,np.r_[z.T@e+.05*coef[:-1],e.sum()]
    opt=minimize(objective,np.zeros(z.shape[1]+1),method='L-BFGS-B',jac=True,options={'ftol':1e-11,'maxiter':300})
    if not opt.success: raise ValueError(opt.message)
    return dict(name=name,mean=mean.tolist(),scale=sd.tolist(),prior=prior.tolist(),weights=opt.x[:-1].tolist(),bias=float(opt.x[-1]))


def predict(rows,model):
    x=matrix(rows,model['name'],np.array(model['prior']));z=np.clip((x-model['mean'])/model['scale'],-8,8)
    if model['name']=='history_rff':z=rff(z)
    return expit(z@model['weights']+model['bias'])


def auc(y,p):
    k=int(y.sum());n=len(y)
    return (rankdata(p)[y==1].sum()-k*(k+1)/2)/(k*(n-k)) if 0<k<n else float('nan')


def ba(y,p):
    return .5*((p[y==1]>=.5).mean()+(p[y==0]<.5).mean())


def split(rows,ids):
    membership=set(ids);return [r for r in rows if r['person'] not in membership],[r for r in rows if r['person'] in membership]


def select(rows,k,seed):
    people=sorted({r['person'] for r in rows}); folds=np.array_split(np.random.default_rng(seed).permutation(people),k)
    scores={m:[] for m in MODELS}
    for ids in folds:
        train_rows,test_rows=split(rows,ids); y=np.array([r['label'] for r in test_rows])
        for name in MODELS:
            scores[name].append(float(ba(y,predict(test_rows,train(train_rows,name)))))
    best=max(MODELS,key=lambda m: np.mean(scores[m]));return best,scores


def evaluate(train_rows,test_rows,choice,phase):
    y=np.array([r['label'] for r in train_rows]); prior={}
    for c in range(1,29):
        ss=[r['label'] for r in train_rows if r['clip']==c];prior[c]=(sum(ss)+1)/(len(ss)+2) if ss else float(np.mean(y))
    models={name:train(train_rows,name) for name in MODELS}
    scores={k:predict(test_rows,m) for k,m in models.items()}
    legacy=load_models()['valence']['parameters']
    out=[]
    for i,r in enumerate(test_rows):
        z={k:float(v[i]) for k,v in scores.items()}
        out.append(dict(phase=phase,person=r['person'],clip=r['clip'],rate=r['rate'],rating=r['rating'],label=r['label'],
                        usable_windows=len(r['legacy']), old=float(expit(logits(legacy,np.array(r['legacy']))).mean()),
                        selected=z[choice],prior=prior[r['clip']],**z))
    return out,models


def summarize(rows,chosen=('old','selected','prior',*MODELS)):
    people=sorted({r['person'] for r in rows});groups=np.array([r['person'] for r in rows]);y=np.array([r['label'] for r in rows]);draw=np.random.default_rng(SEED).integers(0,len(people),(4000,len(people)))
    # A film-only prior trained on separate folds may differ between folds.
    # Compare people within the same film and model fit, then average by film.
    clip_groups=sorted({(r.get('phase',''),r['clip']) for r in rows})
    result={};boot={}
    for name in chosen:
        p=np.array([r[name] for r in rows]); hits=[];aucs=[];bas=[]
        for person in people:
            use=groups==person;yy=y[use];pp=p[use];pos=yy==1;neg=~pos
            hits.append([np.sum(pp[pos]>=.5),sum(pos),np.sum(pp[neg]<.5),sum(neg)]);aucs.append(auc(yy,pp));bas.append(ba(yy,pp) if pos.any() and neg.any() else np.nan)
        sums=np.array(hits)[draw].sum(axis=1);b=.5*(sums[:,0]/sums[:,1]+sums[:,2]/sums[:,3]);ab=np.nanmean(np.array(aucs)[draw],axis=1)
        per_film={}
        for phase,clip in clip_groups:
            subset=np.array([r.get('phase','')==phase and r['clip']==clip for r in rows])
            value=auc(y[subset],p[subset])
            if np.isfinite(value):per_film.setdefault(clip,[]).append(value)
        by_clip=[np.mean(v) for v in per_film.values()]
        result[name]=dict(ba=float(ba(y,p)), ba_ci=np.nanquantile(b,[.025,.975]).tolist(),participant_macro_ba=float(np.nanmean(bas)),
                          within_person_auc=float(np.nanmean(aucs)),within_person_auc_ci=np.nanquantile(ab,[.025,.975]).tolist(), pooled_auc=float(auc(y,p)),
                          within_clip_auc=float(np.nanmean(by_clip)),clips_with_both=int(np.isfinite(by_clip).sum()), people_with_both=int(np.isfinite(aucs).sum()))
        boot[name]=(b,ab)
    for candidate in ('selected','history'):
        if candidate in chosen and 'old' in chosen:
            for key,i in [('ba',0),('within_person_auc',1)]:
                result[candidate][key+'_delta_against_old']=result[candidate][key]-result['old'][key]
                result[candidate][key+'_delta_ci']=np.nanquantile(boot[candidate][i]-boot['old'][i],[.025,.975]).tolist()
    return dict(n=len(rows),people=len(people),positive=int(y.sum()),metrics=result)


def main(out,workers):
    source=out/'source'; feature_file=out/'features.json'
    if feature_file.exists(): rows=json.loads(feature_file.read_text(encoding='utf8'))
    else:
        with ProcessPoolExecutor(workers) as pool: rows=[r for person in pool.map(extract_person,[(source,p) for p in ORDER]) for r in person]
        feature_file.write_text(json.dumps(rows,separators=(',',':'),allow_nan=False),encoding='utf8')
    assert len(rows)==123*28 and len({r['source_hash'] for r in rows})==123*28
    rating_vectors={tuple(r['rating'] for r in sorted((r for r in rows if r['person']==p),key=lambda r:r['clip'])) for p in ORDER}
    assert len(rating_vectors)==123
    ok=[r for r in rows if r['label'] is not None and r['bilat'] is not None]
    dev=[r for r in ok if r['person'] in ORDER[:32]];locked=[r for r in ok if r['person'] in ORDER[32:]]
    # Never use locked participants in feature selection, calibration, thresholds or a global prior.
    nested=[];decisions=[]
    for i,fold in enumerate(np.array_split(ORDER[:32],5)):
        a,b=split(dev,fold);name,inner=select(a,4,SEED+11+i);o,_=evaluate(a,b,name,f'development_oof_{i+1}');nested+=o
        decisions.append(dict(test_people=fold.tolist(),chosen=name,inner_scores=inner))
    choice,inner=select(dev,5,SEED+81);hold,models=evaluate(dev,locked,choice,'holdout')
    test=[dict(r,phase='initial') for r in hold if r['person'] in ORDER[32:48]];rep=[dict(r,phase='replication') for r in hold if r['person'] in ORDER[48:]]
    # New movie challenge requires model choice itself to exclude those films.
    # Every held-out person gets a causal rolling baseline built from previous waveform inputs only.
    movie_out=[];movie_folds=[]
    for i,clips in enumerate(np.array_split(np.random.default_rng(SEED+29).permutation(np.arange(1,29)),4)):
        # Do not let excluded films enter any training subject's historical reference.
        subset=[r.copy() for r in rows if r['person'] in ORDER[:32] and r['clip'] not in clips]
        for group in ORDER[:32]:
            history=[]
            for r in sorted((r for r in subset if r['person']==group),key=lambda r:r['position']):
                r['prior_n']=len(history);r['prior_sum']=np.sum(history,axis=0).tolist() if history else None
                if r['bilat'] is not None:history.append(r['bilat'])
        a=[r for r in subset if r['label'] is not None and r['bilat'] is not None]; b=[r for r in locked if r['clip'] in clips]
        n,sel=select(a,4,SEED+45+i);o,_=evaluate(a,b,n,'unseen_movies_and_people');movie_out+=o;movie_folds.append(dict(clips=clips.tolist(),choice=n))
    out.mkdir(parents=True,exist_ok=True)
    with (out/'trial_predictions.csv').open('w',newline='',encoding='utf8') as fp:
        writer=csv.DictWriter(fp,fieldnames=list(hold[0]));writer.writeheader();writer.writerows(nested+test+rep+movie_out)
    halves={phase:summarize(d) for phase,d in [('development_nested',nested),('initial',test),('replication',rep),('pooled_holdout',hold),('unseen_movies_and_people',movie_out)]}
    subfolds=[]
    for group in np.array_split(ORDER[32:],5):
        subfolds.append(summarize([r for r in hold if r['person'] in group],('old','selected')))
    for rate in (250,1000):
        sub=[r for r in hold if r['rate']==rate]
        if sub:halves[str(rate)+'Hz']=summarize(sub,('old','selected','prior'))
    quality={}
    for phase,people in [('development',ORDER[:32]),('initial',ORDER[32:48]),('replication',ORDER[48:])]:
        src=[r for r in rows if r['person'] in people];qc=Counter()
        for r in src:qc.update(r['qc'])
        quality[phase]={'original':len(src),'labelled':sum(r['label'] is not None for r in src),'eligible':sum(r['label'] is not None and bool(r['legacy']) for r in src),'reject_reasons':dict(qc)}
    payload=dict(cohort='FACED v1.1.4 / Chen et al. doi:10.1038/s41597-023-02650-w',design_sha256=hashlib.sha256((out/'design.txt').read_bytes()).hexdigest(),
                 feature_sha256=hashlib.sha256(feature_file.read_bytes()).hexdigest(),selection=choice, selection_scores=inner,
                 partitions={'development':ORDER[:32],'initial':ORDER[32:48],'replication':ORDER[48:]},quality=quality,
                 results=halves,fixed_splits_heldout=subfolds,nested_design=decisions,unseen_films=movie_folds,unique_rating_vectors=len(rating_vectors))
    (out/'evaluation.json').write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf8')
    (out/'research_weights.json').write_text(json.dumps(dict(models=models,selected=choice,training_subjects=ORDER[:32],rff_seed=83047,threshold=.5),indent=2),encoding='utf8')
    print(choice,[(k,v['n'],round(v['metrics']['selected']['ba'],3),v['metrics']['selected']['ba_ci']) for k,v in halves.items()])


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('out',type=Path);ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args();main(args.out,args.workers)
