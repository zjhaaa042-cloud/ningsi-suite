"""Audit source ratings, held-out weights and the task-preserving pressure null."""
import argparse
import json
from pathlib import Path

import numpy as np
from validate_within_round_stress import METHODS, TASKS, dump, make_contrasts, read_csv, read_json


def shuffled_test(rows, locked_ids, records, kind):
    # Each individual carries their own full 3 rounds x 3 tasks of ratings.
    # Shuffling whole 9-rating profiles between participants preserves every task
    # and round mean as well as within-donor dynamics and ties.
    people=sorted(locked_ids)
    person_index={p:i for i,p in enumerate(people)}
    ratings=np.empty((len(people),3,len(TASKS)))
    for row in rows:
        if row['person'] in locked_ids:
            ratings[person_index[row['person']],row['round']-1,TASKS.index(row['task'])]=row['rating']
    subset=[r for r in records if r['kind']==kind and r['fold']=='holdout']
    tests=np.array([person_index[int(r['person'])] for r in subset])
    donors=[]
    signs=np.array([1,-1]) if kind=='within_round' else np.array([1,-1,-1,1])
    for r in subset:
        ids=[c.rsplit('-',1) for c in r['blocks'].split('|')]
        rep=[int(n)-1 for _,n in ids]
        task=[TASKS.index(t) for t,_ in ids]
        donors.append(np.sign(ratings[:,rep,task]@signs))
    truth=np.stack(donors,axis=1)
    rng=np.random.default_rng(56481)
    permutations=9999
    assignment=np.argsort(rng.random((permutations,len(people))),axis=1)
    assignment=np.vstack([np.arange(len(people)),assignment])
    label=truth[assignment[:,tests],np.arange(len(tests))]
    assert np.array_equal(label[0],np.sign([float(r['rating_delta']) for r in subset]))
    unique=np.unique(tests)
    result={}
    for method in ('legacy','optimized','residual_eeg','task_order','task_plus_eeg'):
        scores=np.sign([float(r[method]) for r in subset])
        correct=np.where(label==0,np.nan,(label*scores+1)/2)
        personal=[]
        for p in unique:
            data=correct[:,tests==p]
            count=np.isfinite(data).sum(axis=1)
            mean=np.full(len(data),np.nan)
            np.divide(np.nansum(data,axis=1),count,out=mean,where=count>0)
            personal.append(mean)
        statistic=np.nanmean(personal,axis=0)
        result[method]=dict(observed=float(statistic[0]),mean_under_null=float(np.nanmean(statistic[1:])),
                            p_one_sided=float((1+sum(statistic[1:]>=statistic[0]-1e-12))/(1+permutations)),
                            central95_null=np.nanpercentile(statistic[1:],[2.5,97.5]).tolist())
    return dict(permutations=permutations,blocks_lookup_donors=len(people),scores=result)


def main(out):
    data=read_json(out/'features.json')
    paper=read_json(out/'paper_table.json')['rows']
    assert len(paper)==42 and any('Stroop' in h for h in paper[1])
    paper_ratings={}
    for row in paper[2:]:
        for rep in range(3):
            for i,task in enumerate(('Stroop','Arithmetic','Mirror_image')):
                paper_ratings[int(row[0]),rep+1,task]=int(row[1+rep*3+i])
    assert all(r['rating']==paper_ratings[r['person'],r['round'],r['task']] for r in data['rows'])
    contrasts=make_contrasts(data['rows'])
    source={(r['person'],r['kind'],'|'.join(r['components'])):r for r in contrasts}
    preds=read_csv(out/'predictions.csv')
    model=read_json(out/'models.json')
    seen=set()
    maxerr=0.
    for r in preds:
        identity=(int(r['person']),r['kind'],r['blocks'])
        assert identity not in seen
        seen.add(identity)
        original=source[identity]
        which=5 if r['fold']=='holdout' else int(r['fold'][3:])-1
        bundle=model[which]
        assert identity[0] in bundle['test'] and identity[0] not in bundle['train']
        assert float(r['rating_delta'])==original['d'] and float(r['legacy'])==original['old']
        signs=np.array(original['sign'])
        nuisance=signs@np.array([bundle['rating_baselines'][str(k)] for k in original['keys']])
        demeaned=original['x']-signs@np.stack([bundle['feature_baselines'][str(k)] for k in original['keys']])
        w=bundle['models'][r['kind']]
        residual=demeaned@np.array(w['beta_residual'])
        expected=[0,original['old'],nuisance,original['x']@w['beta'],residual,nuisance+residual]
        maxerr=max(maxerr,max(abs(float(r[k])-v) for k,v in zip(METHODS,expected)))
    assert maxerr<1e-12
    result=dict(verified_paper_ratings=len(paper_ratings),unique_prediction_rows=len(seen),
                replay_max_abs_error=float(maxerr),task_preserving_permutation={})
    for kind in ('within_round','double_difference'):
        result['task_preserving_permutation'][kind]=shuffled_test(data['rows'],data['held'],preds,kind)
    dump(out/'verification.json',result)
    print(json.dumps(result,ensure_ascii=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    main(parser.parse_args().out)
