"""Apply locked FACED Fp1/Fp2 valence candidates to the separate music study ds002721."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
import csv
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np
from scipy.signal import resample_poly
from scipy.special import expit

from benchmark_faced_valence import filtered_windows,analyze_window,feature_vector,features,predict,summarize
from fetch_faced_frontal import get
from ningsi.models.affect import load_models
from ningsi.models.human import logits


VERSION='1.0.3'
GITHUB=f'https://raw.githubusercontent.com/OpenNeuroDatasets/ds002721/{VERSION}/'
S3='https://s3.amazonaws.com/openneuro.org/ds002721/'


def verify_git(raw,ref):
    if hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()!=ref: raise ValueError('Pinned metadata mismatch')


def read_edf(raw):
    assert raw[:8]==b'0       '
    head=int(raw[184:192]);n=int(raw[236:244]);dt=float(raw[244:252]);nc=int(raw[252:256]);offset=256
    def field(width):
        nonlocal offset
        part=raw[offset:offset+width*nc];offset+=width*nc
        return [part[i:i+width].decode('latin1').strip() for i in range(0,len(part),width)]
    names=[v.upper() for v in field(16)];field(80);units=field(8);pmin=field(8);pmax=field(8);dmin=field(8);dmax=field(8);field(80);counts=list(map(int,field(8)))
    stride=2*sum(counts);assert head==256*(nc+1) and head+n*stride==len(raw)
    channels=[]
    for name in ('FP1','FP2'):
        i=names.index(name);assert counts[i]/dt==1000
        slope=(float(pmax[i])-float(pmin[i]))/(int(dmax[i])-int(dmin[i]));unit=units[i].lower().replace('µ','u')
        multiple={'uv':1.,'microv':1.,'v':1e6,'mv':1e3}[unit]
        v=np.ndarray((n,counts[i]),dtype='<i2',buffer=raw,offset=head+2*sum(counts[:i]),strides=(stride,2))
        channels.append((v.ravel().astype(float)*slope+float(pmin[i])-slope*int(dmin[i]))*multiple)
    return np.array(channels)


def trials(raw):
    rows=list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig')),delimiter='\t')); grouped=defaultdict(list)
    for row in rows: grouped[float(row['onset'])].append(int(row['trial_type']))
    starts=[t for t,codes in grouped.items() if 788 in codes];out=[]
    for j,t in enumerate(starts):
        stop=starts[j+1] if j+1<len(starts) else float('inf')
        later=[(tt,codes) for tt,codes in grouped.items() if t<tt<stop]; responses=[];first_question=None
        for tt,codes in later:
            if any(800<=n<=807 for n in codes):
                first_question=tt if first_question is None else min(first_question,tt)
            if 800 in codes and not any(801<=n<=807 for n in codes):
                ans={c-900 for c in codes if 901<=c<=909};ind={c-832 for c in codes if 833<=c<=841}
                if len(ans)==1 and ans==ind:responses.append((tt,ans.pop()))
        clip=[c-300 for c in grouped[t] if 301<=c<=399]
        reason=None
        if len(responses)!=1:reason='missing_or_ambiguous_paired_rating'
        elif first_question is None or first_question<t+12:reason='prompt_before_window_end'
        # Some exports lack a soundtrack ID adjacent to 788. The trial's own
        # time-locked reports and music onset are sufficient for zero-shot EEG
        # transfer; leave missing soundtrack IDs null rather than guessing.
        out.append(dict(index=j,onset=t,clip=clip[0] if len(clip)==1 else None,rating=responses[0][1] if len(responses)==1 else None,first_question=first_question,reason=reason))
    return out


def run_source(root,stem,tree):
    sub,run=map(int,re.search(r'sub-(\d+)_task-run(\d+)_$',stem).groups())
    target=root/f'causal-sub-{sub:02}-run-{run}.json'
    if target.exists():return json.loads(target.read_text(encoding='utf8'))
    evt=get(GITHUB+stem+'events.tsv')[0];verify_git(evt,tree[stem+'events.tsv']); detections=trials(evt)
    if not detections:
        result=dict(subject=sub,run=run,events_sha256=hashlib.sha256(evt).hexdigest(),rows=[])
        target.write_text(json.dumps(result),encoding='utf8');return result
    path=stem+'eeg.edf'; ref=get(GITHUB+path)[0];verify_git(ref,tree[path]);m=re.search(rb'MD5E-s(\d+)--([0-9a-f]{32})\.edf',ref)
    assert m is not None
    data=get(S3+path)[0]
    if len(data)!=int(m[1]) or hashlib.md5(data).hexdigest()!=m[2].decode():raise ValueError('Full EEG checksum mismatch')
    eeg=read_edf(data); del data
    for row in detections:
        row.update(person=sub,trial=f'{run}-{row["index"]}',rate=1000,run=run,legacy=[],bilat=None)
        # Even when a questionnaire response is ambiguous, its preceding clean
        # EEG is available to a causal participant reference at the next trial.
        off=round(row['onset']*1000);segment=eeg[:,off:off+12000]
        if segment.shape!=(2,12000):row['reason']=row['reason'] or 'truncated_waveform';continue
        if row['first_question'] is not None and row['first_question']<row['onset']+12:
            row['reason']=row['reason'] or 'prompt_before_window_end';continue
        x=resample_poly(segment,1,4,axis=1,padtype='line').astype(np.float32).astype(float)
        rich=[]
        for pair in filtered_windows(x,250,ends=np.array([1500,2000,2500,3000])):
            w=analyze_window(pair,250)
            if w.usable:row['legacy'].append(feature_vector(w).tolist());rich.append(features(pair))
        row['bilat']=np.median(rich,axis=0).tolist() if rich else None
        if not rich:row['reason']=row['reason'] or 'EEG_QC'
        row['label']=int(row['rating']>5) if row['rating'] not in (None,5) else None
    result=dict(subject=sub,run=run,eeg_md5=m[2].decode(),events_sha256=hashlib.sha256(evt).hexdigest(),rows=detections)
    target.write_text(json.dumps(result,allow_nan=False),encoding='utf8');return result


def run(out):
    parent=out.parent;out.mkdir(parents=True,exist_ok=True);source=out/'source';source.mkdir(exist_ok=True)
    version=source/'release_tree.json'
    if not version.exists():version.write_bytes(get(f'https://api.github.com/repos/OpenNeuroDatasets/ds002721/git/trees/{VERSION}?recursive=1')[0])
    inventory=json.loads(version.read_text(encoding='utf8'));assert not inventory['truncated']
    tree={r['path']:r['sha'] for r in inventory['tree'] if r['type']=='blob'}
    marker='sub-01/eeg/sub-01_task-run2_events.json';definition=get(GITHUB+marker)[0];verify_git(definition,tree[marker]);j=json.loads(definition)
    assert 'pleas' in j['800']['Description'].lower() and 'strongly disagree' in j['901']['Description'] and 'strongly agree' in j['909']['Description']
    stems=sorted(k.removesuffix('events.tsv') for k in tree if re.fullmatch(r'sub-\d+/eeg/sub-\d+_task-run\d+_events.tsv',k))
    files=[]
    with ThreadPoolExecutor(12) as pool:
        jobs=[pool.submit(run_source,source,k,tree) for k in stems]
        for count,job in enumerate(as_completed(jobs),1):
            files.append(job.result())
            if count%40==0:print(f'{count}/{len(stems)} runs',flush=True)
    all_rows=[row for record in files for row in record['rows']]; groups=sorted({r['person'] for r in all_rows});scored=[]
    for p in groups:
        hist=[]
        for row in sorted((r for r in all_rows if r['person']==p),key=lambda r:(r['run'],r['onset'])):
            if row['bilat'] is not None:
                row['prior_n']=len(hist);row['prior_sum']=np.sum(hist,axis=0).tolist() if hist else None;hist.append(row['bilat'])
                if not row['reason'] and row['label'] is not None:scored.append(row)
    weights=parent/'research_weights.json';bundle=json.loads(weights.read_text(encoding='utf8')); retained={n:predict(scored,m) for n,m in bundle['models'].items()}
    old=load_models()['valence']['parameters'];pred=[]
    for i,r in enumerate(scored):
        pred.append(dict(person=r['person'],run=r['run'],trial=r['trial'],clip=r['clip'] if r['clip'] is not None else -1,rating=r['rating'],label=r['label'],old=float(expit(logits(old,np.array(r['legacy']))).mean()),
                         selected=float(retained[bundle['selected']][i]),**{k:float(v[i]) for k,v in retained.items()}))
    with (out/'predictions.csv').open('w',newline='',encoding='utf8') as fp:
        w=csv.DictWriter(fp,fieldnames=pred[0].keys());w.writeheader();w.writerows(pred)
    result=dict(dataset='OpenNeuro ds002721 1.0.3, Daly et al., University of Reading CC BY 4.0',tree_sha=inventory['sha'],
                trained_on='FACED only; no training, feature choice or thresholds adapted on ds002721.',research_weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                runs=len(files),runs_with_eeg=sum('eeg_md5' in x for x in files),source_people=groups,source_trials=len(all_rows),
                eligible_non_neutral=len(pred),rejections=dict(Counter(r['reason'] or ('neutral_rating' if r['label'] is None else 'scored') for r in all_rows)),
                results=summarize(pred,('selected','history','old','bilateral','history_rff')),
                history_waveforms=sum(r['bilat'] is not None for r in all_rows))
    result['scored_with_known_soundtrack_id']=sum(r['clip']!=-1 for r in pred)
    for metric in result['results']['metrics'].values():
        metric.pop('within_clip_auc',None);metric.pop('clips_with_both',None)
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8');print(result['eligible_non_neutral'],result['results']['metrics']['history'])


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('out',type=Path);run(ap.parse_args().out)
