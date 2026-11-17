"""Fixed FACED model transfer to the MET personal, per-song SAM ratings."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import struct

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly
from scipy.special import expit

from benchmark_faced_valence import filtered_windows, analyze_window, features, feature_vector, predict, summarize
from fetch_faced_frontal import get
from validate_music_valence_transfer import verify_git
from ningsi.models.affect import load_models
from ningsi.models.human import logits


VERSION = '1.0.4'
RAW = f'https://raw.githubusercontent.com/OpenNeuroDatasets/ds008701/{VERSION}/'
S3 = 'https://s3.amazonaws.com/openneuro.org/ds008701/'


def tsv(raw):
    return list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig')), delimiter='\t'))


def verified(path, tree):
    raw = get(RAW + path)[0]
    verify_git(raw, tree[path])
    return raw


def extract(row, source, tree):
    person, song = row['participant_id'], row['song_id']
    prefix = f'{person}/ses-{song}/'
    stem = prefix + f'eeg/{person}_ses-{song}_task-music_'
    name = stem + 'eeg.set'
    cached = source / f'{person}_{song}.json'
    if cached.exists():
        return json.loads(cached.read_text(encoding='utf8'))
    missing = dict(person=int(person[4:]),clip=int(song[1:]),rating=int(row['Valence']),bilat=None,legacy=[],reference='unavailable')
    if name not in tree:
        return dict(missing,reason='recording_absent_from_release')
    scans = tsv(verified(prefix + f'{person}_ses-{song}_scans.tsv', tree))
    assert len(scans) == 1 and prefix + scans[0]['filename'] == name
    events = tsv(verified(stem + 'events.tsv', tree)) if stem+'events.tsv' in tree else []
    ref = verified(name, tree)
    m = re.search(rb'SHA256E-s(\d+)--([0-9a-f]{64})\.set', ref)
    assert m is not None
    size = int(m[1])
    head = get(S3 + name, (0,183))[0]
    assert head[126:128] == b'IM' and head[172:176] == b'data'
    nc, samples, tag = struct.unpack_from('<III', head, 160)
    dtype, nbytes = struct.unpack_from('<II', head, 176)
    assert dtype == 7 and tag == 262145 and nc*samples*4 == nbytes
    if size < 184 + nbytes:
        full = get(S3+name)[0]
        assert len(full)==size and hashlib.sha256(full).hexdigest()==m[2].decode()
        return dict(missing,reason='release_file_truncated_before_requested_data',expected_file_sha256=m[2].decode())
    tail = get(S3 + name, (184+nbytes, size-1))[0]
    meta = loadmat(io.BytesIO(head[:128]+tail), simplify_cells=True)
    labels = [v['labels'].lower() for v in meta['chanlocs']]
    assert int(meta['nbchan']) == nc and int(meta['srate']) == 1000 and int(meta['pnts']) == samples
    assert labels.count('fp1') == labels.count('fp2') == 1 and samples >= 125000
    assert str(meta['setname']).replace('\\','/').endswith(name)
    row = dict(person=int(person[4:]), clip=int(song[1:]), rating=int(row['Valence']),
               timestamp=scans[0]['acq_time'], legacy=[], bilat=None, reason=None,
               expected_file_sha256=m[2].decode(), metadata_sha256=hashlib.sha256(tail).hexdigest(),
               reference=str(meta.get('ref','unreported')))
    if any(e['trial_type'].lower().startswith('bad') and float(e['onset'])<124 and
           float(e['onset'])+float(e['duration'])>112 for e in events):
        row['reason'] = 'published_bad_annotation'
    else:
        start, end = 184 + 112000*nc*4, 184 + 124000*nc*4-1
        raw = get(S3 + name, (start,end))[0]
        row['range_sha256'] = hashlib.sha256(raw).hexdigest()
        # The .set data matrix stores EEG in microvolts under EEGLAB convention.
        x = np.frombuffer(raw,'<f4').reshape(12000,nc)[:,[labels.index('fp1'),labels.index('fp2')]].T
        assert np.isfinite(x).all()
        x = resample_poly(x.astype(float),1,4,axis=1,padtype='line').astype(np.float32).astype(float)
        rich = []
        for pair in filtered_windows(x,250,ends=np.array([1500,2000,2500,3000])):
            w = analyze_window(pair,250)
            if w.usable:
                row['legacy'].append(feature_vector(w).tolist())
                rich.append(features(pair))
        if rich:
            row['bilat'] = np.median(rich,axis=0).tolist()
        else:
            row['reason'] = 'EEG_QC'
    cached.write_text(json.dumps(row,allow_nan=False),encoding='utf8')
    return row


def run(out):
    out.mkdir(exist_ok=True,parents=True)
    src = out/'source'
    src.mkdir(exist_ok=True)
    tree_file = src/'tree.json'
    if not tree_file.exists():
        tree_file.write_bytes(get(f'https://api.github.com/repos/OpenNeuroDatasets/ds008701/git/trees/{VERSION}?recursive=1')[0])
    inventory = json.loads(tree_file.read_text(encoding='utf8'))
    assert not inventory['truncated']
    tree = {r['path']:r['sha'] for r in inventory['tree'] if r['type']=='blob'}
    survey = verified('ratings.tsv',tree)
    (src/'ratings.tsv').write_bytes(survey)
    ratings = tsv(survey)
    assert len(ratings) == len({(r['participant_id'],r['song_id']) for r in ratings}) == 400
    assert all(0<=int(r['Valence'])<=9 for r in ratings)
    source = []
    with ThreadPoolExecutor(20) as pool:
        jobs = [pool.submit(extract, r, src, tree) for r in ratings]
        for i, job in enumerate(as_completed(jobs),1):
            source.append(job.result())
            if i%40==0:print(f'{i}/400',flush=True)
    rows = []
    order = {(int(r['participant_id'][4:]),int(r['song_id'][1:])):i for i,r in enumerate(ratings)}
    chronology = {}
    for person in sorted({r['person'] for r in source}):
        history = []
        trials = sorted((r for r in source if r['person']==person and 'timestamp' in r),
                        key=lambda r:order[(person,r['clip'])])
        times=[r['timestamp'] for r in trials]
        chronology[person]=dict(adjacent_ties=sum(a==b for a,b in zip(times,times[1:])),
                                adjacent_inversions=sum(a>b for a,b in zip(times,times[1:])))
        for r in trials:
            if r['reason']: continue
            r['prior_n'] = len(history)
            r['prior_sum'] = np.sum(history,axis=0).tolist() if history else None
            history.append(r['bilat'])
            if r['rating'] not in (0,5):
                r['label'] = int(r['rating']>5)
                rows.append(r)
    path = out.parent/'research_weights.json'
    bundle = json.loads(path.read_text(encoding='utf8'))
    cold = [dict(r,prior_n=0,prior_sum=None) for r in rows]
    clips = {k:predict(cold,m) for k,m in bundle['models'].items()}
    clips['history_survey_order']=predict(rows,bundle['models']['history'])
    old = load_models()['valence']['parameters']
    pred = []
    for i,r in enumerate(rows):
        p = {k:float(v[i]) for k,v in clips.items()}
        pred.append(dict(person=r['person'],clip=r['clip'],rating=r['rating'],label=r['label'],
                         old=float(expit(logits(old,np.array(r['legacy']))).mean()),selected=p[bundle['selected']],**p))
    with (out/'predictions.csv').open('w',newline='',encoding='utf8') as fp:
        writer = csv.DictWriter(fp,fieldnames=pred[0]);writer.writeheader();writer.writerows(pred)
    result = dict(source='OpenNeuro ds008701 1.0.4',tree_sha=inventory['sha'],survey_sha256=hashlib.sha256(survey).hexdigest(),
                  research_weights_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),total=len(source),source_people=20,
                  all_ratings=dict(Counter(r['rating'] for r in source)),reference=dict(Counter(r['reference'] for r in source)),
                  rejections=dict(Counter(r['reason'] or {0:'missing_rating',5:'neutral_rating'}.get(r['rating'],'scored') for r in source)),
                  chronology=chronology,history_mode='cold start per file; survey row ordering reported as secondary sensitivity only',
                  results=summarize(pred,('old','selected','bilateral','history','history_rff','history_survey_order')))
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8')
    print(len(rows),result['results']['metrics']['history'])


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('out',type=Path);run(ap.parse_args().out)
