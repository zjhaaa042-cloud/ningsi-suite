"""Extract pre-specified eyes-open Fp1/Fp2 epochs in public OpenNeuro ds004621."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import requests
from fetch_lemon_bipolar import get, resample_epoch


BASE = 'https://s3.amazonaws.com/openneuro.org/ds004621/'
PIN = 'https://raw.githubusercontent.com/OpenNeuroDatasets/ds004621/1.0.4/'
OFFSETS = (20, 75, 130, 185, 240)


def table(raw):
    try:
        decoded = raw.decode('utf-8-sig')
    except UnicodeDecodeError:
        decoded = raw.decode('latin1')
    return list(csv.DictReader(decoded.splitlines(), delimiter='\t'))


def extract(out, person, offsets):
    target = out/'raw_pairs_dc_corrected'/f'{person}.npz'
    if target.exists():
        return 'ok'
    prefix = f'{person}/eeg/{person}_task-rest_'
    try:
        h = get(BASE+prefix+'eeg.vhdr').content
        events = get(PIN+prefix+'events.tsv').content
        channels = table(get(PIN+prefix+'channels.tsv').content)
    except requests.HTTPError as exc:
        if exc.response.status_code==404:
            return 'no_rest_record'
        raise
    header = h.decode('latin1')
    if hashlib.sha256(h).hexdigest().encode() not in get(PIN+prefix+'eeg.vhdr').content:
        raise ValueError(f'{person}: header differs from pinned release')
    info = dict(re.findall(r'(?m)^(\w+)=(.+)$',header))
    if not (info['DataOrientation'].strip()=='VECTORIZED' and info['BinaryFormat'].strip()=='IEEE_FLOAT_32'
            and int(info['SamplingInterval'])==1000 and int(info['NumberOfChannels'])==len(channels)):
        raise ValueError(f'{person}: unexpected EEG encoding')
    times = table(events)
    if [x['event_type'] for x in times[:2]] != ['S 12','S 10']:
        return 'first_block_markers_ambiguous'
    start = round(1000*float(times[0]['onset']))
    stop = round(1000*float(times[1]['onset']))
    if start+1000*(offsets[-1]+12)>stop:
        return 'first_rest_too_short'
    size = int(info['DataPoints'])*len(channels)*4
    pointer = get(PIN+prefix+'eeg.eeg').content
    expected = re.search(rb'SHA256E-s(\d+)--([a-f0-9]{64})\.eeg',pointer)
    if expected is None or int(expected[1])!=size:
        raise ValueError(f'{person}: pinned waveform size differs from the header')
    leads, provenance = [], []
    etag = None
    for lead in ('Fp1','Fp2'):
        indexes = [i for i,row in enumerate(channels) if row['name']==lead and row['units'] in ('µV','μV','uV') and row['type']=='EEG']
        if len(indexes)!=1:
            raise ValueError(f'{person}: ambiguous lead {lead}')
        index = indexes[0]
        definition = info[f'Ch{index+1}'].split(',')
        if definition[0]!=lead:
            raise ValueError(f'{person}: channel order mismatch')
        resolution = float(definition[2].strip() or 1)
        origin = index*int(info['DataPoints'])+start+1000*offsets[0]
        length = 1000*(offsets[-1]-offsets[0]+12)
        lo, hi = 4*origin, 4*(origin+length)-1
        headers = {'Range':f'bytes={lo}-{hi}'}
        if etag:
            headers['If-Match'] = etag
        response = get(BASE+prefix+'eeg.eeg',headers=headers)
        if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {lo}-{hi}/{size}' or len(response.content)!=4*length:
            raise ValueError(f'{person}: invalid EEG range')
        etag = response.headers['ETag']
        raw = np.frombuffer(response.content,dtype='<f4')*resolution
        leads.append([resample_epoch(raw[(s-offsets[0])*1000:(s-offsets[0]+12)*1000],4) for s in offsets])
        provenance.append(dict(lead=lead,index=index,range=[lo,hi],resolution=resolution,sha256=hashlib.sha256(response.content).hexdigest()))
    src = dict(url=BASE+prefix+'eeg.eeg',etag=etag,full_source_sha256_as_published=expected[2].decode(),header_sha256=hashlib.sha256(h).hexdigest(),
               event_sha256=hashlib.sha256(events).hexdigest(),start_stop=[start,stop],offsets=offsets,leads=provenance)
    with target.with_suffix('.part').open('wb') as f:
        np.savez_compressed(f,eeg_uv=np.swapaxes(leads,0,1),provenance=json.dumps(src))
    target.with_suffix('.part').replace(target)
    return 'ok'


def main(out, dense=False):
    if dense:
        out = out/'dense'
    offsets = list(range(20,241,12)) if dense else OFFSETS
    (out/'raw_pairs_dc_corrected').mkdir(parents=True,exist_ok=True)
    scores = {r['participant_id']:int(r['stress']) for r in table(get(PIN+'participants.tsv').content)}
    if any(s not in range(1,6) for s in scores.values()):
        raise ValueError('Unexpected self-rating scale')
    (out/'stress.json').write_text(json.dumps(scores,indent=2),encoding='utf8')
    with ThreadPoolExecutor(max_workers=8) as pool:
        log = dict(zip(scores,pool.map(lambda p:extract(out,p,offsets),scores)))
    (out/'fetch_audit.json').write_text(json.dumps(log,indent=2),encoding='utf8')
    print('Available',sum(v=='ok' for v in log.values()),'of',len(scores))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    parser.add_argument('--dense',action='store_true',help='Check original quality threshold on 19 consecutive epochs')
    args = parser.parse_args()
    main(args.out,args.dense)
