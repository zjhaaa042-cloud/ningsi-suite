"""Fetch bounded segments of original two-electrode LEMON recordings from the INDI public mirror."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import threading
import time
from xml.etree import ElementTree

import numpy as np
import requests
from scipy.signal import resample_poly


HOST = 'https://fcp-indi.s3.amazonaws.com/'
PROJECT = 'data/Projects/INDI/MPI-LEMON/'
RAW = PROJECT+'EEG_MPILMBB_LEMON/EEG_Raw_BIDS_ID/'
QUESTION = PROJECT+'Behavioural_Data_MPILMBB_LEMON/Emotion_and_Personality_Test_Battery_LEMON/'
AGE = PROJECT+'Behavioural_Data_MPILMBB_LEMON/META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv'
local = threading.local()


def resample_epoch(values, ratio):
    """Remove arbitrary electrode DC before anti-alias filtering at epoch edges."""
    x = np.asarray(values,dtype=np.float64)
    return resample_poly(x-x.mean(axis=-1,keepdims=True),1,ratio,axis=-1,padtype='line')


def get(url, **kwargs):
    if not hasattr(local, 'http'):
        local.http = requests.Session()
    for attempt in range(4):
        try:
            r = local.http.get(url, timeout=(15,60), **kwargs)
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(1+attempt)


def entries(prefix):
    params = {'list-type':2,'prefix':prefix}
    found = {}
    while True:
        xml = ElementTree.fromstring(get(HOST,params=params).content)
        for entry in xml.findall('{*}Contents'):
            found[entry.findtext('{*}Key')] = (int(entry.findtext('{*}Size')),entry.findtext('{*}ETag'))
        token = xml.findtext('{*}NextContinuationToken')
        if not token:
            return found
        params['continuation-token'] = token


def paired_record(keyset, out, person):
    target = out/'raw_pairs_dc_corrected'/f'{person}.npz'
    if target.exists():
        return
    files = {ext:[k for k in keyset if '/'+person+'/' in k and k.lower().endswith('.'+ext)] for ext in ('eeg','vhdr','vmrk')}
    if any(len(group)!=1 for group in files.values()):
        return 'missing_or_ambiguous_trio'
    keys = {ext:names[0] for ext,names in files.items()}
    header, marker = [get(HOST+keys[ext]).content for ext in ('vhdr','vmrk')]
    h = header.decode('utf-8-sig', errors='replace')
    def value(key):
        return re.search('(?m)^'+key+'=(.+)$',h).group(1).strip()
    if value('DataOrientation') != 'MULTIPLEXED' or value('DataFormat')!='BINARY':
        return 'unsupported_orientation'
    dtype = {'INT_16':'<i2','IEEE_FLOAT_32':'<f4'}.get(value('BinaryFormat'))
    rate = int(round(1e6/float(value('SamplingInterval'))))
    if dtype is None or rate%250 or rate<250:
        return 'unsupported_encoding_or_rate'
    n = int(value('NumberOfChannels'))
    channels = {fields[0].lower():(int(i)-1,float(fields[2]),fields[3].strip()) for i,line in
                re.findall(r'(?m)^Ch(\d+)=(.+)$',h) if len(fields:=line.split(','))>=4}
    if any(c not in channels or channels[c][2] not in ('µV','μV','uV') for c in ('fp1','fp2')):
        return 'missing_calibrated_front_lead'
    m = [(tag,int(pos)-1) for tag,pos in re.findall(r'(?m)^Mk\d+=Stimulus,S\s*(200|210),(\d+),', marker.decode('utf8',errors='replace'))]
    bins = []
    for tag, pos in m:
        if bins and tag == bins[-1][0]:
            continue
        bins.append((tag,pos))
    data_len, etag = keyset[keys['eeg']]
    stride = np.dtype(dtype).itemsize * n
    eeg, spans = [], []
    for tag in ('200','210'):
        blocks = [(start, bins[i+1][1] if i+1<len(bins) else data_len//stride) for i,(code,start) in enumerate(bins) if code==tag]
        if len(blocks)<2:
            return 'missing_second_condition_block'
        start, stop = blocks[1]
        first_sample = start+10*rate
        end_sample = first_sample+12*rate
        if end_sample>stop or first_sample<0 or end_sample*stride>data_len:
            return 'insufficient_block_length'
        a,b = first_sample*stride,end_sample*stride-1
        r = get(HOST+keys['eeg'],headers={'Range':f'bytes={a}-{b}','If-Match':etag})
        if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {a}-{b}/{data_len}' or len(r.content)!=b-a+1:
            raise ValueError(f'Incorrect byte interval for {person}')
        raw = np.frombuffer(r.content,dtype=dtype).reshape(-1,n)
        pair = np.array([raw[:,channels[c][0]].astype(float)*channels[c][1] for c in ('fp1','fp2')])
        eeg.append(resample_epoch(pair,rate//250))
        spans.append([first_sample,end_sample,hashlib.sha256(r.content).hexdigest()])
    provenance = dict(source=keys['eeg'],etag=etag,rate=rate,header_sha256=hashlib.sha256(header).hexdigest(),
                      markers_sha256=hashlib.sha256(marker).hexdigest(),spans=spans,channels={c:channels[c] for c in ('fp1','fp2')})
    with target.with_suffix('.part').open('wb') as stream:
        np.savez_compressed(stream,eeg_uv=np.array(eeg),provenance=json.dumps(provenance))
    target.with_suffix('.part').replace(target)


def main(out):
    (out/'raw_pairs_dc_corrected').mkdir(parents=True,exist_ok=True)
    for name, key in [('PSQ.csv',QUESTION+'PSQ.csv'),('PSQ_info.txt',QUESTION+'PSQ_info.txt')]:
        path = out/name
        if not path.exists():
            path.write_bytes(get(HOST+key).content)
    demo=out/'demography.csv'
    if not demo.exists():
        retain_demography(get(HOST+AGE).content,demo)
    if 'Last 2 years' not in (out/'PSQ_info.txt').read_text(encoding='utf8',errors='replace'):
        raise ValueError('Unrecognized PSQ reference period')
    rows = list(csv.DictReader(io.StringIO((out/'PSQ.csv').read_text(encoding='utf-8-sig'))))
    if len(rows)!=len({r[''] for r in rows}):
        raise ValueError('Duplicate PSQ participant')
    people = sorted(r[''] for r in rows if r['PSQ_OverallScore'].strip() not in ('','NA','NaN') and 0<=float(r['PSQ_OverallScore'])<=100)
    keyset = entries(RAW)
    def one(person):
        try:
            return person, paired_record(keyset,out,person) or 'ok'
        except requests.RequestException as exc:
            return person,'network_error: '+str(exc)
    audit = {}
    with ThreadPoolExecutor(12) as pool:
        for i,(p,status) in enumerate(pool.map(one,people),1):
            audit[p] = status
            if i%25==0:
                print(f'{i}/{len(people)} participants checked',flush=True)
    (out/'fetch_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
    print('Paired EEG and PSQ:',list(audit.values()).count('ok'),'of',len(audit),flush=True)


def retain_demography(raw,path):
    rows=csv.reader(io.StringIO(raw.decode('utf-8-sig')))
    header=next(rows)
    fields=[0,header.index('Age'),header.index('Gender_ 1=female_2=male')]
    with path.open('w',newline='',encoding='utf8') as dest:
        writer=csv.writer(dest)
        writer.writerow([header[i] for i in fields])
        writer.writerows([row[i] for i in fields] for row in rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    main(parser.parse_args().out)
