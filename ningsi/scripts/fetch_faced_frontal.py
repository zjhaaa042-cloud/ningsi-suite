"""Fetch Fp1/Fp2 and linked self-ratings from pinned FACED BIDS v1.1.4 using verified byte ranges."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import threading
import time

import numpy as np
import requests
from scipy.signal import resample_poly


RAW = 'https://raw.githubusercontent.com/nemarDatasets/nm000112/v1.1.4/'
DATA = 'https://data.nemar.org/nm000112/v1.1.4/'
LOCAL = threading.local()
ORDER = np.random.default_rng(53141).permutation(123).tolist()
PEOPLE = ORDER[:48]
SECONDS = 12


def get(url, span=None):
    if not hasattr(LOCAL, 'session'):
        LOCAL.session = requests.Session()
    headers = {} if span is None else {'Range': f'bytes={span[0]}-{span[1]}'}
    for attempt in range(5):
        try:
            with LOCAL.session.get(url,headers=headers,timeout=(15,50),stream=True) as r:
                r.raise_for_status()
                if span is not None:
                    if r.status_code != 206 or not r.headers['Content-Range'].startswith(f'bytes {span[0]}-{span[1]}/'):
                        raise ValueError('Server returned wrong range')
                data = r.content
                if span is not None and len(data)!=span[1]-span[0]+1:
                    raise IOError('Incomplete data range')
                return data,r.url
        except (requests.RequestException,OSError):
            if attempt == 4: raise
            time.sleep(attempt+1)


def parse_header(b):
    nch = int(b[252:256]); head = int(b[184:192]); records = int(b[236:244]); period=float(b[244:252]); pos=256
    def col(width):
        nonlocal pos
        data=b[pos:pos+nch*width]; pos+=nch*width
        return [data[i:i+width].decode('latin1').strip() for i in range(0,len(data),width)]
    assert b[:8]==b'\xffBIOSEMI' and head==256*(1+nch) and len(b)>=head
    names=col(16); col(80); units=col(8); pmin=col(8); pmax=col(8); dmin=col(8); dmax=col(8); col(80); counts=list(map(int,col(8)))
    front=[names.index(k) for k in ['Fp1','Fp2']]; rates=[counts[i]/period for i in front]
    assert rates[0]==rates[1] and rates[0] in (250,1000) and records>0
    output=[]
    for i in front:
        unit=units[i].lower().replace('µ','u'); scale_to_uv={'uv':1.,'mv':1000.,'v':1e6}[unit]
        slope=(float(pmax[i])-float(pmin[i]))/(int(dmax[i])-int(dmin[i]))
        output.append((3*sum(counts[:i]),counts[i],slope*scale_to_uv,(float(pmin[i])-slope*int(dmin[i]))*scale_to_uv))
    return head,records,period,int(rates[0]),3*sum(counts),output


def channel_data(buffer,offset,nsamp,stride,count):
    v=np.ndarray((count,nsamp,3),dtype=np.uint8,buffer=buffer,offset=offset,strides=(stride,3,1)).astype(np.int32)
    value=v[...,0] + (v[...,1]<<8) + (v[...,2]<<16)
    return ((value ^ (1<<23))-(1<<23)).reshape(-1)


def metadata(root,pid,manifest):
    sub=f'sub-{pid:03}'; stem=f'{sub}/eeg/{sub}_task-watchingVideoClips_'; out=root/sub;out.mkdir(exist_ok=True)
    for name in ('events.tsv','channels.tsv','eeg.json'):
        target=out/name
        if not target.exists(): target.write_bytes(get(RAW+stem+name)[0])
    bdf=stem+'eeg.bdf'; ref=manifest[bdf]; header,url=get(DATA+bdf,(0,8703)); parsed=parse_header(header)
    head,n,period,rate,stride,info=parsed
    assert head+n*stride==int(ref['size'])
    side=json.loads((out/'eeg.json').read_text(encoding='utf-8-sig')); assert side['SamplingFrequency']==rate
    channels=list(csv.DictReader((out/'channels.tsv').read_text(encoding='utf-8-sig').splitlines(),delimiter='\t'))
    assert [(c['name'],c['type'],float(c['sampling_frequency'])) for c in channels if c['name'] in ('Fp1','Fp2')]==[('Fp1','EEG',rate),('Fp2','EEG',rate)]
    events=list(csv.DictReader((out/'events.tsv').read_text(encoding='utf-8-sig').splitlines(),delimiter='\t'))
    rows=[r for r in events if r['video_index']!='n/a'];assert sorted(int(r['video_index']) for r in rows)==list(range(1,29))
    assert all(0<=float(r['Valence'])<=7 for r in rows) and all(float(r['duration'])>=30 for r in rows)
    meta={'source_sha256_full_bdf':ref['checksum'],'bdf_header_sha256':hashlib.sha256(header[:head]).hexdigest(),'rate':rate,'voltage_transform':info,'events_sha256':hashlib.sha256((out/'events.tsv').read_bytes()).hexdigest()}
    (out/'provenance.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
    return pid,url,parsed,rows


def fetch_trial(root,item,index):
    pid,url,parsed,rows=item; head,n,period,rate,stride,info=parsed;row=rows[index];sub=f'sub-{pid:03}'
    target=root/sub/f'clip-{int(row["video_index"]):02}.npz'
    if target.exists(): return
    # Stop the acquisition 1 s before the participant sees the rating screen.
    end=round((float(row['onset'])+float(row['duration'])-1)*rate); start=end-SECONDS*rate
    spr=round(rate*period); first=start//spr; stop=(end+spr-1)//spr
    assert 0<first<stop<=n
    span=(head+first*stride,head+stop*stride-1); buf,_=get(url,span)
    x=np.stack([(channel_data(buf,offset,count,stride,stop-first)*scale+bias)[start-first*spr:end-first*spr] for offset,count,scale,bias in info]); assert x.shape==(2,SECONDS*rate)
    y=resample_poly(x,250//math.gcd(rate,250),rate//math.gcd(rate,250),axis=1,padtype='line').astype(np.float32) if rate!=250 else x.astype(np.float32)
    tmp=target.with_suffix('.tmp');
    with tmp.open('wb') as stream:
        np.savez_compressed(stream,uv=y,sha256_range=hashlib.sha256(buf).hexdigest(),byte_range=span,sample_range=[start,end],rating=float(row['Valence']),position=index)
    tmp.replace(target)


def main(root,replication=False):
    root.mkdir(parents=True,exist_ok=True); path=root/'manifest.json'
    if not path.exists():path.write_bytes(get(DATA+'manifest.json')[0])
    manifest={r['path']:r for r in json.loads(path.read_text(encoding='utf8'))}
    n=0
    with ThreadPoolExecutor(12) as meta_workers, ThreadPoolExecutor(28) as workers:
        prepared=[meta_workers.submit(metadata,root,p,manifest) for p in (ORDER[48:] if replication else PEOPLE)]
        jobs=[]
        for future in as_completed(prepared):
            item=future.result()
            jobs.extend(workers.submit(fetch_trial,root,item,j) for j in range(28))
        for future in as_completed(jobs):
            future.result();n+=1
            if n%140==0: print(f'{n}/{len(jobs)} trials',flush=True)
    print('Completed',n,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root',type=Path)
    ap.add_argument('--replication',action='store_true',help='Fetch all 75 additional held-out subjects.')
    args=ap.parse_args();main(args.root,args.replication)
