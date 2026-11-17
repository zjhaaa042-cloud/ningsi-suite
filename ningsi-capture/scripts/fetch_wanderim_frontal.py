"""Fetch paired raw Fp1/Fp2 pre-probe epochs from Andrillon et al. OSF ey3ca."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import itertools
import json
import re
import threading
import time
from urllib.parse import urlsplit

import numpy as np
import requests
from scipy.io import loadmat


LOCAL = threading.local()


def session():
    if not hasattr(LOCAL, "http"):
        LOCAL.http = requests.Session()
    return LOCAL.http


def request(url, headers=None, stream=False):
    for retry in range(5):
        try:
            r = session().get(url, headers=headers, timeout=(18, 65), stream=stream)
            r.raise_for_status()
            return r
        except requests.RequestException:
            if retry == 4:
                raise
            time.sleep(1.5 ** retry)


def read_index(src, node):
    path = src / f"{node}.json"
    if not path.exists():
        url = f"https://api.osf.io/v2/nodes/{node}/files/osfstorage/?page[size]=100"
        path.write_bytes(request(url).content)
    data = json.loads(path.read_bytes())
    if data["links"]["next"] is not None:
        raise ValueError("Incomplete OSF index")
    return data["data"]


def metadata(entry, src):
    attrs = entry["attributes"]
    path = src / attrs["name"]
    if not path.exists():
        path.write_bytes(request(entry["links"]["download"]).content)
    if hashlib.sha256(path.read_bytes()).hexdigest() != attrs["extra"]["hashes"]["sha256"]:
        raise ValueError(f"Metadata fingerprint mismatch: {path}")
    return path


def make_plan(person, entries, output):
    root = output / "source"
    h, v, b = (root / entries[k]["attributes"]["name"] for k in ("vhdr", "vmrk", "mat"))
    header = h.read_text(encoding="utf-8-sig")
    if "DataOrientation=MULTIPLEXED" not in header or "BinaryFormat=IEEE_FLOAT_32" not in header:
        raise ValueError("Unexpected BrainVision representation")
    rate = int(1e6 / float(re.search(r"(?m)^SamplingInterval=(\d+)", header)[1]))
    channels = int(re.search(r"(?m)^NumberOfChannels=(\d+)", header)[1])
    picks = []
    scales = []
    for electrode in ("Fp1", "Fp2"):
        info = re.search(rf"(?m)^Ch(\d+)={electrode},[^,]*,([\d.]+),[µμ]V", header)
        if info is None:
            raise ValueError(f"Missing {electrode} or unit")
        picks.append(int(info[1])-1)
        scales.append(float(info[2]))
    markers = v.read_text(encoding="utf-8-sig").splitlines()
    at = [int(line.split(",")[2])-1 for line in markers if re.search(r"^Mk\d+=P,P\s+1,",line)]
    probes = loadmat(b,variable_names=["probe_res"])["probe_res"]
    extra = len(at) - len(probes)
    discarded = []
    if 0 < extra <= 2:
        # A participant can have stray pre-run triggers. Accept a match only
        # if exactly one order-preserving alignment fits the independent clock.
        candidates = []
        for omit in itertools.combinations(range(len(at)),extra):
            linked = np.delete(at,omit)
            error = np.diff(linked/rate) - np.diff(probes[:,2])
            if np.max(np.abs(error)) < .1:
                candidates.append((omit,linked.tolist()))
        if len(candidates)==1:
            discarded, at = candidates[0]
    if len(at) != len(probes):
        raise ValueError(f"{person}: {len(at)} triggers vs {len(probes)} probes")
    # Independent clocks cannot drift far enough to misassign a probe.
    drift = np.diff(np.array(at)/rate) - np.diff(probes[:,2])
    if np.max(np.abs(drift)) > 1.:
        raise ValueError(f"{person}: apparent clock discontinuity {max(abs(drift)):.3f}s")
    return dict(person=person, rate=rate, channels=channels, picks=picks, scales=scales, at=at,
                size=entries['eeg']['attributes']['size'], url=entries["eeg"]["links"]["download"],
                eeg_sha=entries["eeg"]["attributes"]["extra"]["hashes"]["sha256"],
                discarded_trigger_ordinals_zero_based=discarded,
                max_interprobe_clock_error_s=float(np.max(np.abs(drift))), behavior=b.name,
                header=h.name, marker=v.name)


def signed_url(plan,old_url=None):
    with plan['lock']:
        if old_url==plan.get('signed'):
            with request(plan['url'],{'Range':'bytes=0-3'},stream=True) as r:
                if r.status_code != 206:
                    raise ValueError("Range unsupported")
                if plan['eeg_sha'] not in urlsplit(r.url).path:
                    raise ValueError("OSF raw-object URI does not match published SHA-256")
                plan['signed'] = r.url
        return plan['signed']


def download_epoch(job):
    plan, i, path = job
    rate, n = plan['rate'], plan['channels']
    start = (plan['at'][i]-8*rate)*4*n
    stop = (plan['at'][i]-rate)*4*n-1
    if start < 0 or stop >= plan['size']:
        raise ValueError("Probe beyond raw recording")
    url = signed_url(plan)
    for attempt in range(3):
        with session().get(url,headers={"Range": f"bytes={start}-{stop}"},stream=True,timeout=(20,70)) as response:
            if response.status_code in (401,403) and attempt<2:
                url=signed_url(plan,url)
                continue
            response.raise_for_status()
            if response.status_code != 206 or response.headers.get('Content-Range') != f"bytes {start}-{stop}/{plan['size']}":
                raise ValueError(f"Unexpected byte range for {plan['person']}/{i}")
            raw = response.content
            break
    data = np.frombuffer(raw, dtype='<f4').reshape(7*rate,n)[:,plan['picks']]
    samples = (data * plan['scales']).T.astype('<f4')
    with path.with_suffix('.part').open('wb') as f:
        np.save(f,samples,allow_pickle=False)
    path.with_suffix('.part').replace(path)


def run(out, workers):
    src = out/'source'
    src.mkdir(exist_ok=True, parents=True)
    raw = read_index(src,'w7gz5')
    behavior = read_index(src,'6g2fh')
    groups = {}
    for row in raw + behavior:
        name = row['attributes']['name']
        match = re.search(r'(?:MWI|_s)(\d{3})',name,re.I)
        if match:
            groups.setdefault(match[1],{})[name.rsplit('.',1)[-1]] = row
    participants = {p:g for p,g in groups.items() if {'mat','vhdr','vmrk','eeg'} <= g.keys()}
    pool = ThreadPoolExecutor(workers)
    for job in as_completed([pool.submit(metadata,g[k],src) for g in participants.values() for k in ('vhdr','vmrk','mat')]):
        job.result()
    plans, errors = [],{}
    for p,g in sorted(participants.items()):
        try:
            plans.append(make_plan(p,g,out))
        except ValueError as e:
            errors[p]=str(e)
    (out/'synchronization.json').write_text(json.dumps(dict(included=plans,excluded=errors),indent=2),encoding='utf8')
    print(f"{len(plans)} synchronized participants; exclusion details {errors}",flush=True)
    def resolve(plan):
        folder = src / ('p'+plan['person'])
        folder.mkdir(exist_ok=True)
        jobs = [(plan,i,folder/f'{i:02}.npy') for i in range(len(plan['at'])) if not (folder/f'{i:02}.npy').is_file()]
        plan['lock'] = threading.Lock()
        return jobs
    jobs = []
    for future in as_completed([pool.submit(resolve,row) for row in plans]):
        jobs.extend(future.result())
    finished, failed = 0, []
    for future in as_completed([pool.submit(download_epoch,j) for j in jobs]):
        try:
            future.result()
        except Exception as e:
            failed.append(type(e).__name__+': '+str(e).split('?')[0][:180])
            if len(failed)<4:
                print(failed[-1],flush=True)
        finished += 1
        if finished%100==0:
            print(f"Fetched {finished-len(failed)} / {len(jobs)} pre-probe pairs, failures {len(failed)}",flush=True)
    pool.shutdown()
    if failed:
        raise RuntimeError(f"{len(failed)} missing; retry command to resume. First error {failed[0]}")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out",type=Path)
    parser.add_argument("--workers",type=int,default=18)
    args=parser.parse_args()
    run(args.out,args.workers)
