"""Extract named frontal recordings from the authors' public Mendeley archives."""
import argparse
import json
import re
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from fetch_new_stress_sources import member_bytes, mendeley_index


def acquire(root):
    def load_item(item):
        dataset, entry = item
        dest = root/dataset/entry["name"].replace("/", "__").replace(" ", "")
        if dest.exists() and dest.stat().st_size == entry["length"]:
            crc=0
            with dest.open("rb") as handle:
                for block in iter(lambda:handle.read(1024*1024),b""):
                    crc=zlib.crc32(block,crc)
            if crc==entry["crc32"]:
                return
        raw = member_bytes(indices[dataset], entry)
        dest.write_bytes(raw)

    indices, queue = {}, []
    for dataset in ("shvym8gxjg", "kt38js3jv7"):
        index_file = root/dataset/"index.json"
        index = json.loads(index_file.read_text()) if index_file.exists() else mendeley_index(dataset, root)
        indices[dataset] = index
        for e in index["files"]:
            thai = dataset == "shvym8gxjg" and re.fullmatch(r"EEG/(AC1|AC2|EO)/\1_Fp[12].csv", e["name"])
            iium = dataset == "kt38js3jv7" and re.fullmatch(r"raw_data/(Stroop_Data|Arithmetic_Data )/(lowlevel|highlevel|natural)-\d+.txt", e["name"]) and not ("Arithmetic" in e["name"] and "natural" in e["name"])
            if thai or iium:
                queue.append((dataset, e))
    with ThreadPoolExecutor(6) as pool:
        list(pool.map(load_item, queue))
    print("CRC-verified selected files",len(queue),flush=True)


def materialize(root):
    out = root.parent/"trials"
    out.mkdir(exist_ok=True)
    manifest=[]
    def save(dataset, person, task, condition, rate, values, files, start=None):
        finite = np.isfinite(values)
        valid = np.flatnonzero(finite.all(axis=0))
        if len(valid) and len(valid) != valid[-1]-valid[0]+1:
            raise ValueError(f"Interior missing values {dataset} {person} {condition}")
        values=values[:,valid].astype("float32")
        name=f"{dataset}_{person:02}_{task}_{condition}.npz"
        np.savez_compressed(out/name, uv=values)
        manifest.append(dict(dataset=dataset,person=person,task=task,condition=condition,
                             rate=rate,samples=len(valid),file=name,source_files=files,start_unix=start))
    for task in ("AC1","AC2","EO"):
        files=[f"EEG__{task}__{task}_Fp{ch}.csv" for ch in (1,2)]
        matrices=[]
        for name in files:
            path=root/"shvym8gxjg"/name
            with path.open("rb") as f:
                ids=list(map(int,f.readline().strip().split(b",")))
                if ids!=list(range(1,30)):
                    raise ValueError(f"Unrecognized participant columns in {path}")
                matrix=np.genfromtxt(f,delimiter=",",dtype=np.float32)
                matrices.append(matrix)
        for i in range(len(ids)):
            save("thai",ids[i],"arithmetic",{"AC1":"low","AC2":"high","EO":"rest"}[task],
                 2000,np.stack([m[:,i] for m in matrices]),files)
    for path in sorted((root/"kt38js3jv7").glob("raw_data__*.txt")):
        m=re.search(r"__(Arithmetic_Data|Stroop_Data)__(lowlevel|highlevel|natural)-(\d+).txt$",path.name)
        if not m: raise ValueError(path)
        task,level,number=m.groups()
        with path.open("rb") as f:
            # Cyton raw CSV: sample-counter, eight EEG channels in the published cap
            # order (Fp1/Fp2 first), auxiliary fields and timestamps. First sample
            # of some recordings contains acquisition startup zeros.
            start=float(f.readline().split(b",")[22])
            f.seek(0)
            arr=np.loadtxt(f,delimiter=",",usecols=(1,2),dtype=np.float64).T
        save("iium",int(number),"arithmetic" if task.startswith("Arithmetic") else "stroop",
             {"lowlevel":"low","highlevel":"high","natural":"rest"}[level],250,arr,[path.name],start)
    for person in {r["person"] for r in manifest if r["dataset"]=="iium"}:
        group=[r for r in manifest if r["dataset"]=="iium" and r["person"]==person]
        prior,=[r for r in group if r["condition"]=="rest"]
        stop=prior["start_unix"]+prior["samples"]/prior["rate"]
        assert all(r["start_unix"]>stop for r in group if r["condition"]!="rest")
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf8")
    print("Frontal recordings",len(manifest))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("--fetch",action="store_true")
    args=p.parse_args()
    if args.fetch:
        acquire(args.root)
    materialize(args.root)
