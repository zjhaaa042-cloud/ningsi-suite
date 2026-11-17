"""Grouped dual-frontal stress-induction benchmarks. No runtime model mutation."""
import argparse
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from scipy.signal import welch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from ningsi.models.human import fit, logits, sigmoid, write_json
from ningsi.models.logistic import roc_auc
from ningsi.models.affect import feature_vector, load_models
from ningsi.signal.window import analyze_window
from compare_frontal_affect import filtered_windows
from prepare_new_stress_eeg import member_bytes, mendeley_index

SEED=20261115
ARMS=("relative","spatial","calibrated","paired_relative_change")


def workbook(path):
    with zipfile.ZipFile(path) as z:
        string_table=ET.fromstring(z.read("xl/sharedStrings.xml"))
        strings=["".join(si.itertext()) for si in string_table]
        sheet=ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
        book=ET.fromstring(z.read("xl/workbook.xml"))
    assert book.find(".//{*}sheet").get("name")=="Non-Normalized"
    rows=[]
    for row in sheet.findall(".//{*}sheetData/{*}row"):
        values={}
        for cell in row:
            val=cell.find("{*}v")
            if val is None: continue
            key=re.match(r"[A-Z]+",cell.get("r"))[0]
            values[key]=strings[int(val.text)] if cell.get("t")=="s" else float(val.text)
        rows.append(values)
    return rows


def mist_data(root, fetch=False):
    folder=root/"source/cyhchpxwps"
    if fetch:
        index=mendeley_index("cyhchpxwps",root/"source")
        for entry in index["files"]:
            if entry["name"].endswith("/EEG.xlsx"):
                stem="control" if "Control" in entry["name"] else "stress"
                (folder/(stem+".xlsx")).write_bytes(member_bytes(index,entry))
    cols=(("E","F"),("M","N"),("U","V"),("AC","AD"),("AK","AL"),("AS","AT"))
    records=[]
    for label,cond in enumerate(("control","stress")):
        book=workbook(folder/(cond+".xlsx"))
        for (left,right),band in zip(cols,("Delta","Theta","Alpha","Beta 1","Beta 2","Gamma")):
            assert book[0][left].startswith(band) and (book[1][left],book[1][right])==("Fp 1","Fp 2")
        level=None
        for row in book[2:]:
            # The source workbook merges the segment label vertically over each
            # 66-participant table; rows beyond the reported tables are blank.
            if "A" in row:
                level=row["A"]
            if level not in {"EO","AC1","AC2","AC3","AC4"} or not str(row.get("B","")).strip():
                continue
            powers=np.array([[row.get(left,np.nan),row.get(right,np.nan)] for left,right in cols])
            if not np.isfinite(powers).all() or (powers<=0).any():
                continue
            # Published samples include six nonoverlapping frequencies (1–45 Hz).
            # Combine beta subbands before relative-power calculation.
            use=powers[[1,2,3]].copy()
            use[2]+=powers[4]
            rel=(use/powers.sum(axis=0)).mean(axis=1)
            records.append(dict(person=int(row["B"]),task=level,label=label,
                                relative=np.log(rel).tolist(),
                                spatial=np.log(use).ravel().tolist(),
                                raw=np.log(use).ravel().tolist()))
    unique={(r["person"],r["task"],r["label"]) for r in records}
    if len(unique)!=len(records): raise ValueError("Multiple power rows for a person/condition/level")
    rest={(r["person"],r["label"]):r for r in records if r["task"]=="EO"}
    for row in records:
        initial=rest.get((row["person"],row["label"]))
        row["calibrated"]=np.subtract(row["spatial"],initial["spatial"]).tolist() if initial else []
        row["paired_relative_change"]=np.subtract(row["relative"],initial["relative"]).tolist() if initial else []
    # Common paired complete-case cohort at each level, ground truth from the manipulated session.
    seen={(r["person"],r["task"],r["label"]) for r in records if r["calibrated"]}
    return [r for r in records if (r["person"],r["task"],1-r["label"]) in seen and r["calibrated"]]


def raw_data(root):
    trials=root/"trials"
    out=defaultdict(list)
    qc=[]
    for item in json.loads((trials/"manifest.json").read_text()):
        uv=np.load(trials/item["file"])["uv"].astype(float)
        rate=item["rate"]
        # Common signal duration and window times in both conditions. No duration
        # or number-of-usable-windows input to either classifier.
        if item["samples"]<rate*10:
            qc.append({**item,"good":0,"attempt":0}); continue
        duration=min(60, item["samples"]//rate)
        packed=item["dataset"]=="thai"
        if packed:
            # These are already screened disjoint 2-s epochs. Never concatenate
            # epochs for filtering or for the live 4-s feature extractor.
            assert item["samples"]%(2*rate)==0
            extent=duration//2*(2*rate)
            pairs=uv[:,:extent].reshape(2,-1,2*rate).transpose(1,0,2)
        else:
            ends=np.arange(8,duration+1,4)*250
            pairs=filtered_windows(uv[:,:duration*rate],rate,ends)
        sparse, spatial=[],[]
        for pair in pairs:
            if not packed:
                w=analyze_window(pair,250)
                if not w.usable: continue
                sparse.append(feature_vector(w).tolist())
            elif np.any(np.ptp(pair,axis=1)<.5):
                continue
            psd_rate=rate if packed else 250
            f,psd=welch(pair,fs=psd_rate,nperseg=psd_rate,noverlap=psd_rate//2,axis=1)
            power=np.stack([np.sum(psd[:,(f>=a)&(f<b)],axis=1) for a,b in ((4,8),(8,13),(13,30))])
            if packed:
                total=psd[:,(f>=1)&(f<45)].sum(axis=1)
                sparse.append(np.log((power/total).mean(axis=1)).tolist())
            spatial.append(np.log(np.maximum(power,1e-12)).ravel().tolist())
        qc.append({**item,"good":len(spatial),"attempt":len(pairs)})
        if not spatial: continue
        frozen=None if packed else float(np.mean(sigmoid(logits(load_models()["stress"]["parameters"],np.array(sparse)))))
        out[item["dataset"]].append(dict(person=item["person"], task=item["task"], condition=item["condition"],
                               label=int(item["condition"]=="high"),relative=np.median(sparse,axis=0).tolist(),
                               frozen_sam40=frozen,raw=np.median(spatial,axis=0).tolist(),spatial=np.median(spatial,axis=0).tolist()))
    write_json(root/"raw_qc.json",qc)
    for name,rows in out.items():
        rest={r["person"]:r for r in rows if r["condition"]=="rest"}
        target=[r for r in rows if r["condition"]!="rest" and r["person"] in rest]
        for row in target:
            row["calibrated"]=np.subtract(row["spatial"],rest[row["person"]]["spatial"]).tolist()
            row["paired_relative_change"]=np.subtract(row["relative"],rest[row["person"]]["relative"]).tolist()
        pairs={(r["person"],r["task"],r["label"]) for r in target}
        out[name]=[r for r in target if (r["person"],r["task"],1-r["label"]) in pairs]
    return out


def train(rows,arm):
    x=np.array([r[arm] for r in rows])
    y=np.array([r["label"] for r in rows])
    people=Counter(r["person"] for r in rows)
    w=np.array([1/people[r["person"]] for r in rows])
    for k in (0,1): w[y==k]/=2*w[y==k].sum()
    if arm=="paired_relative_change":
        # Conditional ranking suppresses between-participant/session differences.
        # Only training participants contribute matched low/high task pairs.
        # At inference only the new person's prior resting EEG and current block
        # are needed. Population midpoint is estimated on the training set.
        groups=defaultdict(dict)
        for i,r in enumerate(rows):
            groups[r["person"],r["task"]][r["label"]]=i
        diffs=np.stack([x[g[1]]-x[g[0]] for g in groups.values()])
        pair_w=np.array([w[g[1]] for g in groups.values()])
        local=fit(np.vstack([diffs,-diffs]),np.r_[np.ones(len(diffs)),np.zeros(len(diffs))],
                  np.r_[pair_w,pair_w],.05)
        local["mean"]=np.average(x,weights=w,axis=0).tolist()
        return local
    return fit(x,y,w,.05)


def predict(rows,params,arm):
    return sigmoid(logits(params,np.array([r[arm] for r in rows]))).tolist()


def summarize(rows,arms):
    people=sorted({r["person"] for r in rows})
    assert all({r["label"] for r in rows if r["person"]==s}=={0,1} for s in people)
    draws=np.random.default_rng(SEED).integers(len(people),size=(10000,len(people)))
    metrics,values={},{}
    for arm in arms:
        per_person=[]
        for s in people:
            rr=[r for r in rows if r["person"]==s]
            neg=[r[arm]<.5 for r in rr if r["label"]==0]
            pos=[r[arm]>=.5 for r in rr if r["label"]==1]
            auc=roc_auc([r["label"] for r in rr],[r[arm] for r in rr])
            per_person.append([.5*(np.mean(pos)+np.mean(neg)),auc])
        v=np.array(per_person)
        def describe(v):
            boot=v[draws].mean(axis=1)
            lo,hi=np.percentile(boot,[2.5,97.5],axis=0)
            return {name: dict(value=float(v[:,i].mean()),ci95=[float(lo[i]),float(hi[i])]) for i,name in enumerate(("macro_ba","within_person_auc"))}
        metrics[arm]=describe(v)
        if arm!=arms[0]:
            metrics[arm]["paired_change"]=describe(v-values[arms[0]])
        values[arm]=v
    return dict(people=len(people),blocks=len(rows),metrics=metrics)


def validate(rows,arms):
    people=sorted({r["person"] for r in rows})
    chunks=np.array_split(np.random.default_rng(SEED).permutation(people),5)
    predictions,folds=[],[]
    for fold,test_ids in enumerate(chunks):
        train_rows=[r for r in rows if r["person"] not in test_ids]
        test_rows=[r for r in rows if r["person"] in test_ids]
        assert not {r["person"] for r in train_rows}&{r["person"] for r in test_rows}
        pp=[dict(person=r["person"],task=r["task"],label=r["label"],fold=fold+1) for r in test_rows]
        fold_weights={}
        for arm in arms:
            fold_weights[arm]=load_models()["stress"]["parameters"] if arm=="frozen_sam40" else train(train_rows,arm)
            scores=[r[arm] for r in test_rows] if arm=="frozen_sam40" else predict(test_rows,fold_weights[arm],arm)
            for p,score in zip(pp,scores):
                p[arm]=score
        folds.append(dict(test_ids=test_ids.tolist(),models=fold_weights,**summarize(pp,arms)))
        predictions.extend(pp)
    return dict(**summarize(predictions,arms),folds=folds,predictions=predictions)


def permutation_test(rows,result,arm,count=399):
    """Refit on each within-person session-label swap, preserving repeated levels."""
    ids=np.array([r["person"] for r in rows])
    unique=np.unique(ids)
    rng=np.random.default_rng(SEED+37)
    flip=np.column_stack([np.zeros(len(unique),dtype=bool),rng.integers(2,size=(len(unique),count))])
    y=np.logical_xor(np.array([r["label"] for r in rows])[:,None],flip[np.searchsorted(unique,ids)])
    pp=np.zeros(y.shape)
    for fold in result["folds"]:
        test=np.isin(ids,fold["test_ids"])
        train_rows=[r for r,keep in zip(rows,~test) if keep]
        xt=np.array([r[arm] for r in train_rows])
        xv=np.array([r[arm] for r,keep in zip(rows,test) if keep])
        counts=Counter(ids[~test])
        w=np.array([1/counts[s] for s in ids[~test]])
        w/=w.sum()
        center=np.average(xt,axis=0,weights=w)
        yy=y[~test]
        if arm=="paired_relative_change":
            matched=defaultdict(dict)
            for i,r in enumerate(train_rows):
                matched[r["person"],r["task"]][r["label"]]=i
            high=np.array([v[1] for v in matched.values()])
            low=np.array([v[0] for v in matched.values()])
            d=xt[high]-xt[low]
            xx=np.vstack([d,-d])
            w=np.tile(w[high],2)
            yy=np.vstack([yy[high],~yy[high]])
        else:
            xx=xt-center
        scale=np.sqrt(np.average(xx**2,axis=0,weights=w))
        scale=np.where(scale<1e-9,1.,scale)
        z=xx/scale
        weights=np.zeros((z.shape[1],yy.shape[1]))
        bias=np.zeros(yy.shape[1])
        for _ in range(900):
            error=(sigmoid(z@weights+bias)-yy)*w[:,None]
            weights-=.15*(z.T@error+.05*weights)
            bias-=.15*error.sum(axis=0)
        pp[test]=(xv-center)/scale@weights+bias
    correct=(pp>=0)==y
    scores=np.stack([correct[ids==s].mean(axis=0) for s in unique]).mean(axis=0)
    observed=result["metrics"][arm]["macro_ba"]["value"]
    assert abs(scores[0]-observed)<1e-9
    return dict(arm=arm,metric="participant_macro_BA",upper_tail_p=float((1+(scores[1:]>=observed-1e-12).sum())/(1+count)),
                random_refits=count,null_q025_q975=np.quantile(scores[1:],[.025,.975]).tolist(),
                observational_scope="conditional on exchangeability of the two sessions/conditions within a participant; fixed task order can violate it")


def transfer(source, target):
    # The two independently recruited Malaysian and Thai data sources share
    # channel names and raw-waveform absolute band features. Keep the Thai
    # 66-person and 29-person sources separate since overlap is undocumented.
    arms=("spatial","calibrated")
    params={k:train(source,k) for k in arms}
    rows=[dict(person=r["person"],task=r["task"],label=r["label"]) for r in target]
    for arm in arms:
        for row,score in zip(rows,predict(target,params[arm],arm)):
            row[arm]=score
    return dict(**summarize(rows,arms),models=params,predictions=rows)


def session_control(phase_result, target):
    estimates=[]
    for fold in phase_result["folds"]:
        test=[r for r in target if r["person"] in fold["test_ids"]]
        p=predict(test,fold["models"]["relative"],"relative")
        estimates.extend(dict(person=r["person"],task=r["task"],label=r["label"],relative=s) for r,s in zip(test,p))
    return dict(**summarize(estimates,("relative",)),predictions=estimates)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("--fetch-mist",action="store_true")
    p.add_argument("--raw",action="store_true")
    p.add_argument("--permutations",action="store_true")
    a=p.parse_args()
    mist=mist_data(a.root,a.fetch_mist)
    data={"mist_matched":[r for r in mist if r["task"]!="EO"],
          "mist_pre_task_negative_control":[r for r in mist if r["task"]=="EO"]}
    if a.raw:
        data.update(raw_data(a.root))
    for case,rows in data.items():
        arms=ARMS if case!="mist_pre_task_negative_control" else ARMS[:2]
        if case=="iium":
            arms=("frozen_sam40",*arms)
        result=validate(rows,arms)
        if a.permutations:
            result["permutation"]=[permutation_test(rows,result,arm) for arm in ("relative","paired_relative_change") if arm in arms]
        write_json(a.root/(case+"_features.json"),rows)
        write_json(a.root/(case+"_results.json"),result)
        print(case,result["people"],result["blocks"],result["metrics"],flush=True)
    if a.raw:
        write_json(a.root/"cross_source_sensitivity.json",
                   {f"{src}_to_{dst}":transfer(data[src],data[dst]) for src,dst in (("thai","iium"),("iium","thai"))})
    phases=("mist_matched","mist_pre_task_negative_control")
    write_json(a.root/"mist_phase_transfer.json",
               {f"{src}_to_{dst}":session_control(json.loads((a.root/(src+"_results.json")).read_text()),data[dst])
                for src,dst in (phases,phases[::-1])})


if __name__=="__main__": main()
