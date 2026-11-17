"""Transfer frozen SAM40 models to independently recruited du Bois participants."""
import argparse
import csv
import hashlib
from math import comb
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import numpy as np
from validate_within_round_stress import read_json, dump, segment_features


def worksheet(path):
    ns = {'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with ZipFile(path) as archive:
        strings = [''.join(cell.itertext()) for cell in
                   ET.fromstring(archive.read('xl/sharedStrings.xml')).findall('m:si', ns)]
        tree = ET.fromstring(archive.read('xl/worksheets/sheet1.xml'))
    rows = []
    for row in tree.findall('.//m:row', ns):
        record = {}
        for cell in row.findall('m:c', ns):
            value = cell.findtext('m:v', None, ns)
            if value is not None:
                col = cell.get('r').rstrip('0123456789')
                record[col] = strings[int(value)] if cell.get('t') == 's' else value
        rows.append(record)
    assert rows[3]['D'] == rows[3]['F'] == 'stress'
    return rows


def blocks(matrix, origin):
    from ningsi.models.affect import feature_vector
    from ningsi.signal.window import analyze_window
    from compare_frontal_affect import filtered_windows

    assert matrix.shape[1] == 33 and origin + 134*250 <= matrix.shape[0]
    features = []
    for sec in (20,55,90):
        start = origin + sec*250
        raw = matrix[start:start+6250, :2].astype('float64').T
        assert raw.shape == (2,6250)
        new,n_clean = segment_features(raw,250)
        old = []
        for window in filtered_windows(raw,250):
            measured = analyze_window(window,250)
            if measured.usable:
                old.append(feature_vector(measured).tolist())
        features.append(dict(start=start, old=old, new=new, n_clean=n_clean))
    return features


def prepare(source, out, legacy_source):
    import h5py
    from contextlib import ExitStack

    hashlist = read_json(source/'manifest.json')
    ratings_file = source/'mood-stress-data.xlsx'
    assert hashlib.sha256(ratings_file.read_bytes()).hexdigest() == hashlist[ratings_file.name]
    subjects = []
    for row in worksheet(ratings_file):
        if row.get('A') in ('MI','NF'):
            arm = row['A']
        if not row.get('A','').startswith('p') or not row['A'][1:].isdigit():
            continue
        person = int(row['A'][1:])
        item = dict(person=person,arm=arm,pre=float(row['D']),post=float(row['F']))
        subjects.append(item)
        names = ['relax1','relax2'] if arm == 'MI' else ['EEG_relax','EEG_game']
        paths = [source/f'EEG_sb{person}_ss1__{n}.mat' for n in names]
        if not all(p.is_file() for p in paths):
            item.update(excluded='missing_complete_files')
            continue
        with ExitStack() as stack:
            datasets = []
            for path in paths:
                assert hashlib.sha256(path.read_bytes()).hexdigest() == hashlist[path.name]
                h = stack.enter_context(h5py.File(path))
                datasets.append(h[next(iter(h))])
            item.update(shape=[d.shape for d in datasets],files=[p.name for p in paths],
                        sha256=[hashlist[p.name] for p in paths])
            if any(d.shape[1] != 33 for d in datasets):
                item.update(excluded='no_Fp1_Fp2_in_8ch_FlexEEG')
                continue
            before,after = datasets
            assert arm == 'NF'
            assert np.all(before[:,-1]==0)
            tr = after[:,-1]
            event = np.flatnonzero((tr!=0) & (tr!=np.r_[0,tr[:-1]]))
            codes = tr[event].astype(int)
            expected = [x for k in range(10) for x in (2*k+1,2*k+2,2*k+1)]
            # In subject 19 the first onset has an anomalous code, as documented.
            # Some files also repeat an onset pulse (e.g. person 23, run 3).
            # Verify the full chronological subsequence and the exact final run.
            observed=iter(codes)
            assert all(any(v==code for v in observed) for code in expected[3:]),(person,codes)
            assert list(codes[-3:])==[19,20,19],(person,codes)
            origin = int(event[-1])+1
            item.update(trigger_onsets=event.tolist(),trigger_codes=codes.tolist(),
                        post_origin=origin,before=blocks(before,0),after=blocks(after,origin))
    old = read_json(legacy_source)['models']['stress']
    data = dict(subjects=subjects,legacy=old['parameters'],legacy_model_id=old['model_id'],
                rating_source_sha256=hashlist[ratings_file.name],
                source='https://doi.org/10.17632/gsxphk87mc.3')
    dump(out/'dubois_features.json',data)
    return data


def cp_interval(k,n):
    def lower(wins):
        if wins == 0:
            return 0.
        low,high = 0.,1.
        for _ in range(70):
            p = (low+high)/2
            tail = sum(comb(n,t)*p**t*(1-p)**(n-t) for t in range(wins,n+1))
            if tail < .025:
                low = p
            else:
                high = p
        return (low+high)/2
    return [lower(k),1-lower(n-k)]


def evaluate(data, out):
    models = read_json(out/'models.json')[-1]
    w = np.array(models['models']['within_round']['beta'])
    old = data['legacy']
    records = []
    for item in data['subjects']:
        row = dict(person=item['person'],arm=item['arm'],rating_before=item['pre'],
                   rating_after=item['post'],truth_delta=item['post']-item['pre'],
                   exclusion=item.get('excluded',''),pre_clean_blocks=0,post_clean_blocks=0,
                   legacy_delta=None,optimized_delta=None)
        records.append(row)
        if row['exclusion']:
            continue
        periods = []
        for key in ('before','after'):
            scores = []
            for block in item[key]:
                if not block['old'] or not block['new']:
                    continue
                z=(np.asarray(block['old'])-old['mean'])/old['scale']
                p=1/(1+np.exp(-(z@old['weights']+old['bias'])))
                scores.append([float(p.mean()),float(w@block['new'])])
            periods.append(scores)
        row['pre_clean_blocks'],row['post_clean_blocks'] = map(len,periods)
        if not all(periods):
            row['exclusion']='no_common_QC_pair'
            continue
        old_delta,new_delta = np.median(periods[1],axis=0)-np.median(periods[0],axis=0)
        row.update(legacy_delta=float(old_delta),optimized_delta=float(new_delta))
        if row['truth_delta']==0:
            row['exclusion']='tied_self_ratings'
    with (out/'dubois_predictions.csv').open('w',encoding='utf8',newline='') as fp:
        writer = csv.DictWriter(fp,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    elig = [r for r in records if not r['exclusion']]
    true = np.sign([r['truth_delta'] for r in elig])
    result = dict(participants=len(records), eligible=len(elig),
                  participants_with_paired_Fp1_Fp2=sum('before' in r for r in data['subjects']),
                  increasing=int(sum(true>0)),decreasing=int(sum(true<0)),metrics={})
    for method in ('legacy','optimized','always_lower','always_higher'):
        predict = (np.sign([r[method+'_delta'] for r in elig]) if method in ('legacy','optimized')
                   else np.full(len(elig),-1 if method=='always_lower' else 1))
        assert not np.any(predict==0), 'Exact predicted tie requires fractional scoring'
        k = int(sum(predict==true))
        n = len(elig)
        result['metrics'][method] = dict(correct=k,n=n,accuracy=k/n if n else None,
               ci95=cp_interval(k,n) if n else None,balanced_accuracy=float(np.mean(
                   [np.mean(predict[true==c]==c) for c in (-1,1)])) if set(true)=={-1,1} else None,
               p_above_chance=sum(comb(n,i) for i in range(k,n+1))/2**n if n else None)
    if elig:
        legacy=np.sign([r['legacy_delta'] for r in elig])==true
        optimized=np.sign([r['optimized_delta'] for r in elig])==true
        gains=int(sum(optimized & ~legacy)); losses=int(sum(legacy & ~optimized))
        n=gains+losses
        result['paired']=dict(gains=gains,losses=losses,delta=(gains-losses)/len(true),
                              exact_p_greater=sum(comb(n,k) for k in range(gains,n+1))/2**n)
    dump(out/'dubois_evaluation.json',result)
    print(result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--legacy',type=Path)
    options=parser.parse_args()
    cache=options.out/'dubois_features.json'
    data=read_json(cache) if cache.exists() else prepare(options.source,options.out,options.legacy)
    evaluate(data,options.out)
