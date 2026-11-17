"""Recalculate the central paired metrics from exported, out-of-person predictions."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as fh:
        return list(csv.DictReader(fh))


def json_file(path):
    return json.loads(path.read_text(encoding='utf8'))


def binary_score(rows,model,kind):
    y = np.array([int(r['label']) for r in rows])
    p = np.array([float(r[model]) for r in rows])
    pos,neg = y==1,y==0
    if not (pos.any() and neg.any()):
        return np.nan
    if kind=='auc':
        n = pos.sum()
        return (rankdata(p)[pos].sum()-n*(n+1)/2)/(n*neg.sum())
    return ((p[pos]>=.5).mean()+(p[neg]<.5).mean())/2


def subject_macro(rows,key,kind):
    groups = {}
    for row in rows:
        groups.setdefault(row['person'],[]).append(row)
    return float(np.nanmean([binary_score(sub,key,kind) for sub in groups.values()]))


def disjoint(rows,split):
    allocations = {}
    for row in rows:
        p = row['person']
        if p in allocations and allocations[p]!=row[split]:
            raise ValueError(f'{p} appears in multiple {split} groups')
        allocations[p] = row[split]


def verify(root):
    checks = {}
    def check(name,actual,expected):
        if not np.isclose(actual,expected,rtol=0,atol=1e-9):
            raise ValueError(f'{name}: {actual} != {expected}')
        checks[name] = round(float(actual),9)

    for filename in ('mist_matched','mist_pre_task_negative_control','thai','iium'):
        record = json_file(root/'stress'/f'{filename}_results.json')
        rows = record['predictions']
        disjoint(rows,'fold')
        for model,scores in record['metrics'].items():
            check(f'{filename}/{model}',subject_macro(rows,model,'ba'),scores['macro_ba']['value'])

    emotion = json_file(root/'emotion/evaluation.json')
    rows = read_csv(root/'emotion/trial_predictions.csv')
    disjoint([r for r in rows if r['phase']!='unseen_movies_and_people'],'phase')
    for phase in ('initial','replication','pooled_holdout','unseen_movies_and_people'):
        phases = ('initial','replication') if phase=='pooled_holdout' else (phase,)
        sub = [r for r in rows if r['phase'] in phases]
        for key in ('old','selected','history'):
            check(f'FACED/{phase}/{key}',binary_score(sub,key,'ba'),emotion['results'][phase]['metrics'][key]['ba'])

    for exp in (root/'emotion').iterdir():
        if exp.is_dir() and (exp/'predictions.csv').is_file() and (exp/'results.json').is_file():
            record = json_file(exp/'results.json')
            summary = record.get('results',{}).get('metrics',{})
            sub = read_csv(exp/'predictions.csv')
            for key in ('old','selected','history'):
                if key in summary:
                    check(f'{exp.name}/{key}',binary_score(sub,key,'ba'),summary[key]['ba'])

    rows = read_csv(root/'focus/predictions.csv')
    disjoint(rows,'split')
    focus = json_file(root/'focus/results.json')
    for phase in ('development','holdout'):
        sub = [r for r in rows if (r['split']=='holdout')==(phase=='holdout')]
        for key in ('candidate','old_0.25','behavior'):
            check(f'focus/{phase}/{key}',subject_macro(sub,key,'auc'),focus[phase]['scores'][key]['macro_auc'])
    rows = read_csv(root/'focus/project_six_predictions.csv')
    disjoint(rows,'split')
    sub = [r for r in rows if r['split']=='holdout']
    original = json_file(root/'focus/verification.json')['project_six_matched_window']['holdout']['scores']
    check('focus/holdout/project_six',subject_macro(sub,'project_six_quarter','auc'),original['project_six_quarter']['macro_auc'])

    rows = read_csv(root/'lemon/predictions.csv')
    disjoint(rows,'split')
    sub = [r for r in rows if r['split']=='locked']
    lemon = json_file(root/'lemon/evaluation.json')
    for key in ('expanded','three','frozen','demographic'):
        check(f'LEMON/{key}',binary_score(sub,key,'auc'),lemon['test']['models'][key]['auroc'])

    for folder in (root/'nencki',root/'nencki/dense'):
        rows = read_csv(folder/'predictions.csv')
        if len(rows)!=len({r['person'] for r in rows}):
            raise ValueError('Multiple Nencki predictions for a participant')
        for r in rows:
            r['label'] = int(r['self_rating'])>3
        sub = [r for r in rows if int(r['self_rating'])!=3]
        folds = sorted({r['outer_fold'] for r in sub})
        record = json_file(folder/'evaluation.json')
        for model in ('expanded','three','frozen'):
            per_fold = [binary_score([r for r in sub if r['outer_fold']==f],model,'auc') for f in folds]
            check(f'Nencki/{folder.name}/{model}',np.mean(per_fold),record['models'][model]['mean_fold_auc_ba'][0])
    return checks


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result_root',type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.result_root),indent=2))
