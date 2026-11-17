"""Test fixed SART time/behavior predictors from the control group on the separate ATTLAPSE P cohort."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from attlapse_artifact_diagnostic import CONFIG, inputs, paired_diff
from attlapse_data import PROBE_TRIALS, archive_index, check_bytes
from attlapse_two_channel_eval import SEED, describe, fit, predict
from validate_attention_fusion import behavior_before_probe


def evaluate(root, out):
    index = {f.filename: f for f in archive_index(root, online=False)}
    manifest = json.loads((root / 'cohort_evidence.json').read_text(encoding='utf8'))
    cohorts = manifest['group_statistics']
    sets = [[f'sub-{p}' for p in cohorts[key]['ids']] for key in ('subs_t', 'subs_p')]
    if len(sets[0]) != 28 or len(sets[1]) != 28 or set(sets[0]) & set(sets[1]):
        raise ValueError('Invalid participant lists')
    keys, _, known_y, _, spectrum, _ = inputs(root / 'two-channel-benchmark/features250.jsonl')
    known = {key: (int(y), bool(np.isfinite(v).all())) for key, y, v in zip(keys, known_y, spectrum)}
    rows, inputs_hash = [], {}
    for group, people in zip(('training', 'transfer'), sets):
        for person in people:
            base = root / 'labels' / person
            source = {}
            for suffix in ('beh.json', 'beh.tsv'):
                filename = f'{person}_task-sart_{suffix}'
                raw = (base / filename).read_bytes()
                entry = index[f'BIDS/sourcedata/{person}/beh/{filename}']
                check_bytes(entry, raw)
                inputs_hash[filename] = hashlib.sha256(raw).hexdigest()
                source[suffix] = raw.decode('utf-8-sig')
            definition = json.loads(source['beh.json'])['att_state_score']
            if any(term not in definition for term in ('on_task (< 43)', 'deliberate mind-wandering (43-52)', 'spontaneous mind-wandering (> 52)')):
                raise ValueError(f'Unexpected attention rating definition for {person}')
            trials = list(csv.DictReader(source['beh.tsv'].splitlines(), delimiter='\t'))
            by_trial = {int(row['trial_number']): row for row in trials}
            if len(by_trial) != len(trials) or list(by_trial) not in (list(range(1,811)), list(range(2,811))):
                raise ValueError(f'Unexpected trial numbers for {person}')
            for ordinal, number in enumerate(PROBE_TRIALS, 1):
                text = by_trial[number]['att_state_score']
                rating = float(text) if text not in ('', 'n/a', 'NaN') else np.nan
                label = int(rating >= 43) if 0 < rating <= 100 else -1
                key = (person, f'probe-{ordinal:02}')
                features = behavior_before_probe(by_trial, number)
                if group == 'training' and known[key][0] != label:
                    raise ValueError(f'Training labels do not match previous benchmark at {key}')
                include = np.isfinite(features).all() and label >= 0
                if group == 'training':
                    include = include and known[key][1]
                rows.append((group, *key, label, include, ordinal, *features))

    x = np.array([r[5:] for r in rows], dtype=float)
    y = np.array([r[3] for r in rows])
    subjects = np.array([r[1] for r in rows])
    train = np.array([r[0] == 'training' and r[4] for r in rows])
    test = np.array([r[0] == 'transfer' for r in rows])
    eligible = np.array([r[4] for r in rows])[test]
    boot = np.random.default_rng(SEED).multinomial(28, np.full(28, 1/28), size=2000)
    scores, results, weights = {}, {}, {}
    for name, design in {'time': x[:, :1], 'behavior': x[:, 1:], 'time_behavior': x}.items():
        params = fit(design[train], y[train], subjects[train], CONFIG)
        scores[name] = predict(params, np.where(eligible[:, None], design[test], np.nan))
        metric, _ = describe(y[test], scores[name], subjects[test], boot)
        results[name] = {key: metric[key] for key in ('macro_auc','balanced_accuracy','both_classes_scored_participants','scored_labelled','probe_coverage')}
        results[name]['macro_auc_ci95'] = metric['ci95']['macro_auc']
        weights[name] = params
    comparison = paired_diff(y[test], subjects[test], scores['time_behavior'], scores['time'], boot)
    report = dict(source='ATTLAPSE 2.0.1, doi:10.5281/zenodo.23592484', train_people=sets[0], test_people=sets[1],
                  training_probes=int(train.sum()), test_probes=int(test.sum()), test_eligible=int(eligible.sum()),
                  models=results, time_behavior_minus_time=comparison, weights=weights, inputs_sha256=inputs_hash,
                  design='Fit once using the original eligible controls, evaluate the complete separate P cohort. Same task and acquisition site. No EEG input in this transfer check. No P subject fitted, no P threshold selection. Paired 2000 participant bootstrap; threshold 0.5.')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'behavior_transfer.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf8')
    with (out / 'behavior_transfer_predictions.csv').open('w', encoding='utf8', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['person','probe','label','time','behavior','time_behavior'])
        for i, row in enumerate(r for r in rows if r[0] == 'transfer'):
            writer.writerow([row[1], row[2], row[3], *[s[i] if np.isfinite(s[i]) else '' for s in scores.values()]])
    return results, comparison


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    print(evaluate(args.root, args.out))
