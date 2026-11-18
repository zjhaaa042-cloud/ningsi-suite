"""Compare mean-feature L2 logistic regression with an ordered and shuffled TCN.

This offline experiment reuses the same public ATTLAPSE probes, frozen person
folds and physical-QC windows for every arm. A probe is one labelled example.
It writes research evidence only. No release model or scoring path is changed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np

import attlapse_two_channel_eval as evaluation
from attlapse_two_channel_250 import iter_probes
from attlapse_two_channel_features import BIPOLAR, _flags, _window


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'var/attlapse/two-channel-benchmark'
OUTPUT = ROOT / 'var/tcn-focus-20261118'
WINDOW = 'pre18_2'
GATE = 'physical'
SEEDS = (17, 43, 71)
TCN_SETTINGS = {'epochs': 80, 'channels': 8, 'lr': .005, 'l2': .01}
LR = evaluation.Candidate(GATE, WINDOW, 'bipolar6', 'logistic', .01)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def reference_rows(path):
    grids = {GATE: {}, 'strict': {}}
    with path.open(encoding='utf-8-sig') as stream:
        for line in stream:
            row = json.loads(line)
            if row['window'] != WINDOW or row['quality'] not in grids:
                continue
            key = row['participant'], row['probe_id']
            grid = grids[row['quality']]
            if key in grid:
                raise ValueError(f'Duplicate reference probe: {key}')
            grid[key] = row
    if len(grids[GATE]) != 420 or sorted(grids[GATE]) != sorted(grids['strict']):
        raise ValueError('Expected all 420 public probes in both gates')
    return grids


def prepare(data, path):
    grids = reference_rows(data / 'features250.jsonl')
    grid = grids[GATE]
    keys = sorted(grid)
    indices = {key: i for i, key in enumerate(keys)}
    values = np.zeros((len(keys), 6, len(BIPOLAR)), dtype=np.float64)
    mask = np.zeros(values.shape[:2], dtype=bool)
    strict = np.zeros(len(keys), dtype=bool)
    y = np.full(len(keys), -1, dtype=int)
    seen = set()
    for probe in iter_probes(data):
        key = probe['participant'], probe['probe_id']
        row = grid[key]
        strict_row = grids['strict'][key]
        i = indices[key]
        if key in seen or row['label'] != probe['label'] or strict_row['label'] != probe['label']:
            raise ValueError(f'Inconsistent probe identity: {key}')
        seen.add(key)
        y[i] = probe['label']
        strict[i] = bool(strict_row['features'])
        if not row['features']:
            continue
        rate = probe['rate']
        step, width, context = 2*rate, 4*rate, 6*rate
        first = int(np.ceil((row['window_start_sec']*rate + width)/step))*step
        last = int(np.floor(row['window_end_sec']*rate))
        end_slots = list(range(first, last+1, step))
        if rate != 250 or len(end_slots) != 6 or last >= probe['stop_sample']:
            raise ValueError(f'Unexpected sample alignment: {key}')
        accepted = set(row['accepted_window_end_samples'])
        if not accepted.issubset(end_slots) or len(accepted) != row['windows_usable']:
            raise ValueError(f'QC window alignment changed: {key}')
        for slot, end in enumerate(end_slots):
            if end not in accepted:
                continue
            lo, hi = max(0, end-context)-probe['start_sample'], end-probe['start_sample']
            if not 0 <= lo < hi <= probe['values_uv'].shape[1]:
                raise ValueError(f'Incomplete pre-probe history: {key}')
            features, pair_qc, bipolar_qc = _window(probe['values_uv'][:, lo:hi].astype(float), rate)
            if _flags(pair_qc, bipolar_qc)[GATE]:
                raise ValueError(f'Recomputed window failed the frozen gate: {key} {end}')
            values[i, slot] = [features[name] for name in BIPOLAR]
            mask[i, slot] = True
        archived = [row['features'][name] for name in BIPOLAR]
        np.testing.assert_allclose(values[i, mask[i]].mean(axis=0), archived, rtol=1e-9, atol=1e-10,
                                   err_msg=f'Feature parity failed for {key}')
    if seen != set(keys) or not np.isin(y, (0, 1)).all() or np.any(strict & ~mask.any(axis=1)):
        raise ValueError('Incomplete or inconsistent data coverage')
    np.savez_compressed(path, features=values, mask=mask, strict=strict, label=y,
                        participants=[k[0] for k in keys], probes=[k[1] for k in keys],
                        source_feature_sha256=sha(data/'features250.jsonl'))


def standardize(x, mask, train):
    observed = x[train][mask[train]]
    spread = observed.std(axis=0)
    spread[spread < 1e-9] = 1.
    return np.where(mask[:, :, None], (x-observed.mean(axis=0))/spread, 0.)


def shuffle_valid(x, mask, seed):
    """Permute complete feature vectors per probe; preserve the missing slots."""
    rng = np.random.default_rng(seed)
    result = x.copy()
    for i, valid in enumerate(mask):
        selected = np.flatnonzero(valid)
        result[i, selected] = x[i, rng.permutation(selected)]
    return result


def fit_and_predict(data, x, mask, y, groups, probes, eligible, *, folds_path=None, order_denominator=15):
    from tcn_focus_numpy import fit_tcn

    source_splits = folds_path or data / 'evaluation250/splits.json'
    folds = json.loads(source_splits.read_text(encoding='utf-8'))['outer']
    people = set(groups.tolist())
    seen = np.zeros(len(y), dtype=int)
    lr_scores = np.full(len(y), np.nan)
    tcn = {name: np.full((len(SEEDS), len(y)), np.nan) for name in ('ordered_tcn', 'shuffled_tcn')}
    means = x.sum(axis=1) / np.maximum(mask.sum(axis=1, keepdims=True), 1)
    fit_seconds = []
    parameter_counts = set()
    details = []
    for fold in folds:
        train_people, test_people = set(fold['train']), set(fold['test'])
        if train_people & test_people or train_people | test_people != people:
            raise ValueError('Participant overlap or omissions in fold')
        held = np.isin(groups, fold['test'])
        seen[held] += 1
        train, test = eligible & ~held, eligible & held
        if np.unique(y[train]).size != 2 or not test.any():
            raise ValueError('Fold lacks training classes or eligible test probes')
        model = evaluation.fit(means[train], y[train], groups[train], LR)
        lr_scores[test] = evaluation.predict(model, means[test])
        scaled = standardize(x, mask, train)
        weights = evaluation.training_weights(y[train], groups[train])
        for j, seed in enumerate(SEEDS):
            shuffled = shuffle_valid(scaled, mask, seed+10000)
            for name, features in (('ordered_tcn', scaled), ('shuffled_tcn', shuffled)):
                start = time.perf_counter()
                model = fit_tcn(features[train], mask[train], y[train], weights, seed=seed, **TCN_SETTINGS)
                fit_seconds.append(time.perf_counter()-start)
                parameter_counts.add(int(model.parameter_count))
                tcn[name][j, test] = evaluation.sigmoid(model.predict_logits(features[test], mask[test]))
        print(f"Fold {fold['outer_fold']}: held-out {test.sum()}/{held.sum()} usable probes", flush=True)
        details.append({'fold': fold['outer_fold'], 'held_out_people': fold['test'],
                        'train_probes': int(train.sum()), 'held_out_probes': int(held.sum()),
                        'held_out_usable': int(test.sum())})
    if not np.all(seen == 1) or not np.array_equal(np.isfinite(lr_scores), eligible):
        raise AssertionError('Each participant must receive exactly one held-out prediction per arm')
    for scores in tcn.values():
        if not np.array_equal(np.isfinite(scores), np.broadcast_to(eligible, scores.shape)):
            raise AssertionError('TCN coverage differs from logistic regression')
    order = np.array([int(p.split('-')[-1]) for p in probes]) / order_denominator
    results = {'logistic': lr_scores, 'probe_order': np.where(eligible, order, np.nan)}
    results.update({name: values.mean(axis=0) for name, values in tcn.items()})
    return results, tcn, {'folds': details, 'parameters': sorted(parameter_counts),
                          'fit_seconds_total': sum(fit_seconds), 'splits_sha256': sha(source_splits)}


def evaluate(y, groups, predictions, seeds, strict):
    people = np.unique(groups)
    boot = np.random.default_rng(20601).multinomial(len(people), np.full(len(people), 1/len(people)), size=5000)
    chosen = {name: evaluation.describe(y, prob, groups, boot) for name, prob in predictions.items()}
    differences = {}
    for a, b in (('ordered_tcn','logistic'), ('ordered_tcn','shuffled_tcn'), ('ordered_tcn','probe_order')):
        a_summary, a_boot = chosen[a]
        b_summary, b_boot = chosen[b]
        metric = 'macro_auc'
        differences[a+'_minus_'+b] = {'difference': a_summary[metric]-b_summary[metric],
                                       'paired_ci95': evaluation.ci(a_boot[metric]-b_boot[metric])}
    per_seed = {name: {str(seed): evaluation.describe(y, scores[j], groups, boot)[0]['macro_auc']
                       for j, seed in enumerate(SEEDS)} for name, scores in seeds.items()}
    result = {'scores': {name: summary for name, (summary, _) in chosen.items()},
              'paired_macro_auc': differences, 'individual_seed_macro_auc': per_seed}
    if strict is not None:
        result['strict_subset_of_frozen_predictions'] = {
            name: evaluation.describe(y, np.where(strict, prob, np.nan), groups, boot)[0]
            for name, prob in predictions.items()}
    return result


def report(out, evidence):
    results = evidence['evaluation']
    scores = results['scores']
    lr = scores['logistic']
    table = [('logistic','L2 逻辑回归（六项特征按窗取均值）'), ('ordered_tcn','小型 TCN（按时间顺序）'),
             ('shuffled_tcn','同结构 TCN（随机打乱窗顺序）'), ('probe_order','仅探针序号对照')]
    lines = ['# 双额区脑电 TCN 走神识别对照', '',
             '使用公开 ATTLAPSE 的 Fp1、Fp2，原始 512 Hz 重采样为 250 Hz。28 人、420 次二分类自评探针。',
             f"既有物理质量门槛下可评分 {lr['scored_labelled']}/{lr['total_probes']} 次，{lr['both_classes_scored_participants']} 人有合格的两类样本可计算个体内 AUROC。",
             '按参与者分五折。每折只用训练者学习参数及标准化。所有模型用同一批合格探针，TCN 的顺序对照只打乱有效窗，保留缺失位置。',
             'TCN 各用三个固定初始化训练，表中分数先按探针平均三次预测。', '',
             '| 方法 | 个体内平均 AUROC | AP | 平衡准确率（阈值 0.5） |', '|---|---:|---:|---:|']
    for key, name in table:
        row = scores[key]
        lines.append(f"| {name} | {row['macro_auc']:.3f} | {row['average_precision']:.3f} | {row['balanced_accuracy']:.3f} |")
    lines.extend(['', '个体内 AUROC 的配对差及 95% 参与者重采样区间。'])
    for label, key in [('TCN − 逻辑回归','ordered_tcn_minus_logistic'), ('TCN − 打乱顺序','ordered_tcn_minus_shuffled_tcn'),
                       ('TCN − 仅探针序号','ordered_tcn_minus_probe_order')]:
        row = results['paired_macro_auc'][key]
        lo, hi = row['paired_ci95']
        lines.append(f"- {label} {row['difference']:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    strict = results['strict_subset_of_frozen_predictions']['logistic']
    lines.extend(['', f"更严格门槛只覆盖 {strict['scored_labelled']} 次探针、{strict['both_classes_scored_participants']} 名有两类样本的参与者，严格子集仅作描述。",
                  '这是已有公开数据的探索性对照。探针标签来自当事人自评。既有物理门槛不含眼电和肌电频谱代理否决，不能排除残留伪迹。',
                  '逻辑回归按训练探针均值进行加权标准化。TCN 按训练者合格的单窗特征标准化。',
                  '探针序号是固定排序对照，其 0.5 阈值仅对应协议中点附近。',
                  '本次模型是频带特征序列模型，结果不外推至原始波形网络、目标头环或眨眼处理。', '',
                  '完整数值和逐探针预测见 `metrics.json` 与 `predictions.csv`。', ''])
    (out/'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    data, out = args.data.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    archive = out/'sequences.npz'
    if not archive.exists():
        prepare(data, archive)
    if args.prepare_only:
        print(f'Prepared matched EEG sequences: {archive}')
        return
    if (out/'metrics.json').exists():
        raise FileExistsError('Choose a new output folder for another benchmark run')
    with np.load(archive, allow_pickle=False) as record:
        x, mask, strict, y = (record[name] for name in ('features','mask','strict','label'))
        groups, probes = record['participants'], record['probes']
        reference_hash = record['source_feature_sha256'].item()
    if reference_hash != sha(data/'features250.jsonl'):
        raise ValueError('Previously prepared inputs no longer match the reference data')
    eligible = mask.any(axis=1)
    scores, seed_scores, run = fit_and_predict(data, x, mask, y, groups, probes, eligible)
    evidence = {'spec': 'attlapse-matched-focus-tcn-v1', 'window': WINDOW, 'quality': GATE,
                'data_doi': '10.5281/zenodo.23592484', 'nominal_rate_hz': 250, 'source_rate_hz': 512,
                'features': list(BIPOLAR), 'input_shape': list(x.shape), 'input_sha256': sha(archive),
                'reference_feature_sha256': reference_hash, 'tcn_settings': TCN_SETTINGS,
                'tcn_seeds': list(SEEDS), 'logistic_l2': LR.parameter,
                'numpy_version': np.__version__, 'implementation_sha256': {
                    name: sha(Path(__file__).with_name(name)) for name in
                    ('benchmark_focus_tcn.py', 'tcn_focus_numpy.py')},
                'run': run, 'evaluation': evaluate(y, groups, scores, seed_scores, strict)}
    with (out/'predictions.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        columns = ['participant','probe','label','eligible','strict','valid_windows',*scores]
        writer = csv.writer(stream)
        writer.writerow(columns)
        for i, (person, probe) in enumerate(zip(groups, probes, strict=True)):
            writer.writerow([person,probe,y[i],eligible[i],strict[i],mask[i].sum(),
                             *(p[i] if np.isfinite(p[i]) else '' for p in scores.values())])
    write_json(out/'metrics.json', evidence)
    report(out, evidence)
    print(out/'RESULTS.md', flush=True)


if __name__ == '__main__':
    main()
