"""Participant-independent, paired self-rated valence tests and sensor ablations for EmoEEG-MC."""
import argparse
import ast
from collections import Counter, defaultdict
import csv
from hashlib import sha256
from io import StringIO
import json
import re
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import expit
from scipy.stats import rankdata

from compare_frontal_affect import filtered_windows, predict_fold
from ningsi.models.affect import feature_vector
from ningsi.models.human import fit, logits, sigmoid, write_json
from ningsi.signal.window import analyze_window
from train_public_affect import RATE, SORTED_CLIPS, video_orders


SEED = 20261115
METHODS = ('absolute', 'prior_rank')
MONTAGES = ('fp', 'all64')


def entropy_vector(de, montage):
    kept = de[[0, 2]] if montage == 'fp' else de
    # The same transform for both montages. Keep absolute values, remove each
    # electrode's broadband shift, then remove the scalp-wide mean by band.
    x = np.median(kept, axis=1)
    return np.concatenate([x.ravel(), (x - x.mean(1, keepdims=True)).ravel(),
                           (x - x.mean(0, keepdims=True)).ravel()])


def score_exported_entropy(model, current, prior_sum, previous):
    """Score the current trial, before adding it to this person's label-free EEG history."""
    x = np.asarray(current)
    if model['method']=='prior_rank':
        n=model['history_pseudotrials']
        x=x-(np.asarray(prior_sum)+n*np.asarray(model['population']))/(previous+n)
    elif model['method']!='absolute':
        raise ValueError('Unknown entropy estimator')
    z=(x-np.asarray(model['mean']))/np.asarray(model['scale'])
    return float(expit(z@np.asarray(model['weights'])+model['bias']))


def extract(cache, topography):
    source = json.loads((cache / 'evaluation/trials.json').read_text('utf8'))['valence']
    lookup = {(row['person'], row['trial']): row for row in source}
    orders = video_orders((cache / 'EmoEEG-readme.md').read_text('utf8'))
    rows, metadata, source_hash = [], defaultdict(list), {}
    ids_by_person = {}
    emotions = ['joy', 'ins', 'ten', 'sad', 'fear', 'dis']
    for person, order in sorted(orders.items()):
        with np.load(cache / 'emoeeg' / f'sub-{person:02d}.npz') as pair_file:
            pairs = pair_file['channels']
            behavior = list(csv.DictReader(StringIO(str(pair_file['ratings'])), delimiter='\t'))
        ids_by_person[person] = [int(float(r['video_name'].strip('[]'))) for r in behavior]
        with np.load(topography / f'sub-{person:02d}.npz', allow_pickle=False) as data:
            full = data['de'].reshape(64, 21, 30, 5)
            source_hash[str(person)] = str(data['sha256'])
        assert np.isfinite(full).all()
        for i, rating in enumerate(behavior):
            for mode, clip in [('block_pairs', order[(i // 6) * 3 + i % 3]),
                               ('separate_halves', order[i % 21]), ('reordered_halves', SORTED_CLIPS[i % 21])]:
                family = clip[:-1]
                if family in emotions:
                    strengths = [float(rating[f'score_{j+1}']) for j in range(6)]
                    metadata[mode].append((int(np.argmax(strengths)) == emotions.index(family),
                                           family in emotions[:3], float(rating['score_8'])))
        history = {m: [] for m in MONTAGES}
        for position, clip in enumerate(order):
            old = lookup[person, clip]
            first = (position // 3) * 6 + position % 3
            ratings = [float(behavior[i]['score_8']) for i in (first, first + 3)]
            label = 1 if min(ratings) > 3.5 else 0 if max(ratings) < 3.5 else None
            assert ratings == old['rating'] and label == old['label']
            ix = SORTED_CLIPS.index(clip)
            values, intervals = [], []
            for j, pair in enumerate(filtered_windows(pairs[:, ix], 200)):
                w = analyze_window(pair, RATE)
                if w.usable:
                    values.append(feature_vector(w))
                    end = 6 * (j + 1)
                    intervals.extend(range(end - 4, end))
            assert len(values) == len(old['windows'])
            if not intervals:
                continue
            assert np.allclose(values, old['windows'], atol=1e-8, rtol=1e-8)
            row = dict(person=person, trial=clip, label=label, position=position, x=np.array(values),
                       previous=len(history['fp']))
            # History includes every eligible preceding trial, even without a class label.
            # Nothing from this trial or a later trial is included in its own reference.
            for m in MONTAGES:
                x = entropy_vector(full[:, ix][:, intervals], m)
                row[m] = x
                row[m + '_prior_sum'] = np.sum(history[m], axis=0) if history[m] else np.zeros_like(x)
                history[m].append(x)
            row['pair_indices'] = (first, first + 3)
            row['pair_ratings'] = ratings
            rows.append(row)
    meta = {}
    for mode, values in metadata.items():
        a = np.array(values)
        meta[mode] = dict(expected_emotion_is_argmax=float(a[:, 0].mean()),
                          intended_vs_self_report_valence_r=float(np.corrcoef(a[:, 1], a[:, 2])[0, 1]),
                          rows=len(a))
    meta['all_video_trial_ratings'] = dict(Counter(str(r['label']) for r in source))
    meta['ambiguous_straddling_no_midpoint'] = sum(min(r['rating']) < 3.5 < max(r['rating']) for r in source)
    meta['one_or_more_midpoints'] = sum(3.5 in r['rating'] for r in source)
    meta['author_de_sha256'] = source_hash
    # Verify every presentation independently using stimulus IDs; learn no mapping
    # from ratings or EEG. A single complete participant supplies the codebook,
    # then the other people with 42 distinct IDs check it independently.
    ids = ids_by_person[25]
    code = {(v-1)%21+1:orders[25][j//6*3+j%3] for j,v in enumerate(ids)}
    assert len(code) == 21 and set(code.values()) == set(SORTED_CLIPS)
    matches = {}
    for person, ids in ids_by_person.items():
        if sorted(ids) == list(range(1,43)):
            matches[person] = sum(code[(v-1)%21+1]==orders[person][j//6*3+j%3] for j,v in enumerate(ids))
    readme = (cache/'EmoEEG-readme.md').read_text('utf8')
    sub54_orders = {m:ast.literal_eval(re.search(r'sub54 '+m+r' sequence\*\*:.*?`(\[.*?\])`', readme, re.S).group(1))
                    for m in ('Imagery','Video')}
    with np.load(cache / 'emoeeg/sub-54.npz') as src:
        b54 = list(csv.DictReader(StringIO(str(src['ratings'])),delimiter='\t'))
    ids54 = [int(float(r['video_name'].strip('[]'))) for r in b54]
    seq = lambda ids, high:[code[(v-1)%21+1] for v in ids if (v>21)==high]
    cross = {f'beh53_{high}_vs_readme54_{m}':sum(a==b for a,b in zip(seq(ids_by_person[53],high),sub54_orders[m]))
             for high in (False,True) for m in sub54_orders}
    cross['beh54_vs_readme53'] = sum(code[(v-1)%21+1]==orders[53][j//6*3+j%3] for j,v in enumerate(ids54))
    assert set(p for p,n in matches.items() if n==42) == set(range(25,53))
    assert cross['beh53_True_vs_readme54_Video']==21 and cross['beh53_False_vs_readme54_Imagery']==21
    meta['behavior_id_audit'] = dict(README_matching_rows_of_42=matches, cross_table_alignment=cross,
        constant_video_name_participants=[p for p,x in ids_by_person.items() if len(set(x))==1],
        base_id_to_clip=code, exact_participants=list(range(25,53)), excluded_in_correction=[53,54])
    mapping_file=topography.parent/'video_rows.json'
    video_rows={int(p):v for p,v in json.loads(mapping_file.read_text(encoding='utf8')).items()}
    assert all(1 <= p <= 52 and len(v)==21 and sorted(set(v))==v for p,v in video_rows.items())
    assert set(range(25,53)).issubset(video_rows)
    for person, indices in video_rows.items():
        for position, behavior_index in enumerate(indices):
            first=position//3*6+position%3
            assert behavior_index in (first,first+3)
        if person>=25:
            assert indices==[i for i,v in enumerate(ids_by_person[person]) if v>21]
    meta['versioned_video_rows'] = dict(people=sorted(video_rows),sha256=sha256(mapping_file.read_bytes()).hexdigest())
    correction_before_qc = defaultdict(Counter)
    for old in source:
        person, clip = old['person'], old['trial']
        if person > 52:
            continue
        position = orders[person].index(clip)
        first = position//3*6+position%3
        c = old['label']
        if person in video_rows:
            value=old['rating'][(first,first+3).index(video_rows[person][position])]
            c=int(value>3.5) if value!=3.5 else None
        counts=correction_before_qc[person]
        counts['all_trials']+=1
        counts['labelled']+=int(c is not None)
        counts['scored']+=int(c is not None and bool(old['windows']))
    meta['corrected_coverage_by_person_before_qc'] = correction_before_qc
    for row in rows:
        person = row['person']
        # The README explicitly numbers missing imagery trials 1..21, video
        # trials 22..42. Subject 53's structured TSV matches the *different*
        # published low-ID imagery and high-ID video sequences of participant
        # 54 in all 42 rows. Their EEG/behavior identity needs author evidence.
        row['corrected'] = row['label'] if person < 25 else None
        row['label_origin'] = 'concordant_pair'
        if person in video_rows:
            index=video_rows[person][row['position']]
            rating = row['pair_ratings'][row['pair_indices'].index(index)]
            row['corrected'] = int(rating>3.5) if rating!=3.5 else None
            row['label_origin'] = 'verified_video_row'
            assert row['label'] is None or row['corrected']==row['label']
    return rows, meta


def group_weights(people, y):
    counts = Counter(people)
    w = np.array([1. / counts[p] for p in people])
    for c in (0, 1):
        w[y == c] /= 2 * w[y == c].sum()
    return w


def weighted_logistic(x, y, weights, ranking):
    l2 = .05

    def objective(coef):
        slope, bias = coef[:-1], coef[-1]
        z = x @ slope + bias
        value = np.sum(weights * (np.logaddexp(0., z) - y * z)) + .5 * l2 * (slope @ slope)
        residual = weights * (expit(z) - y)
        return value, np.r_[x.T @ residual + l2 * slope, residual.sum()]

    bounds = [(None, None)] * x.shape[1] + ([(0, 0)] if ranking else [(None, None)])
    opt = minimize(objective, np.zeros(x.shape[1]+1), jac=True, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 180, 'ftol': 1e-9})
    if not opt.success:
        raise ValueError(opt.message)
    return opt.x[:-1], opt.x[-1]


def candidate(train, test, montage, method, export=False):
    person = np.array([r['person'] for r in train])
    y = np.array([r['label'] for r in train])
    w = group_weights(person, y)
    # Global prior is fitted exclusively on the outer/inner training fold.
    population = np.average(np.array([r[montage] for r in train]), axis=0, weights=w)

    def matrix(rows):
        raw = np.array([r[montage] for r in rows])
        if method == 'absolute':
            return raw
        prior = np.array([(r[montage+'_prior_sum'] + 3 * population) / (r['previous'] + 3) for r in rows])
        return raw - prior

    x = matrix(train)
    mean = np.average(x, axis=0, weights=w)
    scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=w))
    scale[scale < 1e-8] = 1.
    z = (x - mean) / scale
    if method == 'absolute':
        slope, bias = weighted_logistic(z, y, w, False)
    else:
        changes, pair_weights = [], []
        for group in sorted(set(person)):
            positives, negatives = z[(person == group) & (y == 1)], z[(person == group) & (y == 0)]
            if len(positives) and len(negatives):
                diff = (positives[:, None] - negatives[None, :]).reshape(-1, z.shape[1])
                changes.append(diff)
                pair_weights.extend([1./len(diff)] * len(diff))
        # All signed differences are positive-minus-negative; enforce zero intercept.
        design = np.concatenate(changes)
        weights = np.array(pair_weights) / sum(pair_weights)
        slope, _ = weighted_logistic(design, np.ones(len(design)), weights, True)
        raw = z @ slope
        bias = brentq(lambda b: np.average(expit(raw+b), weights=w) - .5, -60, 60)
    p = expit(((matrix(test) - mean) / scale) @ slope + bias)
    model = None if not export else dict(montage=montage, method=method, l2=.05, threshold=.5,
                                          history_pseudotrials=3, population=population.tolist(), mean=mean.tolist(),
                                          scale=scale.tolist(), weights=slope.tolist(), bias=float(bias))
    return p, model


def auc(y, s):
    y = np.asarray(y)
    pos = y.sum()
    if pos == 0 or pos == len(y):
        return np.nan
    return (rankdata(s)[y == 1].sum() - pos*(pos+1)/2) / (pos * (len(y)-pos))


def macro(rows, scores):
    groups = defaultdict(list)
    for row, p in zip(rows, scores):
        groups[row['person']].append((row['label'], p))
    return np.nanmean([auc(*zip(*v)) for v in groups.values()])


def folds(rows, count, seed):
    people = np.array(sorted({r['person'] for r in rows}))
    split = np.array_split(np.random.default_rng(seed).permutation(people), count)
    for part in split:
        hold = set(part.tolist())
        train = [r for r in rows if r['person'] not in hold]
        test = [r for r in rows if r['person'] in hold]
        yield train, test


def select(train, fold_seed):
    gathered = defaultdict(list)
    inner_groups = []
    for inner, test in folds(train, 3, fold_seed):
        inner_groups.append(sorted({r['person'] for r in test}))
        for m in MONTAGES:
            for method in METHODS:
                pred, _ = candidate(inner, test, m, method)
                gathered[m, method].extend(zip(test, pred))
    metrics = {m: {method: float(macro(*zip(*gathered[m,method]))) for method in METHODS} for m in MONTAGES}
    return {m:max(METHODS, key=metrics[m].get) for m in MONTAGES}, metrics, inner_groups


def evaluate(train, test, seed, phase, export=False):
    assert not ({r['person'] for r in train} & {r['person'] for r in test})
    best, scores, inner = select(train, seed)
    output = [dict(person=r['person'], trial=r['trial'], label=r['label'], position=r['position'], phase=phase,
                   eligible_windows=len(r['x'])) for r in test]
    base = predict_fold(train, test, 3)
    rates = {c: np.mean([r['label'] for r in train if r['trial'] == c]) for c in SORTED_CLIPS}
    rates_by_family = {c[:-1]: np.mean([r['label'] for r in train if r['trial'][:-1] == c[:-1]]) for c in SORTED_CLIPS}
    for r, p in zip(output, base):
        r['old'] = p
        r['stimulus_prior'] = float(rates[r['trial']])
        r['stimulus_family_prior'] = float(rates_by_family[r['trial'][:-1]])
    models = {}
    for m in MONTAGES:
        for method in METHODS:
            preds, model = candidate(train, test, m, method, export=export and method==best[m])
            if model is not None:
                models[m] = model
                replay=[score_exported_entropy(model,r[m],r[m+'_prior_sum'],r['previous']) for r in test]
                assert np.allclose(replay,preds,rtol=0.,atol=1e-12)
            for r, p in zip(output, preds):
                r[m+'_'+method] = float(p)
                if method == best[m]:
                    r[m] = float(p)
    return output, dict(phase=phase, selection=best, inner_within_person_auc=scores, inner_groups=inner,
                        train_people=sorted({r['person'] for r in train}), test_people=sorted({r['person'] for r in test})), models


def export_corrected_baseline(train, test, scores, trace):
    counts = Counter(r['person'] for r in train)
    x=np.concatenate([r['x'] for r in train])
    y=np.concatenate([np.repeat(r['label'],len(r['x'])) for r in train])
    w=np.concatenate([np.full(len(r['x']),1./(counts[r['person']]*len(r['x']))) for r in train])
    for c in (0,1):
        w[y==c]/=2*w[y==c].sum()
    params=fit(x,y,w,l2=.05)
    ps=[np.mean(sigmoid(logits(params,r['x']))) for r in test]
    assert np.allclose(ps,[r['old'] for r in scores],rtol=0.,atol=1e-13)
    return dict(parameters=params,model_id=sha256(json.dumps(params,sort_keys=True,allow_nan=False).encode()).hexdigest(),
                features=['log_rel_theta','log_rel_alpha','log_rel_beta'],electrodes=['Fp1','Fp2'],rate_hz=250,
                research_only=True,trained_people=sorted(counts),held_out_people=sorted({r['person'] for r in test}),
                target='Subjective video valence score_8 above vs below 3.5, excluding midpoints. Exact video rows '
                       'where recoverable, otherwise both possible ratings must have the same polarity.',
                identity_excluded=[53,54],video_row_provenance=trace,
                holdout_scores_csv='corrected_holdout.csv, column old')


def summarize(rows):
    names = ['old', 'fp', 'all64', 'fp_absolute', 'fp_prior_rank', 'all64_absolute', 'all64_prior_rank',
             'stimulus_prior', 'stimulus_family_prior']
    refits = all('previous_deployed' in r for r in rows)
    if refits:
        names.append('previous_deployed')
    people = sorted({r['person'] for r in rows})
    rng = np.random.default_rng(SEED)
    sample_indices = rng.integers(len(people), size=(4000, len(people)))
    points, draws = {}, {}
    for key in names:
        y = np.array([r['label'] for r in rows])
        p = np.array([r[key] for r in rows])
        by_person, aucs = [], []
        for person in people:
            mask = np.array([r['person']==person for r in rows])
            yp, pp = y[mask], p[mask]
            aucs.append(auc(yp, pp))
            by_person.append([sum((yp==1)&(pp>=.5)), sum(yp==1), sum((yp==0)&(pp<.5)), sum(yp==0)])
        counts = np.array(by_person)
        def ba(z):
            return .5*(z[...,0]/z[...,1] + z[...,2]/z[...,3])
        draws[key] = {'ba':ba(counts[sample_indices].sum(axis=1)), 'macro_auc':np.nanmean(np.array(aucs)[sample_indices],axis=1)}
        by_clip = defaultdict(list)
        for r in rows:
            by_clip[r['phase'],r['trial']].append(r)
        per_clip = [auc([r['label'] for r in v],[r[key] for r in v]) for v in by_clip.values() if v]
        points[key] = dict(ba=float(ba(counts.sum(0))), macro_auc=float(np.nanmean(aucs)), pooled_auc=float(auc(y,p)),
                           within_clip_across_people_macro_auc=float(np.nanmean(per_clip)), both_class_clip_fold_groups=int(np.isfinite(per_clip).sum()),
                           ci95={k:np.percentile(v,[2.5,97.5]).tolist() for k,v in draws[key].items()})
    delta = {}
    pairs = [('old','fp'), ('old','all64'), ('fp','all64'), ('fp_absolute','all64_absolute'), ('fp_prior_rank','all64_prior_rank')]
    if refits:
        pairs.append(('previous_deployed','old'))
    for first,second in pairs:
        delta[second+'-minus-'+first]={m:dict(point=points[second][m]-points[first][m],ci95=np.percentile(draws[second][m]-draws[first][m],[2.5,97.5]).tolist())
                                      for m in ('ba','macro_auc')}
    return dict(trials=len(rows), people=people, both_class_people=sum(len({r['label'] for r in rows if r['person']==g})==2 for g in people),
                counts=dict(Counter(str(r['label']) for r in rows)), scores=points, paired_deltas=delta)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache', type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    out=args.output
    rows, meta = extract(args.cache, out/'topography')
    original = json.loads((args.cache/'evaluation/frontal_affect.json').read_text('utf8'))['models']['valence']
    all_rows = rows
    rows = [r for r in rows if r['label'] is not None]
    held=set(original['held_out_people'])
    dev=[r for r in rows if r['person'] not in held]
    test=[r for r in rows if r['person'] in held]
    parts, manifests = [], []
    for i, (train, validation) in enumerate(folds(dev, 5, SEED+1)):
        oof, fold, _ = evaluate(train, validation, SEED+20+i,'nested_dev_'+str(i+1))
        parts.extend(oof)
        manifests.append(fold)
    locked, manifest, models = evaluate(dev, test, SEED+70, 'original_holdout', export=True)
    manifests.append(manifest)
    with (args.cache/'evaluation/valence_heldout.csv').open(encoding='utf8') as src:
        historical={(int(r['person']),r['trial']):r for r in csv.DictReader(src)}
    assert set(historical) == {(r['person'],r['trial']) for r in locked}
    max_delta=max(abs(float(historical[(r['person'],r['trial'])]['score']) - r['old']) for r in locked)
    assert max_delta < 1e-8
    with (out/'predictions.csv').open('w',newline='',encoding='utf8') as fp:
        w=csv.DictWriter(fp,fieldnames=parts[0].keys()); w.writeheader(); w.writerows(parts+locked)
    report=dict(dataset='EmoEEG-MC, OpenNeuro ds005540 v1.0.7', seed=SEED, metadata=meta,
                require_original_eligible_windows=True, historical_score_parity_max_abs=max_delta,
                pooled_trials_nested=summarize(parts), original_holdout=summarize(locked), folds=manifests,
                design_sha256=sha256((Path(__file__).resolve().parents[1]/'docs/VALENCE_OPTIMIZATION_PROTOCOL.txt').read_bytes()).hexdigest(),
                script_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    # Separate metadata correction, evaluated transparently on the same people.
    # Check old as-deployed predictions on the expanded, now identifiable set.
    corrected = [{**r,'original_label':r['label'],'label':r['corrected']} for r in all_rows if r['corrected'] is not None]
    corrected_dev = [r for r in corrected if r['person'] not in held]
    corrected_test = [r for r in corrected if r['person'] in held]
    corrected_oof, corrected_folds = [], []
    for i, (train, val) in enumerate(folds(corrected_dev,5,SEED+1)):
        oof, f, _ = evaluate(train,val,SEED+20+i,'corrected_nested_'+str(i+1))
        corrected_oof.extend(oof); corrected_folds.append(f)
    pred, f, corrected_models = evaluate(corrected_dev, corrected_test, SEED+70,'corrected_holdout',export=True)
    corrected_folds.append(f)
    legacy_predictions = predict_fold(dev,corrected_test,3)
    for result, original_p, source in zip(pred,legacy_predictions,corrected_test):
        result['previous_deployed'] = original_p
        result['original_eligible'] = source['original_label'] is not None
        result['label_origin'] = source['label_origin']
    def compare_deployed(subset):
        copy=[]
        for row in subset:
            r=dict(row)
            r['old']=r.pop('previous_deployed')
            copy.append(r)
        return summarize(copy)
    report['metadata_corrected'] = dict(nested=summarize(corrected_oof), original_people=summarize(pred),
        versus_previous_deployed=compare_deployed(pred),
        refit_on_original_trials=summarize([r for r in pred if r['original_eligible']]),
        versus_previous_on_original_trials=compare_deployed([r for r in pred if r['original_eligible']]),
        verified_video_rows_only=compare_deployed([r for r in pred if r['label_origin']=='verified_video_row']),
        scored_trial_sources=dict(Counter(r['label_origin'] for r in corrected)), folds=corrected_folds)
    for name, content in [('corrected_holdout',pred), ('corrected_nested',corrected_oof)]:
        with (out/(name+'.csv')).open('w',newline='',encoding='utf8') as fp:
            w=csv.DictWriter(fp,fieldnames=content[0].keys()); w.writeheader(); w.writerows(content)
    write_json(out/'research_candidates.json',dict(research_only=True,trained_people=sorted({r['person'] for r in dev}),
                                                  test_people=sorted(held), models=models,
                                                  metadata_corrected_models=corrected_models,
                                                  corrected_training_people=sorted({r['person'] for r in corrected_dev})))
    original_features_refit = export_corrected_baseline(corrected_dev,corrected_test,pred,meta['versioned_video_rows'])
    original_features_refit['holdout'] = report['metadata_corrected']['original_people']['scores']['old']
    original_features_refit['paired_delta_to_previous_deployed'] = report['metadata_corrected']['original_people']['paired_deltas']['old-minus-previous_deployed']
    write_json(out/'refit_corrected_three_bands.json',original_features_refit)
    write_json(out/'results.json',report)
    for k in ('pooled_trials_nested', 'original_holdout'):
        print(k, json.dumps({n:report[k]['scores'][n]['ba'] for n in ['old','fp','all64','stimulus_prior']}))


if __name__ == '__main__':
    main()
