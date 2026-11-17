"""Count alignment ambiguity without seeing EEG predictions or filling in labels."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import io
import json
from pathlib import Path

from fetch_faced_frontal import get
from validate_music_valence_transfer import GITHUB, verify_git


def audit(path,root,sha):
    cache=root/path.replace('/','_')
    if not cache.exists():cache.write_bytes(get(GITHUB+path)[0])
    raw=cache.read_bytes();verify_git(raw,sha)
    groups=defaultdict(set)
    for row in csv.DictReader(io.StringIO(raw.decode('utf-8-sig')),delimiter='\t'):
        groups[float(row['onset'])].add(int(row['trial_type']))
    starts=sorted(t for t,g in groups.items() if 788 in g)
    counts=Counter();lag=Counter()
    for i,t in enumerate(starts):
        lim=starts[i+1] if i+1<len(starts) else float('inf')
        events=[(tt,g) for tt,g in sorted(groups.items()) if t<tt<lim]
        q=[(tt,g) for tt,g in events if 800 in g and not g.intersection(range(801,808))]
        both=[];one=[];present=0
        for tt,g in q:
            aa={c-900 for c in g if 901<=c<=909};bb={c-832 for c in g if 833<=c<=841}
            if len(aa)==len(bb)==1:both.append(aa==bb)
            if len(aa)==1 and not bb:one.append('answer')
            if len(bb)==1 and not aa:one.append('response')
            if aa or bb:present+=1
        if sum(both)==1:reason='exactly_one_concordant_timestamp'
        elif sum(both)>1:reason='multiple_concordant_timestamps'
        elif both:reason='both_present_but_scores_disagree'
        elif one:reason='single_score_stream_only'
        elif q:reason='pleasant_prompt_present_but_no_scores'
        else:reason='no_isolated_pleasant_prompt'
        counts[reason]+=1
        # Compare the 90x event on one prompt with the 83x response on
        # surrounding prompts. These descriptive offsets cannot define labels.
        seq=[]
        for tt,g in events:
            qq=g.intersection(range(800,808));aa=g.intersection(range(901,910));bb=g.intersection(range(833,842))
            if len(qq)==1 and (aa or bb):
                seq.append((next(iter(qq)),next(iter(aa))-900 if len(aa)==1 else None,next(iter(bb))-832 if len(bb)==1 else None))
        for j,(n,a,b) in enumerate(seq):
            if a is None:continue
            for offset in range(-2,4):
                k=j+offset
                if 0<=k<len(seq) and seq[k][2] is not None:
                    lag[(offset,'n')]+=1;lag[(offset,'same')]+=a==seq[k][2]
    return counts,lag


def run(out):
    src=out/'source';cache=src/'events';cache.mkdir(exist_ok=True)
    tree=json.loads((src/'release_tree.json').read_text(encoding='utf8'))['tree']
    files=[(x['path'],cache,x['sha']) for x in tree if x['path'].endswith('_events.tsv')]
    with ThreadPoolExecutor(16) as pool: findings=list(pool.map(lambda x:audit(*x),files))
    totals=Counter();offsets=Counter()
    for c,l in findings:totals.update(c);offsets.update(l)
    result=dict(verified_tables=len(files),primary_alignment=dict(totals),all_question_offset_agreement={
        str(o):[offsets[o,'same'],offsets[o,'n']] for o in range(-2,4)},
        independent_behavior_files=[x['path'] for x in tree if '/beh/' in x['path'] or x['path'].endswith('ratings.tsv')])
    (out/'event_audit.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);run(p.parse_args().out)
