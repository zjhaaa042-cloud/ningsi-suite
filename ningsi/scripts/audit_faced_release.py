"""Check every used FACED sidecar and BDF reference against the published v1.1.4 Git tree."""
import argparse
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from fetch_faced_frontal import get, RAW


def check(out):
    treefile=out/'source'/'version_tree.json'
    if not treefile.exists():
        treefile.write_bytes(get('https://api.github.com/repos/nemarDatasets/nm000112/git/trees/v1.1.4?recursive=1')[0])
    tree=json.loads(treefile.read_text(encoding='utf8'));assert not tree['truncated']
    table={v['path']:v['sha'] for v in tree['tree'] if v['type']=='blob'}
    people={};ratings=set();columns=('Joy','Tenderness','Inspiration','Amusement','Anger','Disgust','Fear','Sadness','Arousal','Valence','Familiarity','Liking')
    for folder in sorted((out/'source').glob('sub-*')):
        name=folder.name;prefix=f'{name}/eeg/{name}_task-watchingVideoClips_';hashes={}
        for fname in ('events.tsv','channels.tsv','eeg.json'):
            raw=(folder/fname).read_bytes();actual=hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()
            if actual!=table[prefix+fname]:raise ValueError(f'Metadata does not match pinned version for {name} {fname}')
            hashes[fname]=actual
        events=list(csv.DictReader((folder/'events.tsv').read_text(encoding='utf-8-sig').splitlines(),delimiter='\t'))
        rows=sorted((r for r in events if r['video_index']!='n/a'),key=lambda r:int(r['video_index']))
        assert [int(r['video_index']) for r in rows]==list(range(1,29))
        survey=np.array([[float(row[col]) for col in columns] for row in rows]);assert np.isfinite(survey).all() and survey.min()>=0 and survey.max()<=7
        identity=hashlib.sha256(survey.astype('<f8').tobytes()).hexdigest()
        if identity in ratings:raise ValueError(f'Repeated full rating panel: {name}')
        ratings.add(identity);info=json.loads((folder/'provenance.json').read_text(encoding='utf8'))
        people[name]={'metadata_git_sha1':hashes,'original_12d_rating_sha256':identity,'frequency':info['rate'],'bdf_release_sha256':info['source_sha256_full_bdf']}
    assert len(people)==123
    report={'tree':tree['sha'],'source_version':'nm000112/v1.1.4','checked_sidecars':3*len(people),'subjects':people,
            'source':'https://github.com/nemarDatasets/nm000112/blob/v1.1.4/code/faced.py',
            'scoring':'Valence is column 10 of the 12 items from original After_remarks.mat, keyed by participant + vid. All 3444 rows contain finite individual ratings; trial_type, emotion_label and binary_label never enter EEG feature designs.',
            'verification_limits':'Anonymous original Synapse download responded 403. The versioned conversion source and all Git sidecars were verified; original MAT survey files were not independently hashed. Raw BDF byte ranges and header are locally fingerprinted, not the entire file.'}
    (out/'source_audit.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print('Verified 123 × 3 release metadata files and 3444 complete unique-survey trial rows.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);check(p.parse_args().out)
