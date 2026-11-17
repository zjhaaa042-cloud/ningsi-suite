"""Resolve exact video-rating rows from versioned events and independently cross-check late stimulus IDs."""
import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np


def complete_pairing(kinds):
    return (len(kinds) == 42 and kinds.count('vid') == 21 and kinds.count('ima') == 21
            and all(len(set(kinds[i:i+3])) == 1 for i in range(0, 42, 3))
            and all(kinds[i] != kinds[i+3] for i in range(0, 42, 6)))


def stimuli_from_events(rows):
    # A recording can span two files with clocks restarting. Preserve the BIDS row order.
    ratings = [i for i, row in enumerate(rows) if row['trial_type'] == 'rating']
    intervals = [[r['trial_type'] for r in rows[a+1:b] if r['trial_type'] in ('vid', 'ima')]
                 for a, b in zip([-1]+ratings, ratings)]
    # The protocol includes practice before the 42 formal ratings. Only accept a
    # rating-anchored sequence when *every* formal rating has exactly one stimulus.
    last = intervals[-42:]
    if len(last) == 42 and all(len(part)==1 for part in last):
        kinds = [part[0] for part in last]
        if complete_pairing(kinds):
            return kinds
    # Some files contain surplus rating markers, but all 42 stimulus onsets are
    # present exactly once in the correct protocol structure.
    all_stim = [r['trial_type'] for r in rows if r['trial_type'] in ('vid','ima')]
    if complete_pairing(all_stim):
        return all_stim
    return None


def resolve(events, cache, output):
    manifest = json.loads((events/'source_versions.json').read_text(encoding='utf8'))
    audit, mappings, id_checks = {}, {}, []
    for person in range(1,55):
        subject = f'sub-{person:02}'
        v = manifest[subject]
        if not v['rating_sha1_unchanged'] or not v['video_pointer_sha1_unchanged']:
            raise ValueError(f'Dataset records changed for {subject}')
        with (events/f'{subject}_events.tsv').open(encoding='utf-8-sig') as stream:
            tokens = list(csv.DictReader(stream, delimiter='\t'))
        event_kinds = stimuli_from_events(tokens)
        with np.load(cache/f'{subject}.npz',allow_pickle=False) as data:
            rows = list(csv.DictReader(io.StringIO(str(data['ratings'])), delimiter='\t'))
        numbers = [int(float(r['trial_number'].strip('[]'))) for r in rows]
        if numbers != list(range(1,43)):
            raise ValueError(f'Unexpected behavior numbering {subject}')
        ids = [int(float(row['video_name'].strip('[]'))) for row in rows]
        id_kinds = ['vid' if x>21 else 'ima' for x in ids] if sorted(ids)==list(range(1,43)) else None
        if id_kinds is not None and not complete_pairing(id_kinds):
            id_kinds = None
        if event_kinds is not None and id_kinds is not None:
            if id_kinds != event_kinds:
                raise ValueError(f'Independent modality identifiers disagree for {subject}')
            id_checks.append(person)
        kinds = event_kinds or id_kinds
        source = ('event+id' if event_kinds and id_kinds else 'event' if event_kinds else 'id' if id_kinds else 'unresolved')
        if person >= 53:
            # Stimulus-name lists for these people conflict across source files.
            # A verified modality order alone cannot identify the correct clip EEG.
            source += '; excluded: clip order discrepancy'
        elif kinds is not None:
            mappings[person] = [i for i,kind in enumerate(kinds) if kind=='vid']
        audit[subject] = dict(source=source, event_count=len(tokens), video_rows_0based=mappings.get(person))
    if len(id_checks) < 20:
        raise ValueError('Insufficient independent event agreement to use stimulus IDs')
    output.mkdir(parents=True,exist_ok=True)
    (output/'video_rows.json').write_text(json.dumps(mappings,indent=2),encoding='utf8')
    (output/'event_alignment.json').write_text(json.dumps(dict(agreement_subjects=id_checks, audit=audit),indent=2),encoding='utf8')
    print(f'{len(mappings)} subjects recovered; {len(id_checks)} independent full-ID checks.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('events','cache','output'):
        ap.add_argument(name,type=Path)
    args = ap.parse_args()
    resolve(args.events,args.cache,args.output)
