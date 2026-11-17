"""Recover versioned EmoEEG-MC BIDS event tables from the OpenNeuro publisher mirror."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from urllib.request import urlopen


REPOSITORY = 'OpenNeuroDatasets/ds005540'


def get(url):
    for attempt in range(3):
        try:
            with urlopen(url, timeout=35) as r:
                return r.read()
        except OSError:
            if attempt == 2:
                raise
            time.sleep(1 + attempt)


def tree(version):
    obj = json.loads(get(f'https://api.github.com/repos/{REPOSITORY}/git/trees/{version}?recursive=1'))
    if obj['truncated']:
        raise ValueError('Incomplete source index')
    return {r['path']:r['sha'] for r in obj['tree'] if r['type'] == 'blob'}


def fetch(out):
    out.mkdir(exist_ok=True, parents=True)
    old, current = tree('1.0.3'), tree('1.0.7')
    def person(index):
        subject = f'sub-{index:02}'
        event = f'{subject}/{subject}_events.tsv'
        beh = f'{subject}/beh/{subject}_task-emotion_beh.tsv'
        signal = f'derivatives/{subject}/ses-vid/eeg/{subject}_ses-vid_task-emotion_reorder.npy'
        path = out/f'{subject}_events.tsv'
        raw = path.read_bytes() if path.exists() else get(f'https://raw.githubusercontent.com/{REPOSITORY}/1.0.3/{event}')
        git_hash = hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()
        if git_hash != old[event]:
            raise ValueError(f'Event hash mismatch at {event}')
        path.write_bytes(raw)
        return subject, {'event_git_sha1_v1.0.3':git_hash, 'events_in_v1.0.7':event in current,
                         'rating_sha1_unchanged':old.get(beh) is not None and old[beh] == current.get(beh),
                         'video_pointer_sha1_unchanged':old.get(signal) is not None and old[signal] == current.get(signal)}
    with ThreadPoolExecutor(10) as pool:
        stamps = dict(pool.map(person, range(1, 55)))
    (out/'source_versions.json').write_text(json.dumps(stamps,indent=2), encoding='utf8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    fetch(parser.parse_args().out)
