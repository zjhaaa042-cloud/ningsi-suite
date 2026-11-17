"""Fetch the first-session, eyes-open rest source files from du Bois et al. v3."""
import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests


BASE = 'https://data.mendeley.com/public-api/datasets/gsxphk87mc'


def read_url(url):
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return response.content


def main(cache):
    cache.mkdir(parents=True, exist_ok=True)
    folders = json.loads(read_url(BASE + '/folders/3'))
    selected = []
    for folder in folders:
        if folder['name'] not in {'MI_EEG_data (mat)', 'NF_EEG_data (mat)', 'workbooks', 'questionnaires'}:
            continue
        files = json.loads(read_url(BASE + '/files?folder_id=' + folder['id'] + '&version=3'))
        for entry in files:
            name = entry['filename']
            needed = ('_ss1_' in name and ('relax' in name or 'EEG_game' in name))
            if needed or name in {'mood-stress-data.xlsx','Pre-Post NF Q_Kinyarwanda.pdf'}:
                selected.append(entry)

    def fetch(entry):
        name = entry['filename']
        assert Path(name).name == name
        target = cache / name
        details = entry['content_details']
        expected = details['sha256_hash']
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
            return name
        for _ in range(3):
            try:
                data = read_url(details['download_url'])
                if hashlib.sha256(data).hexdigest() == expected:
                    target.write_bytes(data)
                    return name
            except OSError:
                continue
        raise RuntimeError('Download or SHA-256 mismatch: ' + name)

    with ThreadPoolExecutor(6) as pool:
        for name in pool.map(fetch, selected):
            print(name, flush=True)
    manifest = {r['filename']:r['content_details']['sha256_hash'] for r in selected}
    (cache/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')


if __name__ == '__main__':
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument('cache', type=Path)
    main(args.parse_args().cache)
