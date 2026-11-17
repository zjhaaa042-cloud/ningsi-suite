"""Cache author-provided EmoEEG-MC video differential entropy without executing downloaded pickle code."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import pickle
import re
import tarfile

import numpy as np

from fetch_emoeeg_affect import BASE, get


class ArrayUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module in ('numpy.core.multiarray', 'numpy._core.multiarray') and name == '_reconstruct':
            return np.core.multiarray._reconstruct
        if module == 'numpy' and name in ('ndarray', 'dtype'):
            return getattr(np, name)
        raise ValueError(f'Unsupported serialized global: {module}.{name}')


def fetch(root, person):
    target = root / f'sub-{person:02d}.npz'
    if target.exists():
        return
    sub = f'sub-{person:02d}'
    path = f'{BASE}/derivatives/{sub}/ses-vid/eeg/{sub}_ses-vid_task-emotion_de.npy'
    data, etag = get(path)
    fp = BytesIO(data)
    if np.lib.format.read_magic(fp) != (1, 0):
        raise ValueError('Unknown NPY version')
    shape, _, dtype = np.lib.format.read_array_header_1_0(fp)
    if shape != () or dtype.kind != 'O':
        raise ValueError('Expected scalar dictionary')
    values = ArrayUnpickler(fp).load().item()
    if not isinstance(values, dict):
        raise ValueError('Expected dictionary of arrays')
    for key, array in values.items():
        if not isinstance(key, str) or not isinstance(array, np.ndarray) or array.dtype.kind not in 'fiu':
            raise ValueError(f'Not a numeric matrix: {key!r}')
    with target.with_suffix('.part').open('wb') as out:
        np.savez_compressed(out, **values, etag=etag, sha256=sha256(data).hexdigest())
    target.with_suffix('.part').replace(target)


def verify_release(root, count):
    url='https://codeload.github.com/OpenNeuroDatasets/ds005540/tar.gz/refs/tags/1.0.7'
    archive,_=get(url)
    items={}
    with tarfile.open(fileobj=BytesIO(archive)) as tree:
        for member in tree:
            match=re.search(r'/sub-(\d+)_ses-vid_task-emotion_de\.npy$',member.name)
            if not match or int(match[1])>count:
                continue
            expected=re.fullmatch(r'SHA256E-s\d+--([0-9a-f]{64})\.npy',member.linkname.rsplit('/',1)[-1])
            with np.load(root/f'sub-{match[1]}.npz',allow_pickle=False) as f:
                actual=str(f['sha256'])
            if not expected or expected[1]!=actual:
                raise ValueError(f'Official version SHA mismatch: {member.name}')
            items[match[1]]=actual
    if len(items)!=count:
        raise ValueError(f'Only found {len(items)} of {count} checksums')
    (root.parent/'topography_release_sha256.json').write_text(json.dumps({'version':'1.0.7','url':url,'verified':items},indent=2),encoding='utf8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--subjects', type=int, default=53)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(6) as pool:
        list(pool.map(lambda index: fetch(args.root, index), range(1, args.subjects + 1)))
    verify_release(args.root,args.subjects)
    with np.load(args.root / 'sub-01.npz', allow_pickle=False) as sample:
        print(json.dumps({key: list(sample[key].shape) for key in sample.files}))
