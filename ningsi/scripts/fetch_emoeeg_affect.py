"""Fetch only the frontal channels and valence labels for complete EmoEEG-MC video sessions."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import io
from pathlib import Path
import time
from urllib.request import Request, urlopen

import numpy as np


BASE = "https://s3.amazonaws.com/openneuro.org/ds005540"


def get(url, byte_range=None):
    headers = {"Range": f"bytes={byte_range[0]}-{byte_range[1]}"} if byte_range else {}
    for attempt in range(4):
        try:
            with urlopen(Request(url, headers=headers), timeout=90) as response:
                if byte_range and (response.status != 206 or not response.headers.get(
                        "Content-Range", "").startswith(f"bytes {byte_range[0]}-{byte_range[1]}/")):
                    raise ValueError("Unexpected byte range from OpenNeuro")
                return response.read(), response.headers.get("ETag")
        except OSError:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def fetch(root, index):
    subject = f"sub-{index:02d}"
    target = root / f"{subject}.npz"
    if target.is_file():
        return
    path = f"{BASE}/derivatives/{subject}/ses-vid/eeg/{subject}_ses-vid_task-emotion_reorder.npy"
    head, etag = get(path, (0, 127))
    fp = io.BytesIO(head)
    if np.lib.format.read_magic(fp) != (1, 0):
        raise ValueError("Unexpected NPY version")
    shape, fortran, dtype = np.lib.format.read_array_header_1_0(fp)
    if shape != (64, 21, 6000) or fortran or dtype.str != "<f8" or fp.tell() != 128:
        raise ValueError(f"Unexpected complete-trial array for {subject}: {shape}")
    data, data_tag = get(path, (128, 127 + 3 * 21 * 6000 * 8))
    if data_tag != etag:
        raise ValueError("Dataset changed during download")
    channels = np.frombuffer(data, dtype=dtype).reshape(3, 21, 6000)[[0, 2]]
    ratings, _ = get(f"{BASE}/{subject}/beh/{subject}_task-emotion_beh.tsv")
    with target.with_suffix(".part").open("wb") as out:
        np.savez(out, channels=channels, ratings=ratings.decode("utf-8-sig"), etag=etag)
    target.with_suffix(".part").replace(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda index: fetch(args.root, index), range(1, 55)))
    print("Fetched video-session Fp1, Fp2 and original self-report rows for 54 complete participants.")
