"""Verify publisher metadata and retrieve two forehead EEG channels for never-fitted participants 55–60."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from fetch_emoeeg_affect import BASE, get
from fetch_emoeeg_events import REPOSITORY, tree


def collect(out):
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "trees.json"
    if cache.exists():
        trees = json.loads(cache.read_text(encoding="utf8"))
        old, current = trees["1.0.3"], trees["1.0.7"]
    else:
        old, current = tree("1.0.3"), tree("1.0.7")
        selected = {v:{k:s for k,s in entries.items() if any(f"sub-{p}" in k for p in range(55,61))}
                    for v,entries in (("1.0.3",old),("1.0.7",current))}
        cache.write_text(json.dumps(selected), encoding="utf8")
    def person(p):
        sub = f"sub-{p}"
        folder = out / sub
        folder.mkdir(exist_ok=True)
        rel = next(k for k in current if f"derivatives/{sub}/ses-vid/" in k and k.endswith("_reorder.npy"))
        beh = f"{sub}/beh/{sub}_task-emotion_beh.tsv"
        event = f"{sub}/{sub}_events.tsv"
        event_bytes = (folder/"events.tsv").read_bytes() if (folder/"events.tsv").exists() else get(f"https://raw.githubusercontent.com/{REPOSITORY}/1.0.3/{event}")[0]
        if hashlib.sha1(f"blob {len(event_bytes)}\0".encode() + event_bytes).hexdigest() != old[event]:
            raise ValueError("Unexpected publisher event blob")
        (folder / "events.tsv").write_bytes(event_bytes)
        ratings = (folder/"behavior.tsv").read_bytes() if (folder/"behavior.tsv").exists() else get(f"{BASE}/{beh}")[0]
        (folder / "behavior.tsv").write_bytes(ratings)
        stamp = {"event_git_blob":old[event], "behavior_versions_match":old[beh]==current[beh],
                 "waveform_pointer_versions_match":old[rel]==current[rel], "ratings_sha256":hashlib.sha256(ratings).hexdigest()}
        if p >= 57:
            return sub, {**stamp, "skipped_waveform":"All ten ratings duplicate earlier complete participants."}
        head, tag = get(f"{BASE}/{rel}", (0, 127))
        fp = io.BytesIO(head)
        if np.lib.format.read_magic(fp) != (1,0):
            raise ValueError("Unexpected npy header version")
        shape, fort, dtype = np.lib.format.read_array_header_1_0(fp)
        if (shape[0] != 64 or shape[2] != 6000 or not 1 <= shape[1] <= 21 or fort
                or dtype.str not in ("<f8", "<f4") or fp.tell() != 128):
            raise ValueError(f"Unexpected waveform dimensions in {sub}: {shape}")
        stop = 127 + 3 * shape[1] * 6000 * dtype.itemsize
        chunk, chunk_tag = get(f"{BASE}/{rel}", (128, stop))
        if tag != chunk_tag or len(chunk) != stop - 127:
            raise ValueError("Waveform payload identity or length changed")
        channels = np.frombuffer(chunk, dtype).reshape(3, shape[1], 6000)[[0, 2]]
        np.save(folder / "fp.npy", channels)
        ica_url = f"{BASE}/{rel.replace('_reorder.npy','_ica.npy')}"
        ica_header, _ = get(ica_url, (0, 127))
        stream = io.BytesIO(ica_header)
        assert np.lib.format.read_magic(stream) == (1, 0)
        dims, f_order, dt = np.lib.format.read_array_header_1_0(stream)
        assert dims[0] == 64 and dims[1] <= 21 and dims[2] == 6000 and not f_order and dt.kind == "f"
        occupied = 3 * dims[1] * dims[2] * dt.itemsize
        ica_chunk, _ = get(ica_url, (128, 127 + occupied))
        np.save(folder/"fp_ica.npy", np.frombuffer(ica_chunk, dt).reshape(3, dims[1], dims[2])[[0,2]])
        return sub, {**stamp, "range_bytes": len(chunk),
                     "waveform_etag": tag, "shape": shape, "frontal_data_sha256": hashlib.sha256(chunk).hexdigest()}
    with ThreadPoolExecutor(6) as pool:
        info = dict(pool.map(person, range(55,61)))
    (out / "sources.json").write_text(json.dumps(info, indent=2), encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    collect(parser.parse_args().output)
