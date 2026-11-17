"""Prepare and read Fp1/Fp2-only pre-probe epochs from the pinned raw ATTLAPSE archive.

Offline usage:
    python attlapse_two_channel_data.py ROOT_TO_VAR_ATTLAPSE
    for probe in iter_probes(ROOT_TO_VAR_ATTLAPSE / "two-channel-benchmark"): ...

The exclusive upper endpoint is the EEG sample marking probe onset. Epochs
contain unfiltered, source-scaled microvolts, without other electrode inputs.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np

import attlapse_data as source


HISTORY_SEC = 46
RATE_HZ = 512
SOURCE = "sourcedata"
SCHEMA_VERSION = 1


def _verify_subject(folder, person):
    details = json.loads((folder / "epoch_metadata.json").read_text(encoding="utf8"))
    if (details["schema_version"] != SCHEMA_VERSION or details["subject"] != person
            or details["source_doi"] != source.DOI or details["source"] != SOURCE
            or details["archive_index_sha256"] != source.INDEX_SHA256
            or details["selected_channels"] != list(source.CHANNELS) or details["unit"] != "µV"):
        raise ValueError(f"Two-channel epoch identity mismatch for {person}")
    path = folder / "epochs_uv.npy"
    if hashlib.sha256(path.read_bytes()).hexdigest() != details["epochs_sha256"]:
        raise ValueError(f"Two-channel epochs failed checksum for {person}")
    epochs = np.load(path, mmap_mode="r", allow_pickle=False)
    probes = details["probes"]
    if epochs.dtype != np.float32 or epochs.shape != (len(source.PROBE_TRIALS), 2, HISTORY_SEC * RATE_HZ):
        raise ValueError(f"Two-channel array dimensions/dtype changed for {person}")
    if (len(probes) != len(source.PROBE_TRIALS) or details["sample_rate_hz"] != RATE_HZ
            or tuple(p["trial_number"] for p in probes) != source.PROBE_TRIALS):
        raise ValueError(f"Source probe roster changed for {person}")
    for probe in probes:
        end = probe["onset_sample"]
        if (end != probe["stop_sample"] or probe["start_sample"] != end - HISTORY_SEC * RATE_HZ
                or probe["onset_sec"] != end / RATE_HZ or probe["phase_start_sec"] > probe["start_sample"] / RATE_HZ):
            raise ValueError(f"Invalid probe endpoints or block boundaries for {person} {probe['probe_id']}")
    return details, epochs


def _prepare_subject(root, out, entries, person):
    dest = out / "raw_epochs" / person
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "epoch_metadata.json").is_file():
        return _verify_subject(dest, person)[0]
    members, eeg, before, binary, after = source.member_groups(entries, person, SOURCE)
    download = root / "downloads" / person
    metadata = source.read_metadata((download / "before.part").read_bytes(), before[0], members)
    metadata.update(source.read_metadata((download / "after.part").read_bytes(), after[0], members))
    # Parse only the source behavior, timestamps and the fixed recording layout.
    nchan, columns, scales, rate, bad = source.channel_layout(metadata, selected=source.CHANNELS)
    probes = source.parse_probes(metadata, rate)
    for probe in probes:
        probe["stop_sample"] = probe["onset_sample"]
        probe["start_sample"] = probe["stop_sample"] - rate * HISTORY_SEC
        if probe["onset_sec"] != probe["onset_sample"] / rate:
            raise ValueError(f"Source event time mismatch in {person}")
        if probe["start_sample"] / rate < probe["phase_start_sec"]:
            raise ValueError(f"Full epoch crosses task phase in {person} {probe['probe_id']}")
    chunks = [download / f"eeg-{j:03}.part" for j, _ in enumerate(range(*binary, source.CHUNK))]
    # CRC is checked on the full original multiplexed ZIP member. Only the two
    # specified source columns are retained or used in any signal calculation.
    continuous = source.selected_channels(eeg, chunks, nchan, columns, scales)
    epochs = source.segments_for_probes(continuous, probes)
    path = dest / "epochs_uv.npy"
    source.publish_array(path, epochs)
    details = {
        "schema_version": SCHEMA_VERSION, "subject": person, "source_doi": source.DOI, "source": SOURCE,
        "source_archive_member": eeg.filename, "source_eeg_crc32": f"{eeg.CRC:08x}",
        "source_eeg_size_bytes": eeg.file_size, "source_eeg_recording_samples": continuous.shape[1],
        "source_metadata_crc32": {r.filename: f"{r.CRC:08x}" for r in members if not r.is_dir() and r != eeg},
        "archive_index_sha256": source.INDEX_SHA256, "sample_rate_hz": rate,
        "selected_channels": list(source.CHANNELS), "channel_indices_zero_based": columns,
        "source_scaling_to_uv": scales, "source_bad_channels": bad,
        "source_eeg_reference": json.loads(source.file_ending(metadata, "_eeg.json")).get("EEGReference"),
        "unit": "µV", "epochs_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "probes": probes,
    }
    source.publish_json(dest / "epoch_metadata.json", details)
    return _verify_subject(dest, person)[0]


def prepare(root, out=None, workers=3):
    root = Path(root)
    out = Path(out) if out is not None else root / "two-channel-benchmark"
    entries = source.archive_index(root, online=False)
    people = source.cohort(entries)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(_prepare_subject, root, out, entries, person) for person in people]
        summaries = [job.result() for job in jobs]
    rows = [p for s in summaries for p in s["probes"]]
    label_counts = Counter(p["label"] for p in rows)
    if len(rows) != 420 or label_counts != {0: 272, 1: 148}:
        raise ValueError("Probe or class counts differ from the pinned dataset")
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "source_doi": source.DOI, "source_url": "https://zenodo.org/records/23592484",
        "source_version": "2.0.1", "license": "CC BY 4.0",
        "authors": ["Gabriela Horáková", "Sébastien Weibel", "Ugo Tissot",
                    "Bich-Thuy Pham", "Madalina Elena Costache", "Anne Bonnefond"],
        "source": SOURCE, "group": "authors' subs_t roster: 28 healthy controls",
        "group_roster_sha256": source.CONTROL_IDS_SHA256, "archive_index_sha256": source.INDEX_SHA256,
        "signals": list(source.CHANNELS), "sample_rate_hz": RATE_HZ, "unit": "µV",
        "samples_per_probe": RATE_HZ * HISTORY_SEC,
        "interval": "[-46,0) seconds relative to the start_probe EEG marker. First sample included, onset sample excluded.",
        "phase_check": "Every complete epoch starts on or after its first stimulus of the current task block.",
        "feature_preprocessing": "No participant- or label-derived transformations. Original multiplexed recordings; retain only the two named channels, apply the BrainVision header scale. Other electrodes are neither extracted nor used.",
        "source_mapping": {"on_task": 0, "deliberate_mw": 1, "spontaneous_mw": 1},
        "source_answers": dict(Counter(p["source_answer"] for p in rows)),
        "probe_count": len(rows), "class_counts": dict(label_counts), "subjects": people,
        "source_reference": sorted({s["source_eeg_reference"] for s in summaries}),
        "source_eeg_crc32_by_person": {s["subject"]: s["source_eeg_crc32"] for s in summaries},
        "epoch_sha256_by_person": {s["subject"]: s["epochs_sha256"] for s in summaries},
    }
    source.publish_json(out / "data_provenance.json", provenance)
    return provenance


def iter_probes(root, participant=None):
    """Yield raw frontal waveforms, phase start and exact sample indexes from the benchmark folder."""
    root = Path(root)
    if participant is None:
        proven = json.loads((root / "data_provenance.json").read_text(encoding="utf8"))
        people = proven["subjects"]
        roster = hashlib.sha256("\n".join(people).encode("ascii")).hexdigest()
        if roster != source.CONTROL_IDS_SHA256:
            raise ValueError("Unexpected subject roster in two-channel benchmark")
    else:
        if not isinstance(participant, str) or not participant.startswith("sub-T") or not participant[5:].isdigit():
            raise ValueError("Expected a control participant ID")
        people = [participant]
    for person in people:
        meta, epochs = _verify_subject(root / "raw_epochs" / person, person)
        for i, p in enumerate(meta["probes"]):
            times = np.arange(p["start_sample"], p["stop_sample"]) / RATE_HZ
            yield {**p, "participant": person, "rate": RATE_HZ, "channels": list(source.CHANNELS),
                   "bad_channels": meta["source_bad_channels"], "unit": meta["unit"],
                   "source_doi": meta["source_doi"], "source": meta["source"],
                   "timestamps": times, "values_uv": epochs[i]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path, help="Folder containing pinned ZIP directory and downloaded subject chunks")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    result = prepare(args.source_root, args.output, args.workers)
    print(f"Verified two raw frontal channels for {result['probe_count']} probes from {len(result['subjects'])} controls.")
