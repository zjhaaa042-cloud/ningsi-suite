"""Select the 28 control recordings from the versioned ATTLAPSE ZIP on Zenodo.

The BIDS source is Horakova et al., doi:10.5281/zenodo.23592484 (CC BY).
For each participant retain Fp1/Fp2 around every thought probe, plus the original
metadata. Range transfers resume at chunk boundaries. Verify the pinned ZIP64
index and each extracted member's original CRC and size before publishing data.

Example: python ningsi-capture/scripts/attlapse_data.py var/attlapse --with-aux
Optional --participant sub-T07, --offline, --source derivatives.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import configparser
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import time
import zipfile
import zlib

import numpy as np


URL = "https://zenodo.org/records/23592484/files/BIDS.zip"
DOI = "10.5281/zenodo.23592484"
ARCHIVE_BYTES = 38055298087
INDEX_OFFSET = 38055100545
INDEX_SHA256 = "2f0a415eeca1251d9eee037707b977347373168535f81cf4514a55d39f77625e"
# Hash of the authors' explicit subs_t list in ghorakova/attlapse_erp,
# 2_ANALYSES/bids_formatting.py (Git blob e015b5aad6cbac6dda538dfa9cbcae3cb3a09f86).
CONTROL_IDS_SHA256 = "83b30a449ca793c2b0abf4731528c823b9b59653849aaf193bae197314b63c94"
CHUNK = 16 * 1024 * 1024
CHANNELS = ("Fp1", "Fp2")
AUX_CHANNELS = ("left_ear", "right_ear", "eog_up", "eog_down", "eog_left", "eog_right")
# Qtrial, GNG_probe.m, lines 207, 220-235 in the authors' ATTLAPSE task.
PROBE_TRIALS = (81, 108, 162, 243, 297, 324, 351, 405, 486, 513, 594, 648, 702, 729, 810)


def fetch_range(target, start, stop):
    """Retrieve [start, stop), checking that the server honored the exact range."""
    length = stop - start
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == length:
        return
    with tempfile.TemporaryDirectory(dir=target.parent) as staging:
        temp = Path(staging) / "incoming"
        for attempt in range(6):
            proc = subprocess.run([
                "curl", "--location", "--silent", "--show-error", "--fail",
                "--range", f"{start}-{stop - 1}", "--max-filesize", str(length),
                "--connect-timeout", "30", "--speed-limit", "1024", "--speed-time", "90",
                "--max-time", "420", "--output", str(temp),
                "--write-out", "%{http_code} %{size_download}", URL,
            ], capture_output=True, text=True)
            if proc.returncode == 0 and proc.stdout.strip() == f"206 {length}" and temp.stat().st_size == length:
                os.replace(temp, target)
                return
            if attempt < 5:
                time.sleep(min(60, 3 * 2**attempt))
        raise OSError(f"Range {start}-{stop - 1} failed: {proc.stderr} {proc.stdout}")


def archive_index(root, online=True):
    path = root / "zip-index.bin"
    if online:
        fetch_range(path, INDEX_OFFSET, ARCHIVE_BYTES)
    data = bytearray(path.read_bytes())
    if hashlib.sha256(data).hexdigest() != INDEX_SHA256:
        raise ValueError("ATTLAPSE archive index differs from the pinned v2.0.1 index")
    # Open just the ZIP64 directory in memory. Adjust the two directory pointers
    # into this slice; the member offsets in the directory stay absolute.
    end = data.rindex(b"PK\x06\x06")
    struct.pack_into("<Q", data, end + 48, 0)
    struct.pack_into("<Q", data, end + 64, end)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        return archive.infolist()


def cohort(entries):
    names = sorted({Path(item.filename).parts[2] for item in entries
                    if item.filename.startswith("BIDS/sourcedata/sub-T")})
    ids_digest = hashlib.sha256("\n".join(names).encode("ascii")).hexdigest()
    if len(names) != 28 or ids_digest != CONTROL_IDS_SHA256:
        raise ValueError("Controls do not match the authors' published group roster")
    return names


def member_groups(entries, participant, source="sourcedata"):
    files = [item for item in entries if item.filename.startswith(f"BIDS/{source}/{participant}/")]
    eeg = next(item for item in files if item.filename.endswith("_eeg.eeg"))
    before = files[0].header_offset, eeg.header_offset
    end_eeg = next(item.header_offset for item in files if item.header_offset > eeg.header_offset)
    last = files[-1]
    next_offset = next((i.header_offset for i in entries if i.header_offset > last.header_offset), INDEX_OFFSET)
    return files, eeg, before, (eeg.header_offset, end_eeg), (end_eeg, next_offset)


def selection(root, entries, participants, source="sourcedata"):
    for person in participants:
        folder = root / ("downloads" if source == "sourcedata" else "derivatives-downloads") / person
        files, eeg, before, binary, after = member_groups(entries, person, source)
        yield folder / "before.part", *before
        yield folder / "after.part", *after
        for j, offset in enumerate(range(*binary, CHUNK)):
            yield folder / f"eeg-{j:03}.part", offset, min(binary[1], offset + CHUNK)


def check_bytes(record, data):
    if zlib.crc32(data) != record.CRC or len(data) != record.file_size:
        raise ValueError(f"Corrupt member: {record.filename}")


def data_offset(group, start, record):
    offset = record.header_offset - start
    if group[offset:offset + 4] != b"PK\x03\x04":
        raise ValueError(f"Missing header for {record.filename}")
    name_size, extra_size = struct.unpack_from("<HH", group, offset + 26)
    name = group[offset + 30:offset + 30 + name_size].decode()
    if name != record.filename or record.compress_type != zipfile.ZIP_DEFLATED:
        raise ValueError(f"Unexpected ZIP member at {record.header_offset}: {name}")
    return offset + 30 + name_size + extra_size


def read_metadata(group, start, records):
    metadata = {}
    for record in records:
        if record.is_dir() or not start <= record.header_offset < start + len(group):
            continue
        offset = data_offset(group, start, record)
        raw = zlib.decompress(group[offset:offset + record.compress_size], -15)
        check_bytes(record, raw)
        metadata[Path(record.filename).name] = raw
    return metadata


def publish_json(path, obj):
    with tempfile.TemporaryDirectory(dir=path.parent) as staging:
        temp = Path(staging) / path.name
        temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf8")
        os.replace(temp, path)


def publish_array(path, array):
    with tempfile.TemporaryDirectory(dir=path.parent) as staging:
        temp = Path(staging) / path.name
        np.save(temp, array, allow_pickle=False)
        os.replace(temp, path)


def file_ending(metadata, suffix):
    return next(value for key, value in metadata.items() if key.endswith(suffix)).decode("utf-8-sig")


def parse_tsv(text):
    return list(csv.DictReader(io.StringIO(text, newline=""), delimiter="\t"))


def channel_layout(metadata, selected=CHANNELS):
    text = file_ending(metadata, "_eeg.vhdr")
    config = configparser.ConfigParser(interpolation=None)
    config.read_string(text[text.index("[Common Infos]"):])
    common = config["Common Infos"]
    if common["DataOrientation"] != "MULTIPLEXED" or config["Binary Infos"]["BinaryFormat"] != "IEEE_FLOAT_32":
        raise ValueError("Unsupported BrainVision binary layout")
    rate = 1e6 / float(common["SamplingInterval"])
    count = int(common["NumberOfChannels"])
    if rate != 512.0:
        raise ValueError("Different Source EEG sampling rate")
    columns = [config["Channel Infos"][f"Ch{i}"].split(",") for i in range(1, count + 1)]
    indices = [next(i for i, col in enumerate(columns) if col[0] == channel) for channel in selected]
    scales = [float(columns[i][2]) for i in indices]
    rows = {row["name"]: row for row in parse_tsv(file_ending(metadata, "_channels.tsv"))}
    bad = []
    for name, i in zip(selected, indices):
        row = rows[name]
        is_ear = name.endswith("_ear")
        kind = "MISC" if is_ear else ("EOG" if name.startswith("eog_") else "EEG")
        unit = "n/a" if is_ear else "µV"
        if columns[i][3] != unit or row["units"] != unit or row["type"] != kind or float(row["sampling_frequency"]) != rate:
            raise ValueError(f"Unexpected channel units/type for {name}")
        if row["status"].lower() != "good":
            bad.append(name)
    return count, indices, scales, int(rate), bad


def parse_probes(metadata, rate):
    behavior = parse_tsv(file_ending(metadata, "_beh.tsv"))
    events = parse_tsv(file_ending(metadata, "_events.tsv"))
    definition = json.loads(file_ending(metadata, "_beh.json"))["att_state_score"]
    for term in ("on_task (< 43)", "deliberate mind-wandering (43-52)", "spontaneous mind-wandering (> 52)"):
        if term not in definition:
            raise ValueError("Changed label definitions")
    by_trial = {int(r["trial_number"]): r for r in behavior}
    # Some tables omit trial 1, before the first probe. The explicit trial
    # identifiers, not row positions, correspond to the task script.
    if len(by_trial) != len(behavior) or list(by_trial) not in (list(range(1, 811)), list(range(2, 811))):
        raise ValueError("Unexpected trial IDs in the raw behavioral table")
    trials_seen = 0
    phase_start = None
    probes = []
    for event in events:
        kind = event["trial_type"]
        sample = int(event["sample"])
        if abs(float(event["onset"]) * rate - sample) > .01:
            raise ValueError("Inconsistent event clock")
        if kind.startswith("stimulus_"):
            if phase_start is None:
                phase_start = sample
            trials_seen += 1
        elif kind == "end_probe":
            phase_start = None
        elif kind == "start_probe":
            # The actual MATLAB task triggers the questionnaire BEFORE showing
            # the stimulus associated with this behavioral table row.
            number = PROBE_TRIALS[len(probes)]
            trial = by_trial[number]
            if phase_start is None:
                raise ValueError("Probe events and trial numbers are not aligned")
            raw_score = trial["att_state_score"]
            score = None if raw_score in ("", "n/a", "NaN") else float(raw_score)
            answer = "missing"
            label = None
            if score is not None and 0 < score <= 100:
                answer = "on_task" if score < 43 else ("deliberate_mw" if score <= 52 else "spontaneous_mw")
                label = int(score >= 43)
            probes.append(dict(probe_id=f"probe-{len(probes) + 1:02}", onset_sample=sample,
                               onset_sec=sample / rate, phase_start_sec=phase_start / rate,
                               start_sample=sample - 18 * rate, stop_sample=sample - 2 * rate + 1,
                               source_answer=answer, source_score=score,
                               trial_number=number, stimulus_triggers_before=trials_seen,
                               missing_stimulus_triggers=number - 1 - trials_seen, label=label))
    if len(probes) != len(PROBE_TRIALS):
        raise ValueError("Unexpected number of trials or probes")
    # Some source EEG files dropped an ordinary stimulus trigger. A correct
    # block ordering can still be verified from all 15 questionnaire triggers,
    # the prespecified behavior rows and the counts of ordinary stimuli.
    deficits = [p["missing_stimulus_triggers"] for p in probes] + [810 - trials_seen]
    if any(n < 0 or n > 5 for n in deficits) or deficits != sorted(deficits):
        raise ValueError(f"Unexpected discrepancy between task and recorded triggers: {deficits}")
    answered = {p["trial_number"] for p in probes}
    if any(float(row["att_state_score"]) != 0 for row in behavior if int(row["trial_number"]) not in answered):
        raise ValueError("Nonempty response outside the matched probes")
    return probes


def selected_channels(record, chunks, nchan, indices, scales):
    inflater = zlib.decompressobj(-15)
    consumed = 0
    frames = []
    tail = b""
    crc = 0
    for j, chunk_path in enumerate(chunks):
        with chunk_path.open("rb") as handle:
            if j == 0:
                ptr = data_offset(handle.read(4096), record.header_offset, record)
                handle.seek(ptr)
            while consumed < record.compress_size:
                compressed = handle.read(min(1024 * 1024, record.compress_size - consumed))
                if not compressed:
                    break
                consumed += len(compressed)
                raw = inflater.decompress(compressed)
                crc = zlib.crc32(raw, crc)
                buf = tail + raw
                cut = len(buf) // (4 * nchan) * (4 * nchan)
                data = np.frombuffer(buf[:cut], dtype="<f4").reshape(-1, nchan)
                frames.append(data[:, indices].T.copy())
                tail = buf[cut:]
    if not inflater.eof or consumed != record.compress_size or crc != record.CRC or tail:
        raise ValueError(f"EEG CRC, length, or DEFLATE failure: {record.filename}")
    result = np.concatenate(frames, axis=1)
    if result.shape[1] * 4 * nchan != record.file_size:
        raise ValueError("EEG sample count disagrees with the published ZIP index")
    result *= np.asarray(scales, dtype=np.float32)[:, None]
    return result


def materialize(root, entries, person, source="sourcedata"):
    if source not in ("sourcedata", "derivatives"):
        raise ValueError(f"Unsupported data source: {source}")
    folder = root / ("downloads" if source == "sourcedata" else "derivatives-downloads") / person
    dest = root / ("cache" if source == "sourcedata" else "derivatives-cache") / person
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / "probes.json"
    if marker.is_file():
        return
    files, record, before, binary, after = member_groups(entries, person, source)
    metadata = read_metadata((folder / "before.part").read_bytes(), before[0], files)
    metadata.update(read_metadata((folder / "after.part").read_bytes(), after[0], files))
    for name, content in metadata.items():
        (dest / name).write_bytes(content)
    nchan, indices, scales, rate, bad = channel_layout(metadata)
    if source == "derivatives":
        # The processed behavior table drops the questionnaire answers. Bind
        # the original answers only after checking the processed EEG clocks.
        original = root / "cache" / person
        raw_meta = {p.name: p.read_bytes() for p in original.glob("*.tsv")}
        if parse_tsv(file_ending(raw_meta, "_events.tsv")) != parse_tsv(file_ending(metadata, "_events.tsv")):
            raise ValueError("Processed and raw EEG event streams differ")
        probes = json.loads((original / "probes.json").read_text(encoding="utf8"))["probes"]
    else:
        probes = parse_probes(metadata, rate)
    chunks = [folder / f"eeg-{j:03}.part" for j, _ in enumerate(range(*binary, CHUNK))]
    signals = selected_channels(record, chunks, nchan, indices, scales)
    publish_array(dest / "probe_eeg_uv.npy", segments_for_probes(signals, probes))
    report = dict(source_doi=DOI, source=source, participant=person, rate=rate, channels=CHANNELS,
                  bad_channels=bad, units="µV", scales_to_uv=scales, observed_channels=nchan,
                  eeg_zip_crc32=f"{record.CRC:08x}", original_eeg_bytes=record.file_size,
                  archive_index_sha256=INDEX_SHA256, eeg_npy_sha256=hashlib.sha256(
                      (dest / "probe_eeg_uv.npy").read_bytes()).hexdigest(),
                  source_member_crc32={r.filename: f"{r.CRC:08x}" for r in files if not r.is_dir()},
                  probes=probes)
    publish_json(marker, report)


def segments_for_probes(source, probes):
    segments = []
    for p in probes:
        a, b = p["start_sample"], p["stop_sample"]
        if a < 0 or b > source.shape[1]:
            raise ValueError("Probe interval outside continuous recording")
        segments.append(source[:, a:b])
    return np.stack(segments)


def materialize_aux(root, entries, person):
    folder = root / "cache" / person
    marker = folder / "source_aux.json"
    if marker.is_file():
        return
    summary = json.loads((folder / "probes.json").read_text(encoding="utf8"))
    files, record, _, binary, _ = member_groups(entries, person)
    metadata = {p.name: p.read_bytes() for p in folder.iterdir() if p.name.endswith(("_channels.tsv", "_eeg.vhdr"))}
    names = {r["name"] for r in parse_tsv(file_ending(metadata, "_channels.tsv"))}
    selected = [ch for ch in AUX_CHANNELS if ch in names]
    if not selected:
        raise ValueError(f"{person} has no recorded auxiliary channels")
    nchan, indices, scales, rate, bad = channel_layout(metadata, selected)
    download_dir = root / "downloads" / person
    chunks = [download_dir / f"eeg-{j:03}.part" for j, _ in enumerate(range(*binary, CHUNK))]
    source = selected_channels(record, chunks, nchan, indices, scales)
    path = folder / "source_aux.npy"
    publish_array(path, segments_for_probes(source, summary["probes"]))
    details = dict(channels=selected, units=["n/a" if ch.endswith("_ear") else "µV" for ch in selected],
                   scales_to_declared_units=scales, bad_channels=bad, missing_channels=sorted(set(AUX_CHANNELS) - set(selected)),
                   sampling_rate=rate, original_eeg_crc32=summary["eeg_zip_crc32"],
                   sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    publish_json(marker, details)


def iter_probes(root, participant=None, *, with_aux=False, source="sourcedata"):
    """Yield completed subjects only, with microvolt arrays shaped (2, time)."""
    root = Path(root)
    if source not in ("sourcedata", "derivatives") or (source == "derivatives" and with_aux):
        raise ValueError("Auxiliary electrodes are only cached for the original recordings")
    records = cohort(archive_index(root, online=False))
    if participant is not None and participant not in records:
        raise ValueError(f"Unknown control: {participant}")
    for person in ([participant] if participant is not None else records):
        folder = root / ("cache" if source == "sourcedata" else "derivatives-cache") / person
        marker = folder / "probes.json"
        if not marker.is_file():
            if participant is not None:
                raise FileNotFoundError(marker)
            continue
        summary = json.loads(marker.read_text(encoding="utf8"))
        path = folder / "probe_eeg_uv.npy"
        if summary["source_doi"] != DOI or summary["participant"] != person or summary["channels"] != list(CHANNELS):
            raise ValueError(f"Cache identity mismatch: {person}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != summary["eeg_npy_sha256"]:
            raise ValueError(f"Cache hash mismatch: {person}")
        data = np.load(path, mmap_mode="r", allow_pickle=False)
        rate = summary["rate"]
        if with_aux:
            auxpath = folder / "source_aux.npy"
            aux = json.loads((folder / "source_aux.json").read_text(encoding="utf8"))
            if hashlib.sha256(auxpath.read_bytes()).hexdigest() != aux["sha256"]:
                raise ValueError(f"Auxiliary cache hash mismatch: {person}")
            aux_data = np.load(auxpath, mmap_mode="r", allow_pickle=False)
        for i, probe in enumerate(summary["probes"]):
            times = np.arange(probe["start_sample"], probe["stop_sample"], dtype=np.int64) / rate
            fields = (dict(aux_channels=aux["channels"], aux_units=aux["units"], aux_values=aux_data[i],
                           bad_aux_channels=aux["bad_channels"]) if with_aux else {})
            yield dict(**probe, participant=person, rate=rate, timestamps=times,
                       bad_channels=summary["bad_channels"], values_uv=data[i], **fields)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--participant", action="append", help="e.g. sub-T07, defaults to all controls")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--offline", action="store_true", help="use previously downloaded chunks")
    parser.add_argument("--source", choices=["sourcedata", "derivatives"], default="sourcedata",
                        help="derivatives require a previously verified raw cache for the same person")
    parser.add_argument("--with-aux", action="store_true", help="also retain EOG and ears (ear units unspecified)")
    args = parser.parse_args()
    entries = archive_index(args.root, online=not args.offline)
    controls = cohort(entries)
    people = args.participant or controls
    if any(person not in controls for person in people):
        parser.error("Only control IDs in the published ZIP index can be selected")
    if args.with_aux and args.source == "derivatives":
        parser.error("--with-aux is only available for sourcedata")
    for person in people:
        if not args.offline:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(fetch_range, *job) for job in selection(args.root, entries, [person], args.source)]
                for future in concurrent.futures.as_completed(futures):
                    future.result()
        materialize(args.root, entries, person, args.source)
        if args.with_aux:
            materialize_aux(args.root, entries, person)
        print(f"Verified {person}", flush=True)


if __name__ == "__main__":
    main()
