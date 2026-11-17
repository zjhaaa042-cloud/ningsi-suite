"""Pre-answer spectral and stimulus-locked feature extraction from the pinned ATTLAPSE archive."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

import attlapse_data as source
from attlapse_two_channel_data import _verify_subject


LEADS = ("Fp1", "Fp2", "P3", "P4", "eog_up", "eog_down", "eog_left", "eog_right")
BANDS = ((4, 8), (8, 13), (13, 30))
RATE = 250
SPEC_NAMES = [f"{lead}_log_{band}" for lead in ("channelmean", "differential")
              for band in ("theta", "alpha", "beta")]
ERP_NAMES = [f"{lead}_go_{lo}_{hi}ms" for lead in ("common", "differential")
             for lo, hi in ((80, 150), (180, 300), (300, 500))]


def grid(raw, rate, first, onset):
    """Resample to 250 Hz with a 65-tap Kaiser antialias FIR, excluding all data from -2 s on."""
    stop = onset - 2 * rate
    raw = np.asarray(raw[:, :stop - first], dtype=float)
    if raw.shape[1] <= 65 or first + raw.shape[1] != stop:
        raise ValueError("Epoch does not reach the safe cutoff")
    n = np.arange(-32, 33)
    kernel = 2 * 100 / rate * np.sinc(n * 2 * 100 / rate) * np.kaiser(len(n), 7)
    kernel /= kernel.sum()
    source_t = (first + 32 + np.arange(raw.shape[1] - 64) - onset) / rate
    dest_t = np.arange(np.ceil(RATE * source_t[0]), np.floor(RATE * source_t[-1]) + 1) / RATE
    resampled = np.stack([np.interp(dest_t, source_t, np.convolve(row, kernel, "valid")) for row in raw])
    # Remove slow drift. All support for scoring and fit sections lies before the probe cutoff.
    width = 251
    padded = np.pad(resampled, ((0, 0), (width // 2, width // 2)), mode="edge")
    trend = np.stack([np.convolve(row, np.ones(width) / width, "valid") for row in padded])
    return resampled - trend, dest_t


def features(pair, t, go_onsets):
    """Compute spectral medians and event-locked amplitudes; both use only pre-probe samples."""
    segments = []
    taper = np.hanning(2 * RATE)
    freqs = np.fft.rfftfreq(len(taper), 1 / RATE)
    for start in range(-18, -4):
        idx = np.searchsorted(t, start)
        data = pair[:, idx:idx + len(taper)]
        if data.shape[1] != len(taper) or np.max(np.ptp(data, axis=1)) > 150:
            continue
        channels = np.vstack((data, data[0] - data[1]))
        psd = abs(np.fft.rfft((channels - channels.mean(axis=1, keepdims=True)) * taper)) ** 2
        psd /= RATE * np.sum(taper ** 2)
        band = np.stack([psd[:, (freqs >= lo) & (freqs < hi)].sum(axis=1)
                         * (freqs[1] - freqs[0]) for lo, hi in BANDS], axis=1)
        lp = np.log(np.maximum(band, 1e-8))
        segments.append(np.concatenate((lp[:2].mean(axis=0), lp[2])))
    epochs = []
    for sec in go_onsets:
        idx = np.searchsorted(t, sec)
        if idx < 50 or idx + 125 > len(t):
            continue
        segment = pair[:, idx - 50:idx + 125]
        if np.max(np.ptp(segment, axis=1)) > 120:
            continue
        segment = segment - segment[:, :50].mean(axis=1, keepdims=True)
        channels = np.stack((segment.mean(axis=0), segment[0] - segment[1]))
        epochs.append([float(np.mean(row[50 + lo:50 + hi])) for row in channels
                       for lo, hi in ((20, 38), (45, 75), (75, 125))])
    spec = np.median(segments, axis=0).tolist() if len(segments) >= 3 else None
    erp = np.median(epochs, axis=0).tolist() if len(epochs) >= 4 else None
    return {"spec": spec, "erp": erp, "n_spectra": len(segments), "n_go": len(epochs)}


def coupling(eeg, eyes):
    return np.max(abs(np.corrcoef(np.vstack((eeg, eyes)))[:2, 2:]), axis=1).tolist()


def subject(job):
    root, person = job
    entries = source.archive_index(root, online=False)
    files, record, before, binary, after = source.member_groups(entries, person)
    folder = root / "downloads" / person
    metadata = source.read_metadata((folder / "before.part").read_bytes(), before[0], files)
    metadata.update(source.read_metadata((folder / "after.part").read_bytes(), after[0], files))
    nchan, indices, scales, rate, bad = source.channel_layout(metadata, LEADS)
    if rate != 512:
        raise ValueError("Unexpected sampling rate")
    parts = [folder / f"eeg-{i:03}.part" for i, _ in enumerate(range(*binary, source.CHUNK))]
    continuous = source.selected_channels(record, parts, nchan, indices, scales)
    cache = root / "two-channel-benchmark" / "raw_epochs" / person
    details, original = _verify_subject(cache, person)
    stimulus = [int(x["sample"]) for x in source.parse_tsv(source.file_ending(metadata, "_events.tsv"))
                if x["trial_type"] == "stimulus_go"]
    output = []
    for i, probe in enumerate(details["probes"]):
        onset, first = probe["onset_sample"], probe["start_sample"]
        raw = continuous[:, first:onset]
        if not np.array_equal(raw[:2], original[i]) or first < rate * probe["phase_start_sec"]:
            raise ValueError("Original two-channel cache, phase or timing mismatch")
        signal, t = grid(raw, rate, first, onset)
        go = [(event - onset) / rate for event in stimulus if
              onset - rate * 40 <= event <= onset - rate * 2.6]
        frontal, post = signal[:2], signal[2:4]
        eyes = np.stack((signal[4] - signal[5], signal[6] - signal[7]))
        train = (t >= -44) & (t < -41)
        score = t >= -18
        weights = np.linalg.lstsq(eyes[:, train].T, frontal[:, train].T, rcond=1e-5)[0]
        residual = frontal - weights.T @ eyes
        row = {"participant": person, "probe": probe["probe_id"], "label": probe["label"],
               "rating": probe["source_score"], "n_go_markers": len(go),
               "source_bad_channels": bad, "eeg_crc32": f"{record.CRC:08x}"}
        for tag, pair, lead_names in (("front", frontal, LEADS[:2]), ("posterior", post, LEADS[2:4]),
                                      ("eyes", eyes, LEADS[4:]), ("clean", residual, LEADS[:2] + LEADS[4:])):
            row[tag] = features(pair, t, go) if set(lead_names).isdisjoint(bad) else {}
        if set(LEADS[:2] + LEADS[4:]).isdisjoint(bad):
            row["raw_coupling"] = coupling(frontal[:, score], eyes[:, score])
            row["clean_coupling"] = coupling(residual[:, score], eyes[:, score])
        output.append(row)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    subjects = source.cohort(source.archive_index(args.root, online=False))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = [row for result in pool.map(subject, [(args.root, s) for s in subjects]) for row in result]
    assert len(rows) == 420 and len({(r['participant'], r['probe']) for r in rows}) == 420
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=1, allow_nan=False), encoding="utf8")
