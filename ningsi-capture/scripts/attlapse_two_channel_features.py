"""Offline feature comparisons on original ATTLAPSE Fp1 and Fp2 (ordered, µV).

Run on the verified two-channel source cache. Filters see at most six seconds
ending at each aligned window. A probe's task answer only accompanies the
output. The three representations use exactly the same windows within a gate.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import importlib
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ningsi" / "src"))
from ningsi import config
from ningsi.models.logistic import FEATURE_NAMES, features_from_window
from ningsi.models.research_features import CONTRACT, union_seconds
from ningsi.signal.artifacts import check_channels
from ningsi.signal.bands import band_powers, relative_band_powers
from ningsi.signal.preprocess import preprocess
from ningsi.signal.spectrum import welch_psd


BANDS = ("theta", "alpha", "beta")
CHANNELS = ("fp1", "fp2")
WINDOWS = {"pre12_2": (-12, -2), "pre18_2": (-18, -2), "pre32_2": (-32, -2)}
GATES = ("strict", "bipolar_strict", "physical")
BIPOLAR = tuple(FEATURE_NAMES)
BILATERAL = BIPOLAR + tuple(f"{ch}_rel_{band}" for ch in CHANNELS for band in BANDS) + tuple(
    f"fp2_minus_fp1_log_power_{band}" for band in BANDS)
FEATURE_SETS = {
    "bipolar6": BIPOLAR,
    "bilateral15": BILATERAL,
    "bilateral_variability18": BILATERAL + tuple(f"sd_bipolar_rel_{band}" for band in BANDS),
}
SPECTRAL_PROXY_FLAGS = frozenset(("eog", "emg"))


def _window(raw, rate):
    # All filters in the production chain are linear. Subtracting the two
    # conditioned inputs gives the same bipolar signal to numerical precision.
    size = round(CONTRACT["window_sec"] * rate)
    if not np.isfinite(raw).all():
        return {}, {"non_finite_signal"}, {"non_finite_signal"}
    pair = np.stack([preprocess(ch - ch.mean(), rate)[0][-size:] for ch in raw])
    bipolar = pair[0:1] - pair[1:2]
    pair_spectra = [welch_psd(ch, rate) for ch in pair]
    bipolar_spectrum = welch_psd(bipolar[0], rate)
    pair_qc = check_channels(pair, rate, spectra=pair_spectra)
    bipolar_qc = check_channels(bipolar, rate, spectra=[bipolar_spectrum])
    relative = relative_band_powers(bipolar_spectrum.freqs, bipolar_spectrum.psd)
    features = features_from_window(SimpleNamespace(rel=relative))
    powers = []
    for channel, spectrum in zip(CHANNELS, pair_spectra, strict=True):
        rel = relative_band_powers(spectrum.freqs, spectrum.psd)
        features.update({f"{channel}_rel_{band}": rel[band] for band in BANDS})
        powers.append(band_powers(spectrum.freqs, spectrum.psd))
    for band in BANDS:
        features[f"fp2_minus_fp1_log_power_{band}"] = (
            math.log(max(powers[1][band], config.INDEX_FLOOR))
            - math.log(max(powers[0][band], config.INDEX_FLOOR)))
    return features, set(pair_qc.reasons), set(bipolar_qc.reasons)


def _flags(pair, bipolar):
    def names(p, b):
        return [f"input_{reason}" for reason in sorted(p)] + [f"bipolar_{reason}" for reason in sorted(b)]
    return {"strict": names(pair, bipolar),
            "bipolar_strict": names(pair - SPECTRAL_PROXY_FLAGS, bipolar),
            "physical": names(pair - SPECTRAL_PROXY_FLAGS, bipolar - SPECTRAL_PROXY_FLAGS)}


def extract_features(probe):
    """Return 9 JSON-ready rows, with metadata, QC and a shared feature map."""
    rate = float(probe["rate"])
    values = np.asarray(probe["values_uv"], dtype=float)
    times = np.asarray(probe["timestamps"], dtype=float)
    if rate not in (250, 512) or len(times) == 0 or values.shape != (2, len(times)):
        raise ValueError("Expected two Fp1/Fp2 channels at 512 Hz or 250 Hz.")
    if probe["source"] != "sourcedata" or probe["unit"] != "µV" or list(probe["channels"]) != ["Fp1", "Fp2"]:
        raise ValueError("Expected the verified original Fp1/Fp2 sources in microvolts.")
    start = int(probe["start_sample"])
    onset_sample = int(probe["onset_sample"])
    onset = float(probe["onset_sec"])
    if not math.isfinite(onset) or not np.allclose(times, (start + np.arange(len(times))) / rate, atol=1e-7, rtol=0):
        raise ValueError("Source sample grid or probe time does not match the raw recording.")
    if "aux_values" in probe:
        raise ValueError("Use the raw two-channel iterator without auxiliary channels.")
    common = {k: probe[k] for k in ("participant", "probe_id", "label", "source_answer", "source_score")}
    common.update(onset_sec=onset, onset_sample=onset_sample, phase_start_sec=probe["phase_start_sec"],
                  bad_channels=list(probe["bad_channels"]), rate=rate)
    size = round(CONTRACT["window_sec"] * rate)
    hop = round(CONTRACT["step_sec"] * rate)
    context = round(CONTRACT["context_sec"] * rate)
    calculated = {}
    output = []
    for name, (left, right) in WINDOWS.items():
        lower = onset + left
        upper = onset + right
        first_end = max(size, math.ceil((lower * rate + size) / hop) * hop)
        ends = list(range(first_end, math.floor(upper * rate) + 1, hop))
        for end in ends:
            if end in calculated or probe["bad_channels"]:
                continue
            lo, hi = max(0, end - context) - start, end - start
            if lo < 0 or hi > len(times):
                raise ValueError(f"Raw excerpt lacks the requested conditioning history for {name}.")
            features, pair, bip = _window(values[:, lo:hi], rate)
            calculated[end] = (features, _flags(pair, bip))
        for gate in GATES:
            reasons = []
            counts = Counter()
            accepted = []
            if probe["bad_channels"]:
                reasons.append("source_channel_marked_bad:" + ",".join(probe["bad_channels"]))
            else:
                for end in ends:
                    rejected = calculated[end][1][gate]
                    counts.update(rejected)
                    if not rejected:
                        accepted.append(end)
            seconds = union_seconds(((end - size) / rate, end / rate) for end in accepted)
            if not probe["bad_channels"] and (len(accepted) < CONTRACT["min_windows"] or seconds < CONTRACT["min_seconds"]
                                                or len(accepted) < CONTRACT["min_valid_ratio"] * len(ends)):
                reasons.append("insufficient_usable_eeg")
            if probe["phase_start_sec"] > lower:
                reasons.append("lookback_crosses_phase")
            features = {}
            if not reasons:
                window_features = [calculated[end][0] for end in accepted]
                features = {key: float(np.mean([f[key] for f in window_features])) for key in BILATERAL}
                for band in BANDS:
                    features[f"sd_bipolar_rel_{band}"] = float(np.std([f[f"rel_{band}"] for f in window_features], ddof=0))
            output.append({**common, "window": name, "quality": gate, "window_start_sec": lower,
                           "window_end_sec": upper, "windows_total": len(ends) if not probe["bad_channels"] else 0,
                           "windows_usable": len(accepted), "usable_seconds": seconds,
                           "accepted_window_end_samples": accepted, "window_rejections": dict(counts),
                           "reasons": "|".join(reasons), "features": features})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Destination JSONL path")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--waveform-source", default="attlapse_two_channel_data",
                        help="Python module with iter_probes(data_root) yielding verified original two-channel inputs")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    iter_probes = importlib.import_module(args.waveform_source).iter_probes
    args.output.parent.mkdir(parents=True, exist_ok=True)
    total = Counter()
    keys = set()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.output.parent, delete=False) as dest:
            tmp_path = Path(dest.name)
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for rows in pool.map(extract_features, iter_probes(args.data), chunksize=2):
                    first = rows[0]
                    key = (first["participant"], first["probe_id"])
                    if key in keys:
                        raise ValueError(f"Duplicate probe: {key}")
                    keys.add(key)
                    total[key[0]] += 1
                    for row in rows:
                        dest.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    if len(keys) % 15 == 0:
                        print(f"Extracted {len(keys)} probes", flush=True)
        if len(total) != 28 or set(total.values()) != {15}:
            raise ValueError(f"Incomplete healthy-control extraction: {dict(total)}")
        os.replace(tmp_path, args.output)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
