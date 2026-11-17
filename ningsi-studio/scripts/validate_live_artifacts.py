"""Replay one prespecified 6 s segment per public ATTLAPSE probe, with and without EOG."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "ningsi" / "src"), str(ROOT / "ningsi-capture" / "scripts")]
from attlapse_data import DOI, iter_probes
from ningsi.signal.artifacts import check_channels
from ningsi.signal.live_artifacts import SPEC, process_live_artifacts
from ningsi.signal.preprocess import preprocess
from ningsi.signal.bands import band_powers
from ningsi.signal.spectrum import welch_psd


def coupling(eeg, eog):
    # Max absolute correlation of each EEG lead with the measured VEOG/HEOG.
    return np.abs(np.corrcoef(np.vstack((eeg, eog)))[:2, 2:]).max(axis=1)


def review(root):
    count = eligible = before_pass = after_pass = 0
    seen, valid_people = set(), set()
    before, after, altered = [], [], []
    low_eog = Counter()
    power_delta = {name: [] for name in ("theta", "alpha", "beta")}
    flags, skipped = Counter(), Counter()
    names = ["Fp1", "Fp2", "eog_up", "eog_down", "eog_left", "eog_right"]
    for row in iter_probes(root, with_aux=True):
        count += 1
        seen.add(row["participant"])
        if row["bad_channels"] or set(row["bad_aux_channels"]).intersection(names[2:]):
            skipped["source_flagged_bad_lead"] += 1
            continue
        if not set(names[2:]).issubset(row["aux_channels"]):
            skipped["reference_missing"] += 1
            continue
        rate = row["rate"]
        indices = [row["aux_channels"].index(k) for k in names[2:]]
        if any(row["aux_units"][i] not in ("µV", "uV") for i in indices):
            skipped["reference_unit_unknown"] += 1
            continue
        samples = np.vstack((row["values_uv"], row["aux_values"][indices]))[:, -6*rate:]
        if not np.isfinite(samples).all() or samples.shape[1] != 6*rate:
            skipped["invalid_segment"] += 1
            continue
        filtered = np.stack([preprocess(x - x.mean(), rate)[0] for x in samples])
        plain = filtered[:2, -4*rate:]
        eyes = np.stack((filtered[2]-filtered[3], filtered[4]-filtered[5]))[:, -4*rate:]
        if np.min(np.std(eyes, axis=1)) < 1e-9 or np.min(np.std(plain, axis=1)) < 1e-9:
            skipped["flat_channel"] += 1
            continue
        cleaned, info = process_live_artifacts(filtered, samples, rate, 4*rate, names, ["uV"]*6)
        _, no_reference = process_live_artifacts(filtered[:2], samples[:2], rate, 4*rate, names[:2], ["uV"]*2)
        before_pass += check_channels(plain, rate).ok and not no_reference["reasons"]
        after_pass += check_channels(cleaned, rate).ok and not info["reasons"]
        if np.max(np.std(eyes, axis=1)) < 12:
            low_eog.update(probes=1, leads_corrected=len(info["ocular_corrected_channels"]),
                           new_proxy_flags_without_eog=int(bool(no_reference["reasons"])))
            for original, adjusted in zip(plain, cleaned):
                p0, p1 = welch_psd(original, rate), welch_psd(adjusted, rate)
                b0, b1 = band_powers(p0.freqs, p0.psd), band_powers(p1.freqs, p1.psd)
                for band, changes in power_delta.items():
                    changes.append(abs(10*np.log10(max(b1[band], 1e-12) / max(b0[band], 1e-12))))
        valid_people.add(row["participant"])
        eligible += 1
        before.extend(coupling(plain, eyes))
        after.extend(coupling(cleaned, eyes))
        altered.extend(i in info["ocular_corrected_channels"] for i in range(2))
        flags.update(no_reference["reasons"])
    mask = np.array(altered, dtype=bool)
    old, new = np.array(before), np.array(after)
    def groups(subset):
        if not subset.any():
            return {}
        return {"lead_windows": int(subset.sum()), "median_before": float(np.median(old[subset])),
                "median_after": float(np.median(new[subset])),
                "reduced": int(np.count_nonzero(new[subset] < old[subset]-1e-9)),
                "increased": int(np.count_nonzero(new[subset] > old[subset]+1e-9))}
    return {"spec": SPEC, "dataset_doi": DOI, "participants": len(seen), "probes": count,
            "timing": "Last six cached pre-probe seconds. Coefficients fitted to first two seconds, next four evaluated.",
            "eligible_participants": len(valid_people), "eligible_probes": eligible, "skipped": skipped,
            "coupling_measure": "Per Fp lead, maximum absolute correlation with synchronized bipolar VEOG and HEOG.",
            "all_eeg_leads": groups(np.ones(len(mask), dtype=bool)), "corrected_leads": groups(mask),
            "accepted_without_reference": before_pass, "accepted_with_reference": after_pass,
            "eeg_only_transient_flags": flags,
            "low_eog": {"definition": "Both held-out bipolar reference SDs below 12 uV. Other artifacts unlabelled.",
                        **low_eog, "absolute_band_change_db": {
                            band: dict(median=float(np.median(v)), p95=float(np.quantile(v, .95)))
                            for band, v in power_delta.items() if v}},
            "scope": "Reference coupling is a contamination proxy. Clean EEG ground truth, IMU, EMG and local-device transfer were unavailable."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = review(args.cache_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
