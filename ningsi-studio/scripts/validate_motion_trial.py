"""Replay PhysioNet motion-artifact v1.0.0 EEG trial 1 with its experimenter trigger."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ningsi" / "src"))
from ningsi.signal.artifacts import check_channels
from ningsi.signal.live_artifacts import process_live_artifacts
from ningsi.signal.preprocess import preprocess


def review(path):
    # Official header eeg_1.hea: ten interleaved little-endian int16 streams,
    # nominal 2048 Hz, 1000 ADC units/mV for both EEG channels, baseline zero.
    source = np.memmap(path, dtype="<i2", mode="r").reshape(1106064, 10)
    raw_rate = 2048
    rate = 256
    taps = np.arange(-64, 65)
    h = np.sinc(2 * 90 / raw_rate * taps) * np.hamming(len(taps))
    h /= h.sum()
    eeg = np.stack([np.convolve(row, h, "same")[::8] for row in source[:, :2].T])
    trigger = source[::8, 9]  # synchronized ACC trigger; low marks induced motion.
    cutoff = (np.quantile(trigger, .1) + np.quantile(trigger, .9)) / 2
    groups = {"motion": Counter(), "undisturbed_period": Counter()}
    for end in range(6*rate, len(trigger)-rate, 4*rate):
        status = trigger[end-4*rate:end] > cutoff
        if np.any(status) != np.all(status):
            continue  # exclude the four-second windows that straddle a transition.
        group = groups["undisturbed_period" if np.all(status) else "motion"]
        raw = eeg[:, end-6*rate:end]
        filtered = np.stack([preprocess(row-row.mean(), rate)[0] for row in raw])
        out, artifact = process_live_artifacts(filtered, raw, rate, 4*rate, ("EEG1", "EEG2"))
        legacy_ok = check_channels(out, rate).ok
        group.update(total=1, excluded_legacy=int(not legacy_ok),
                     excluded_new=int(not legacy_ok or bool(artifact["reasons"])),
                     excluded_transient_only=int(bool(artifact["reasons"])))
    return {"source": "https://physionet.org/content/motion-artifact/1.0.0/", "record": "eeg_1",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "groups": groups,
            "method": "First public record, nonoverlapping 4 s windows, one 6 s context, two EEG channels, unit uV. ACC experimenter trigger supplies labels only.",
            "scope": "One recording. Accelerometer WFDB unit is nd; this experiment does not evaluate the physical-g motion threshold. A motion-labelled period may contain intermittent motion."}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    args.output.write_text(json.dumps(review(args.data), indent=2) + "\n", encoding="utf8")
