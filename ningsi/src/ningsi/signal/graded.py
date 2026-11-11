"""Conservative descriptive salvage; never changes strict quality or scores.

Two-second channel fragments use existing screening thresholds. All new rules
are engineering choices awaiting device validation, not artifact correction.
"""
import math
from statistics import median

import numpy as np

from ningsi import config
from ningsi.signal.artifacts import check_channels
from ningsi.signal.bands import relative_band_powers
from ningsi.signal.spectrum import welch_psd


def describe_fragments(raw, srate, strict_ok=False):
    data = np.atleast_2d(np.asarray(raw, dtype=float))
    channels = []
    complete = True
    size = round(2 * srate)
    if data.ndim != 2 or size < 2 or data.shape[1] < 2 * size:
        return {"grade": "C", "channels": [], "reason": "insufficient_data"}
    for index, channel in enumerate(data):
        kept = []
        intervals = []
        for start in range(0, len(channel) - size + 1, size):
            fragment = channel[start:start + size]
            if not np.isfinite(fragment).all():
                continue
            # Repeated extrema indicate possible clipping; do not interpolate.
            if max(np.mean(fragment == fragment.min()), np.mean(fragment == fragment.max())) >= .1:
                continue
            spectrum = welch_psd(fragment, srate)
            if check_channels(fragment, srate, spectra=[spectrum]).ok:
                kept.append(relative_band_powers(spectrum.freqs, spectrum.psd))
                intervals.append([(start-len(channel))/srate, (start+size-len(channel))/srate])
        complete = complete and len(kept) == len(channel) // size
        if kept:
            channels.append({"channel": index, "fragments": len(kept),
                             "intervals": intervals,
                             "relative": {band: median(row[band] for row in kept) for band in kept[0]}})
    return {"grade": "A" if strict_ok and complete and len(channels) == len(data) else "B" if channels else "C",
            "channels": channels, "threshold_status": "not_validated",
            "note": "仅保留通过片段筛查的同通道频带；不能解释为注意力评分"}


def robust_reference(rows):
    """Same-channel log ratios, median/MAD; no conversion into a clinical score."""
    grouped = {}
    coverage = {}
    for row in rows:
        for channel in row.get("channels", []):
            rel = channel["relative"]
            value = math.log(max(rel.get("beta", 0), config.INDEX_FLOOR) /
                             max(rel.get("theta", 0), config.INDEX_FLOOR))
            grouped.setdefault(str(channel["channel"]), []).append(value)
            if "t_end" in row:
                coverage.setdefault(str(channel["channel"]), []).extend(
                    (row["t_end"]+left, row["t_end"]+right) for left,right in channel.get("intervals", []))
    return {key: {"median": median(values), "mad": median(abs(v - median(values)) for v in values),
                  "n": len(values), "usable_seconds": union_seconds(coverage.get(key, [])),
                  "metric": "log_beta_theta", "validated": False}
            for key, values in grouped.items()}


def union_seconds(intervals):
    total, end = 0., -math.inf
    for left, right in sorted(intervals):
        total += max(0., right-max(left,end))
        end = max(end,right)
    return round(total, 3)


class FeedbackGate:
    """Downgrade immediately; require three consecutive windows to upgrade."""
    def __init__(self):
        self.mode = "guided"
        self.pending = None
        self.count = 0

    def update(self, grade, baseline_valid):
        desired = "eeg" if grade == "A" and baseline_valid else "quality" if grade != "C" else "guided"
        rank = {"guided": 0, "quality": 1, "eeg": 2}
        if desired == self.pending:
            self.count += 1
        else:
            self.pending, self.count = desired, 1
        if rank[desired] <= rank[self.mode] or self.count >= 3:
            self.mode = desired
        return self.mode
