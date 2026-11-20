"""Local integrity flags and frequency estimates for periods of a recording."""

import json

import numpy as np

from .quality_integrity import audit_integrity
from .spectrum_quality import describe_interval_spectrum


POLICY = "capture-local-quality-review-v1"
SPAN_FIELDS = ("channel", "start_sample_index", "stop_sample_index", "start_lsl", "end_lsl", "reason")
WINDOW_FIELDS = ("sample_end_index", "start_lsl", "end_lsl", "clock_and_samples_intact",
                 "strict_usable", "dual_spectral_status", "eeg_quality_flags")
PERIOD_FIELDS = ("kind", "phase", "probe_id", "start_lsl", "end_lsl", "status", "reason",
                 "support_sec", "support_fraction", "segments", "rel_theta", "rel_alpha", "rel_beta",
                 "intervals_lsl")


class _Timeline:
    def __init__(self, values, stamps, rate, scale, motion, invalid_spans):
        self.values = values
        self.stamps = stamps.copy()
        self.rate, self.scale, self.motion = rate, scale, motion
        for span in invalid_spans:
            self.stamps[span["start_sample_index"]:span["stop_sample_index"]] = np.nan
        self.finite = np.flatnonzero(np.isfinite(self.stamps))
        self.times = self.stamps[self.finite]
        self.ordered = np.all(np.diff(self.times) > 0)

    def spectrum(self, start, stop):
        if self.ordered:
            left = np.searchsorted(self.times, start, side="left")
            right = np.searchsorted(self.times, stop - 1 / self.rate, side="right")
            eligible = self.finite[left:right]
        else:
            # Resolve each period independently. An ambiguous period that straddles
            # a clock reset is still rejected by the spectrum estimator.
            eligible = np.flatnonzero((self.stamps >= start) & (self.stamps <= stop - 1 / self.rate))
        if not len(eligible):
            return None
        first, last = int(eligible[0]), int(eligible[-1]) + 1
        history = round(6 * self.rate) - round(4 * self.rate)
        context = max(0, first - history)
        return describe_interval_spectrum(self.values[:, context:last], self.stamps[context:last],
                                          self.rate, self.scale, first=first - context, motion=self.motion)


def _period(timeline, kind, phase, start, stop, *, probe_id="", eligible=True):
    row = {"kind": kind, "phase": phase, "probe_id": probe_id,
           "start_lsl": start if np.isfinite(start) else None, "end_lsl": stop if np.isfinite(stop) else None,
           "status": "unavailable", "reason": "outside_task_phase", "support_sec": 0.,
           "support_fraction": 0., "segments": 0, "intervals_lsl": "[]"}
    if row["start_lsl"] is None or row["end_lsl"] is None or stop <= start:
        row["reason"] = "invalid_period_clock"
        return row
    if not eligible:
        return row
    if timeline is None:
        row["reason"] = "continuous_analysis_clock_unavailable"
        return row
    if timeline.scale is None:
        row["reason"] = "eeg_units_or_type_unconfirmed"
        return row
    result = timeline.spectrum(start, stop)
    if result is None:
        row["reason"] = "clock_span_unavailable"
        return row
    if result["status"] == "unavailable":
        row["reason"] = "no_screened_two_second_segment"
        return row
    intervals = result["intervals"]
    support = sum(right - left for left, right in intervals)
    row.update(status="descriptive", reason="", support_sec=round(support, 6),
               support_fraction=round(min(1., support / (stop - start)), 6),
               segments=result["segments"], intervals_lsl=json.dumps(intervals, separators=(",", ":")),
               **{f"rel_{band}": result["relative_bands"].get(band) for band in ("theta", "alpha", "beta")})
    return row


def build_review(values, stamps, rate, scale, windows, phase_starts, phase_ends, probes, *, motion=None,
                 continuous_clock=True):
    """Review signal retention and estimate bands inside independently bounded periods."""
    integrity = audit_integrity(values, stamps, rate)
    width = round(4 * rate)
    window_rows = []
    for row in windows:
        stop = row["sample_end_index"] + 1
        window_rows.append({
            "sample_end_index": stop - 1, "start_lsl": row["start"], "end_lsl": row["end"],
            "clock_and_samples_intact": integrity.window_intact(stop - width, stop),
            "strict_usable": row["window"].usable,
            "dual_spectral_status": row.get("descriptive_spectrum", {}).get("status", "unavailable"),
            "eeg_quality_flags": "|".join(row["window"].quality.reasons),
        })
    timeline = _Timeline(values, stamps, rate, scale, motion, integrity.spans) if continuous_clock else None
    periods = []
    for kind, phase in (("baseline", "baseline_calibration"), ("baseline_holdout", "baseline_holdout")):
        if phase in phase_starts and phase in phase_ends:
            periods.append(_period(timeline, kind, phase, phase_starts[phase], phase_ends[phase]))
    for probe in probes:
        if probe["practice"]:
            continue
        phase = probe["phase"]
        left, right = probe["window_start_lsl"], probe["window_end_lsl"]
        eligible = (phase in phase_starts and phase in phase_ends
                    and phase_starts[phase] <= left < right <= phase_ends[phase])
        periods.append(_period(timeline, "probe", phase, left, right, probe_id=probe["probe_id"], eligible=eligible))
    probe_periods = [row for row in periods if row["kind"] == "probe"]
    summary = {
        "policy": POLICY,
        "scope": "recording_integrity_and_descriptive_frequency_only",
        "motion_screening": "matched_gyro" if motion is not None else "eeg_only",
        "windows_total": len(window_rows),
        "windows_strict": sum(row["strict_usable"] for row in window_rows),
        "windows_integrity_only": sum(row["clock_and_samples_intact"] for row in window_rows),
        "windows_dual_spectral": sum(row["dual_spectral_status"] in ("full", "partial") for row in window_rows),
        "local_integrity_flags": len(integrity.spans),
        "baseline": next((row for row in periods if row["kind"] == "baseline"), None),
        "formal_probes_total": len(probe_periods),
        "formal_probes_with_descriptive_spectrum": sum(row["status"] == "descriptive" for row in probe_periods),
        "formal_probes_with_at_least_6_sec": sum(row["support_sec"] >= 6 for row in probe_periods),
    }
    return summary, integrity.spans, window_rows, periods
