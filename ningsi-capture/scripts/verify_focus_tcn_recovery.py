"""Paired engineering replay of frozen Capture scores on saved EEG.

The local recordings are unlabeled signal checks. Virtual probe cutoffs
exercise the arrival-time contract, and the report never treats a score
as a real participant response. The optional public replay measures input
and score preservation only, because the shipped models were refit on it.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "ningsi" / "src"), str(ROOT / "ningsi-capture" / "src")]

from attlapse_two_channel_250 import iter_probes
from ningsi.acquisition.timing import (BIOMULTI_CONFIRM_SAMPLES, BIOMULTI_LSL_TIMING, BIOMULTI_LSL_RECOVERY_TIMING,
                                       BIOMULTI_MAX_DELAY, biomulti_packet_clock, biomulti_recovery_clock)
from ningsi.models import attlapse_shadow, focus_tcn, focus_tcn_recovery
from ningsi_capture.export import analysis_windows
from ningsi_capture.probe_prediction import ProbePredictions, _tcn_score
from ningsi_capture.storage import open_readonly, read_metadata


FIELD = ROOT / "var" / "tcn-field-20261118-01"
PUBLIC = ROOT / "var" / "attlapse" / "two-channel-benchmark"
RATE = attlapse_shadow.RATE
WINDOW_SAMPLES = 4 * RATE
STEP_SAMPLES = 2 * RATE
HISTORY_SAMPLES = 6 * RATE


def analyze_with(compute):
    def analyze(context, source):
        values = np.stack([values for values, _ in context]).T
        timestamps = np.asarray([stamp for _, stamp in context], dtype=float)
        return compute(values, timestamps, source["srate"], source["scale_to_uv"], (0, 1))
    return analyze


def window_stats(rows):
    passed = [not row["physical_reasons"] for row in rows]
    nominal_groups = [sum(passed[end - 5:end + 1]) for end in range(5, len(passed))]
    return {
        "windows": len(rows),
        "physical_passes": sum(passed),
        "strict_passes": sum(not row["strict_reasons"] for row in rows),
        "physical_rejections": dict(Counter(reason for row in rows for reason in row["physical_reasons"])),
        "max_usable_in_six": max(nominal_groups, default=0),
        "six_window_groups_with_four_or_more_usable": sum(total >= 4 for total in nominal_groups),
    }


def virtual_cutoffs(reader, step_sec):
    arrivals = np.asarray(reader.execute(
        "SELECT received_lsl,received_perf FROM samples ORDER BY sample_index"
    ).fetchall(), dtype=float)
    if (len(arrivals) < 2 or not np.isfinite(arrivals).all()
            or np.any(np.diff(arrivals[:, 1]) < 0)):
        raise ValueError("Expected finite, nondecreasing recorded arrival clocks")
    lsl_start, perf_start = arrivals[0]
    for elapsed in np.arange(20., arrivals[-1, 1] - perf_start, step_sec):
        perf = perf_start + elapsed
        count = int(np.searchsorted(arrivals[:, 1], perf, side="right"))
        yield elapsed, SimpleNamespace(lsl=lsl_start + elapsed, perf=perf), count, lsl_start


def probe_stats(results):
    return {
        "cuts": len(results),
        "scored": sum("mw_score" in row for row in results),
        "usable_frame_counts": dict(Counter(row.get("windows_usable", 0) for row in results)),
        "rejections": dict(Counter(row["reason"] for row in results if "reason" in row)),
    }


def replay_field(path, step_sec):
    with closing(open_readonly(path)) as reader:
        reader.execute("BEGIN")
        meta = read_metadata(reader)
        source = meta["source"]
        if source.get("timestamp_policy") != BIOMULTI_LSL_TIMING or source.get("channels") != 2:
            raise ValueError("Expected a recorded two-channel BioMulti packet-clock source")
        raw_counts = reader.execute(
            "SELECT count(*),sum(corrected_timestamp IS NULL) FROM samples"
        ).fetchone()
        known_montage = [name.casefold() for name in source.get("channel_labels", ())] == ["fp1", "fp2"]
        scenarios = {
            "legacy": (BIOMULTI_LSL_TIMING, attlapse_shadow.compute_window),
            "timing_only": (BIOMULTI_LSL_RECOVERY_TIMING, attlapse_shadow.compute_window),
            "signal_only": (BIOMULTI_LSL_TIMING, focus_tcn_recovery.compute_window),
            "combined": (BIOMULTI_LSL_RECOVERY_TIMING, focus_tcn_recovery.compute_window),
        }
        arms = {}
        for name, (timing, compute) in scenarios.items():
            variants_meta = {**meta, "source": {**source, "timestamp_policy": timing}}
            windows = analysis_windows(reader, variants_meta, analyze=analyze_with(compute))
            arms[name] = window_stats(windows)
        result = {
            "samples": raw_counts[0],
            "missing_corrected_timestamps": raw_counts[1],
            "as_recorded_montage_confirmed": known_montage,
            "window_arms": arms,
        }
        if known_montage:
            model = focus_tcn.load_candidate()
            before, after = [], []
            with closing(ProbePredictions(path.parent, meta, lambda *_: None)) as runtime:
                for _, stamp, count, lsl_start in virtual_cutoffs(reader, step_sec):
                    for lane, use_recovery in [(before, False), (after, True)]:
                        if use_recovery:
                            data = runtime._shadow_windows(reader, stamp, count, (lsl_start, 0), recovery=True)
                        else:
                            data = runtime._shadow_windows(reader, stamp, count, (lsl_start, 0))
                        lane.append(data if "reason" in data else _tcn_score(
                            model, data["windows"], data["last_end"], stamp.lsl, lsl_start))
            result["virtual_cuts"] = {"legacy": probe_stats(before), "combined": probe_stats(after)}
            result["virtual_cuts"]["paired"] = dict(Counter(
                f"{'score' if 'mw_score' in a else 'reject'}_to_{'score' if 'mw_score' in b else 'reject'}"
                for a, b in zip(before, after, strict=True)))
        return result


def public_row(probe, end, compute):
    start = max(probe["start_sample"], end - HISTORY_SAMPLES) - probe["start_sample"]
    stop = end - probe["start_sample"]
    values = probe["values_uv"][:, start:stop]
    times = probe["timestamps"][start:stop]
    row = compute(values, times, RATE, 1., (0, 1))
    return {**row, "sample_end_index": end - 1}


def replay_public(data):
    model = focus_tcn.load_candidate()
    paired = Counter()
    both_score_deltas = []
    newly_passing_frames = 0
    original_passing_frames = 0
    original_pass_features_changed = 0
    for probe in iter_probes(data):
        onset = probe["onset_sec"]
        left, right = (onset + offset for offset in attlapse_shadow.LOOKBACK)
        if probe["phase_start_sec"] > left:
            paired["out_of_phase"] += 1
            continue
        first = math.ceil((left * RATE + WINDOW_SAMPLES) / STEP_SAMPLES - 1e-9) * STEP_SAMPLES
        last = math.floor(right * RATE / STEP_SAMPLES + 1e-9) * STEP_SAMPLES
        originals, revised = [], []
        for end in range(first, last + 1, STEP_SAMPLES):
            if end > probe["stop_sample"] or end - WINDOW_SAMPLES < probe["start_sample"]:
                continue
            before = public_row(probe, end, attlapse_shadow.compute_window)
            after = public_row(probe, end, focus_tcn_recovery.compute_window)
            originals.append(before)
            revised.append(after)
            if not before["physical_reasons"]:
                original_passing_frames += 1
                original_pass_features_changed += before["features"] != after["features"]
            newly_passing_frames += bool(before["physical_reasons"] and not after["physical_reasons"])
        a, b = (_tcn_score(model, rows, last - 1, onset, probe["phase_start_sec"])
                for rows in (originals, revised))
        scored_a, scored_b = "mw_score" in a, "mw_score" in b
        paired[(scored_a, scored_b)] += 1
        if scored_a and scored_b:
            both_score_deltas.append(abs(a["mw_score"] - b["mw_score"]))
    return {
        "scope": "input preservation on the public source used for refitting, no performance estimate",
        "probe_counts": {
            f"{'score' if key[0] else 'reject'}_to_{'score' if key[1] else 'reject'}" if isinstance(key, tuple)
            else key: val for key, val in paired.items()
        },
        "original_passing_frames": original_passing_frames,
        "original_pass_features_changed": original_pass_features_changed,
        "newly_passing_frames": newly_passing_frames,
        "median_absolute_paired_score_change": float(np.median(both_score_deltas)) if both_score_deltas else None,
        "max_absolute_paired_score_change": max(both_score_deltas, default=None),
    }


def verify_clock_sentinels():
    """Keep genuine missing packets distinct from delayed releases that catch up."""
    nominal = 100. + np.arange(3000) / RATE
    start = 1000
    losses = {}
    for delay in (.12, .30):
        # A permanent offset is equivalent to the recorded timeline whose
        # intervening packets never appeared in the captured sample array.
        source = nominal.copy()
        source[start:] += delay
        legacy = biomulti_packet_clock(source)
        revised = biomulti_recovery_clock(source)
        clipped = revised[start:start + 10]
        if np.isfinite(clipped).any():
            raise AssertionError("Revision filled an unresolved packet after a true gap")
        shared = np.isfinite(revised)
        if np.max(np.abs(revised[shared] - source[shared])) > BIOMULTI_MAX_DELAY:
            raise AssertionError("Revised clock exceeded its maximum shift")
        losses[str(round(delay * 1000))] = {
            "legacy_finite_samples": int(np.isfinite(legacy).sum()),
            "revised_finite_samples": int(shared.sum()),
            "revised_first_post_gap_packet_finite_samples": int(np.isfinite(clipped).sum()),
        }
    delayed = nominal.copy()
    delayed[start:start + 40] += np.repeat([.12, .09, .06, .03], 10)
    estimate = biomulti_recovery_clock(delayed)
    if not np.isfinite(estimate[start:start + 40]).all():
        raise AssertionError("Revision rejected the synthetic bounded catch-up sequence")
    prefix = 2000
    changed = delayed.copy()
    changed[prefix:] += .3
    confirmed = prefix - BIOMULTI_CONFIRM_SAMPLES
    stable = np.array_equal(estimate[:confirmed], biomulti_recovery_clock(changed)[:confirmed], equal_nan=True)
    if not stable:
        raise AssertionError("Post-cutoff releases changed confirmed timestamps")
    return {"permanent_gap_milliseconds": losses,
            "buffered_then_caught_up_samples": 40,
            "confirmed_prefix_unchanged_when_future_releases_changed": stable}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", type=Path, default=FIELD)
    parser.add_argument("--public", type=Path, nargs="?", const=PUBLIC)
    parser.add_argument("--step-sec", type=float, default=4.)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.step_sec) or args.step_sec <= 0:
        parser.error("--step-sec must be positive and finite")
    runs = {path.name.removesuffix("-export"): path / "recording.sqlite3"
            for path in sorted(args.field.glob("preflight*-export"))}
    runs.update({f"continuous_{index}": path for index, path in
                 enumerate(sorted((args.field / "capture").glob("*/recording.sqlite3")), 1)})
    if not runs:
        parser.error("No field recordings found")
    report = {"virtual_probe_step_sec": args.step_sec,
              "scope": "engineering coverage on unlabeled field waveforms; no recorded thought responses",
              "synthetic_clock_sentinels": verify_clock_sentinels(),
              "recordings": {}}
    for name, path in runs.items():
        result = replay_field(path, args.step_sec)
        with path.open("rb") as stream:
            result["recording_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        report["recordings"][name] = result
        print(name, "physical window passes", {k: v["physical_passes"] for k, v in result["window_arms"].items()},
              "virtual scores", result.get("virtual_cuts", {}).get("paired"), flush=True)
    if args.public is not None:
        report["public_sensitivity"] = replay_public(args.public)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
