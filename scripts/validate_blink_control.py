"""Reproducible engineering checks for calibrated dual-frontal blink control.

All signals are synthetic mixtures with known, independently generated EEG.
These results measure software behavior, never human attention validity or
recovery of real brain activity. Run from the suite root with workspace Python.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for package in ("ningsi", "ningsi-studio"):
    sys.path.insert(0, str(ROOT / package / "src"))

from ningsi.signal.baseline import build_baseline, index_values
from ningsi.signal.blinks import BlinkCalibration, calibrate_blinks, process_blinks
from ningsi.signal.preprocess import preprocess
from ningsi.signal.window import analyze_window


RATE = 250
LABELS = ("Fp1", "Fp2")
MONTAGE = "synthetic-dual-frontal-reference-v1"
OFFSET = np.array([-230000., -215000.])[:, None]


def background(seconds, seed):
    """Shared and distinct oscillations; each trial has independent phases/noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(round(seconds * RATE)) / RATE
    common = 4 * np.sin(2 * np.pi * 10 * t + rng.uniform(-np.pi, np.pi))
    rows = []
    for _ in LABELS:
        phase = rng.uniform(-np.pi, np.pi, 3)
        amplitude = np.array([6., 10., 5.]) * rng.uniform(.9, 1.1, 3)
        rows.append(common + sum(a * np.sin(2 * np.pi * hz * t + p)
                                 for a, hz, p in zip(amplitude, (6., 10., 20.), phase))
                    + rng.normal(0, .7, len(t)))
    return np.stack(rows) + OFFSET


def add_blink(data, peak_sec, amplitude=125., width=.10, ratio=.78, lag=0.):
    """A smooth biphasic blink, with a small recovery tail and per-channel delay."""
    out = data.copy()
    t = np.arange(out.shape[1]) / RATE
    for i, (scale, delay) in enumerate(((1., 0.), (ratio, lag))):
        z = t - peak_sec - delay
        wave = np.exp(-.5 * (z / width) ** 2)
        wave -= .12 * np.exp(-.5 * ((z - .22) / (.85 * width)) ** 2)
        out[i] += amplitude * scale * wave
    return out


def calibration():
    rng = np.random.default_rng(101)
    data = background(46, 102)
    peaks = np.arange(2., 44., 2.2)
    for peak in peaks:
        data = add_blink(data, peak, rng.uniform(110., 160.), rng.uniform(.093, .108),
                         rng.uniform(.75, .81), rng.uniform(-.008, .008))
    profile = calibrate_blinks(data, RATE, np.rint(peaks * RATE).astype(int),
                               channel_labels=LABELS, montage_id=MONTAGE)
    restored = BlinkCalibration.from_dict(json.loads(json.dumps(profile.as_dict(), allow_nan=False)))
    return restored


def run(data, profile):
    return process_blinks(data, RATE, profile, channel_labels=LABELS, montage_id=MONTAGE)


def condition(data):
    centered = data - data.mean(axis=1, keepdims=True)
    return np.stack([preprocess(row, RATE)[0] for row in centered])[:, -4 * RATE:]


def rms_error(candidate, truth):
    return float(np.sqrt(np.mean((candidate - truth) ** 2)))


def focus_index(data):
    return index_values(analyze_window(data, RATE))["focus"]


def stats(values):
    return {"n": len(values), "mean": float(np.mean(values)),
            "median": float(np.median(values)), "p05": float(np.percentile(values, 5)),
            "p95": float(np.percentile(values, 95)), "max": float(np.max(values))}


def core_benchmark(profile, trials, *, data_seed=50000, parameter_seed=733219):
    rng = np.random.default_rng(parameter_seed)
    clean_changed = clean_false_gate = corrected = masked = 0
    before_errors, after_errors, focus_before, focus_after = [], [], [], []
    band_errors = {band: [] for band in ("theta", "alpha", "beta")}
    event_statuses = Counter()
    input_preserved = outside_preserved = True
    rows = []
    for trial in range(trials):
        clean = background(6., data_seed + trial)
        untouched = clean.copy()
        clean_result = run(clean, profile)
        clean_changed += bool(np.any(clean_result.cleaned_uv != clean))
        clean_false_gate += bool(np.any(clean_result.mask))
        clean_before = analyze_window(condition(clean), RATE)
        clean_after = analyze_window(condition(clean_result.cleaned_uv), RATE)
        for band in band_errors:
            band_errors[band].append(abs(clean_after.powers[band] / clean_before.powers[band] - 1))

        parameters = {"peak_sec": float(rng.uniform(3.0, 4.8)),
                      "amplitude": float(rng.uniform(90, 185)),
                      "width": float(rng.uniform(.084, .117)),
                      "ratio": float(rng.uniform(.72, .84)),
                      "lag": float(rng.uniform(-.012, .012))}
        corrupted = add_blink(clean, **parameters)
        original = corrupted.copy()
        result = run(corrupted, profile)
        trusted = bool(np.any(result.corrected_mask)) and not np.any(result.mask)
        corrected += trusted
        masked += bool(np.any(result.mask))
        input_preserved &= np.array_equal(clean, untouched) and np.array_equal(corrupted, original)
        outside_preserved &= np.array_equal(result.cleaned_uv[:, ~result.corrected_mask],
                                            corrupted[:, ~result.corrected_mask])
        truth, before, after = condition(clean), condition(corrupted), condition(result.cleaned_uv)
        rmse_before, rmse_after = rms_error(before, truth), rms_error(after, truth)
        index_before = abs(focus_index(before) - focus_index(truth))
        index_after = abs(focus_index(after) - focus_index(truth))
        before_errors.append(rmse_before)
        after_errors.append(rmse_after)
        focus_before.append(index_before)
        focus_after.append(index_after)
        for event in result.events:
            event_statuses[str(event.get("status", event.get("state", "event")))] += 1
        rows.append({"trial": trial, **parameters, "corrected_and_unmasked": trusted,
                     "masked": bool(np.any(result.mask)), "rmse_before_uv": rmse_before,
                     "rmse_after_uv": rmse_after, "focus_log_ratio_error_before": index_before,
                     "focus_log_ratio_error_after": index_after, "evidence": result.as_dict()})
    return {"trial_count": trials, "clean_changed_trials": clean_changed,
            "clean_false_gated_trials": clean_false_gate,
            "clean_max_relative_band_power_change": {k: max(v) for k, v in band_errors.items()},
            "corrected_and_unmasked_trials": corrected, "masked_trials": masked,
            "input_preserved": bool(input_preserved),
            "outside_correction_support_preserved": bool(outside_preserved),
            "conditioned_rmse_before_uv": stats(before_errors),
            "conditioned_rmse_after_uv": stats(after_errors),
            "conditioned_mean_rmse_reduction_fraction": 1 - float(np.mean(after_errors) / np.mean(before_errors)),
            "focus_log_ratio_absolute_error_before": stats(focus_before),
            "focus_log_ratio_absolute_error_after": stats(focus_after),
            "focus_error_note": "Descriptive log(beta/theta) error against known synthetic clean data; unusable windows are never scored.",
            "event_status_counts": dict(event_statuses)}, rows


def adversarial_benchmark(profile):
    clean = background(6., 8000)
    cases = {}
    cases["uncalibrated_blink"] = (add_blink(clean, 3.5), None)
    step = clean.copy()
    step[0, round(3.5 * RATE):] += 180
    cases["unilateral_contact_step"] = (step, profile)
    plateau = clean.copy()
    plateau[:, round(3.3 * RATE):round(3.6 * RATE)] = OFFSET + 500
    cases["contact_plateau"] = (plateau, profile)
    saturated = clean.copy()
    saturated[:, round(3.4 * RATE):round(3.8 * RATE)] = OFFSET + 2000
    cases["saturation"] = (saturated, profile)
    invalid = clean.copy()
    invalid[0, round(3.5 * RATE)] = np.nan
    cases["non_finite"] = (invalid, profile)
    cases["left_boundary_partial_blink"] = (add_blink(clean, .04), profile)
    cases["right_boundary_partial_blink"] = (add_blink(clean, 5.96), profile)
    dense = clean.copy()
    for peak in np.arange(2.2, 5.9, .45):
        dense = add_blink(dense, float(peak), width=.11)
    cases["dense_overlapping_blinks"] = (dense, profile)
    wrong_ratio = add_blink(clean, 3.5, ratio=.18)
    cases["out_of_profile_channel_ratio"] = (wrong_ratio, profile)
    rows = {}
    for name, (data, candidate_profile) in cases.items():
        original = data.copy()
        result = run(data, candidate_profile)
        rows[name] = {"gated": bool(np.any(result.mask)),
                      "corrected": bool(np.any(result.corrected_mask)),
                      "input_preserved": bool(np.array_equal(original, data, equal_nan=True)),
                      "evidence": result.as_dict()}
    return rows


def live_pipeline_benchmark(profile, trials, *, data_seed=70000):
    """Exercise the actual source conditioning, quality veto, baseline and index code."""
    from ningsi.processing import process_window
    from ningsi_studio.acquisition.lsl import StreamBuffer, StreamDescriptor
    from ningsi_studio.core.blink_control import context
    from ningsi_studio.core.live_source import ManagedLslSource

    source = ManagedLslSource("synthetic-eeg", device="synthetic-eeg", fill_timeout=0)
    descriptor = StreamDescriptor("eeg", "synthetic-eeg", "EEG", 2, RATE, LABELS,
                                  declared_units=("uV", "uV"))
    source._manager._buffers["eeg"] = StreamBuffer(descriptor)
    # Match the source binding used by the product; the benchmark is explicitly synthetic.
    _, _, _, montage = context(source)
    bound_profile = replace(profile, montage_id=montage)
    binding = {"participant": "synthetic-only", "device": "lsl:synthetic-eeg", "montage_id": montage,
               "unit_confirmation": None, "profile": bound_profile.as_dict()}
    source.blink_binding = binding
    baseline_windows = []
    for number in range(10):
        wave, evidence = source._condition(background(6., 9000 + number), 4 * RATE,
                                            return_artifacts=True)
        baseline_windows.append(process_window(wave, RATE, t_end=4. * (number + 1),
                                               artifact_evidence=evidence).window)
    baseline = build_baseline(baseline_windows, device="synthetic-eeg", srate=RATE)
    errors = []
    coverage = Counter()
    flags = Counter()
    for trial in range(trials):
        clean = background(6., data_seed + trial)
        corrupted = add_blink(clean, 3.7, amplitude=100 + 3 * (trial % 20), width=.09 + .001 * (trial % 20))
        truth, truth_evidence = source._condition(clean, 4 * RATE, return_artifacts=True)
        truth_result = process_window(truth, RATE, baseline=baseline, artifact_evidence=truth_evidence)
        source.blink_binding = None
        old_wave, old_evidence = source._condition(corrupted, 4 * RATE, return_artifacts=True)
        old_result = process_window(old_wave, RATE, baseline=baseline, artifact_evidence=old_evidence)
        source.blink_binding = binding
        wave, evidence = source._condition(corrupted, 4 * RATE, return_artifacts=True)
        result = process_window(wave, RATE, baseline=baseline, artifact_evidence=evidence)
        coverage["clean_available"] += truth_result.indicators.available
        coverage["uncalibrated_available"] += old_result.indicators.available
        coverage["calibrated_available"] += result.indicators.available
        for reason in evidence.get("reasons", ()):
            flags[reason] += 1
        if truth_result.indicators.available and result.indicators.available:
            errors.append(abs(result.indicators.scores["focus"] - truth_result.indicators.scores["focus"]))
    return {"trial_count": trials, "baseline_valid": baseline.valid,
            "baseline_reasons": list(baseline.reasons), "coverage": dict(coverage),
            "focus_absolute_error": stats(errors) if errors else None,
            "artifact_gate_reasons": dict(flags)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "var" / "blink-control-20261117")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--cohort", choices=("holdout", "development"), default="holdout",
                        help="Development mixtures informed template smoothing. Holdout has separate phases/noise and blink parameters.")
    parser.add_argument("--skip-live", action="store_true", help="Core development only; final validation must include the live chain.")
    args = parser.parse_args()
    if args.trials < 10:
        parser.error("Use at least 10 independent held-out mixtures.")
    profile = calibration()
    data_seed, parameter_seed = (50000, 733219) if args.cohort == "holdout" else (2000, 20261117)
    summary, trials = core_benchmark(profile, args.trials, data_seed=data_seed, parameter_seed=parameter_seed)
    adversarial = adversarial_benchmark(profile)
    live = None if args.skip_live else live_pipeline_benchmark(profile, min(args.trials, 40),
                                                              data_seed=70000 if args.cohort == "holdout" else 10000)
    checks = {"clean_samples_unchanged": summary["clean_changed_trials"] == 0,
              "no_clean_false_gates": summary["clean_false_gated_trials"] == 0,
              "input_preserved": summary["input_preserved"],
              "local_correction_only": summary["outside_correction_support_preserved"],
              "all_adversarial_cases_gated": all(row["gated"] for row in adversarial.values()),
              "uncalibrated_never_corrected": not adversarial["uncalibrated_blink"]["corrected"],
              "mean_rmse_reduced": summary["conditioned_mean_rmse_reduction_fraction"] > .5,
              "focus_log_ratio_error_not_worsened": (
                  summary["focus_log_ratio_absolute_error_after"]["mean"] <=
                  summary["focus_log_ratio_absolute_error_before"]["mean"])}
    if live is not None:
        checks["live_baseline_valid"] = live["baseline_valid"]
        checks["live_calibration_has_available_windows"] = live["coverage"].get("calibrated_available", 0) > 0
    report = {"schema": "synthetic-blink-validation-v1", "sample_rate_hz": RATE,
              "cohort": args.cohort, "eeg_background_seed_start": data_seed, "blink_parameter_seed": parameter_seed,
              "calibration_seed": [101, 102], "signal_source": "synthetic known mixtures, no human recordings",
              "limitations": ["No real EEG/EOG/video labels or participant-held-out evaluation.",
                              "Synthetic detection/correction results do not establish human blink precision/recall, neural recovery or attention accuracy.",
                              "Existing Studio acq-condition-v1 performs zero-phase filtering inside the acquired window; this is bounded-window scoring, not zero-delay sample-causal correction."],
              "calibration": profile.as_dict(), "core": summary,
              "adversarial": adversarial, "live_pipeline": live,
              "checks": checks, "passed": all(checks.values()), "trials": trials}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# 双通道眨眼控制：合成信号工程验证", "",
             "所有输入均为已知脑电振荡与人工眨眼混合，未使用真人脑电。这不是设备效果或专注度有效性的证明。", "",
             f"- 评估组：{args.cohort}；脑电种子起点：{data_seed}；眨眼参数种子：{parameter_seed}；采样率：{RATE} Hz；独立混合：{args.trials}。",
             f"- 干净片段被改写：{summary['clean_changed_trials']}/{args.trials}；误门控：{summary['clean_false_gated_trials']}/{args.trials}。",
             f"- 校正且未被核心门控：{summary['corrected_and_unmasked_trials']}/{args.trials}。",
             f"- 经过实际 DC/高通/陷波/低通后的平均波形 RMSE：{summary['conditioned_rmse_before_uv']['mean']:.3f} → {summary['conditioned_rmse_after_uv']['mean']:.3f} µV。",
             f"- 专注 log(β/θ) 绝对误差均值：{summary['focus_log_ratio_absolute_error_before']['mean']:.6f} → {summary['focus_log_ratio_absolute_error_after']['mean']:.6f}；P95：{summary['focus_log_ratio_absolute_error_before']['p95']:.6f} → {summary['focus_log_ratio_absolute_error_after']['p95']:.6f}。",
             f"- 全部工程检查通过：{all(checks.values())}。", "",
             "详细条件、逐次结果、门控原因和真实产品链路检查见 metrics.json。",
             "真实硬件仍需独立记录眨眼事件并比较检测精确率/召回率、有效覆盖、频带偏移及受试者外专注度结果。", ""]
    if live is not None:
        lines[8:8] = [f"- 真实 Studio 代码链合成可评分窗：未校准 {live['coverage'].get('uncalibrated_available', 0)}，校准后 {live['coverage'].get('calibrated_available', 0)}，共 {live['trial_count']}；独立干净基线有效：{live['baseline_valid']}。"]
    (args.out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "checks": checks, "core": summary,
                      "live_pipeline": live, "output": str(args.out)}, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
