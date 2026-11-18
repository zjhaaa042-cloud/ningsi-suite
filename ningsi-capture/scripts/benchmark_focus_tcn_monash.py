"""Fixed LR versus ordered/shuffled feature TCN replication on local Monash EEG.

This second public cohort is specified before inspecting primary ATTLAPSE TCN
results. It reuses only original Fp1/Fp2 epochs and their existing physical QC;
models are refitted by participant, never transferred from the first cohort.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import benchmark_focus_tcn as shared
import attlapse_two_channel_eval as evaluation
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, resample_poly, sosfiltfilt

from ningsi.models.logistic import FEATURE_NAMES, features_from_window
from ningsi.signal.bands import relative_band_powers
from ningsi.signal.spectrum import welch_psd
from validate_wanderim_frontal import clean_windows


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "var/new-cohorts-20261115/focus"
OUTPUT = ROOT / "var/tcn-focus-20261118/monash"
SLOT_ENDS = (1250, 1500, 1750)
SLOT_INTERVALS = ((-7, -3), (-6, -2), (-5, -1))


def prescribe(out, people, source):
    """Write fold assignments and settings before any model fit or score access."""
    partitions = evaluation.folds(np.asarray(people), 5, evaluation.SEED)
    outer = [{"outer_fold": index + 1,
              "train": sorted(set(people) - set(test)), "test": test}
             for index, test in enumerate(partitions)]
    protocol = {"spec": "monash-feature-tcn-replication-v1",
                "prescription": "Independent second-cohort experiment fixed before inspecting primary ATTLAPSE TCN scores.",
                "participants": sorted(people), "fold_seed": evaluation.SEED,
                "source_features_sha256": shared.sha(source / "features.json"),
                "synchronization_sha256": shared.sha(source / "synchronization.json"),
                "features": list(FEATURE_NAMES), "rate_hz": 250,
                "preprobe_observed_interval_seconds": [-8, -1],
                "slots_preprobe_seconds": [list(value) for value in SLOT_INTERVALS],
                "eligibility": "At least one existing physical-QC-accepted four-second slot; preserve all three positions and their missing mask.",
                "labels": {"0": "Q2=1, on-task", "1": "Q2=2, off-task/mind-wandering"},
                "excluded_states": "Q2 other than 1/2; do not treat blanking/unknown as mind-wandering.",
                "arms": ["fixed L2 logistic, average of the same six features over accepted slots",
                         "ordered small feature TCN", "shuffled-valid-slots same TCN", "probe ordinal / 60"],
                "tcn_settings": shared.TCN_SETTINGS, "tcn_seeds": list(shared.SEEDS),
                "logistic_l2": shared.LR.parameter,
                "outer": outer,
                "scope": "Participant-held-out exploratory replication; different pre-probe interval and self-report wording; separately refitted cohort, not direct model transfer.",
                "selection": "No settings, quality thresholds, seeds or folds selected using current model scores."}
    path = out / "splits.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != protocol:
            raise ValueError("Existing prescribed protocol differs; use a fresh output")
    else:
        shared.write_json(path, protocol)
    return path, protocol


def slot_features(raw):
    """Exactly reproduce prior preprocessing/QC while retaining failed-slot positions."""
    if raw.shape != (2, 3500) or not np.isfinite(raw).all():
        raise ValueError("Expected finite original 500 Hz Fp1/Fp2 epoch, seven seconds")
    centered = raw.astype(np.float64)
    centered -= np.median(centered, axis=1, keepdims=True)
    wave = resample_poly(centered, 1, 2, axis=1, padtype="line")
    wave = sosfiltfilt(butter(4, (.5, 40), btype="bandpass", fs=250, output="sos"), wave)
    vectors = np.zeros((3, len(FEATURE_NAMES)), dtype=float)
    mask = np.zeros(3, dtype=bool)
    accepted = []
    for index, end in enumerate(SLOT_ENDS):
        pair = wave[:, end - 1000:end]
        span = np.ptp(pair, axis=1)
        if np.any((span < 1) | (span > 180)) or np.max(np.abs(np.diff(pair, axis=1))) > 65:
            continue
        accepted.append(pair)
        mask[index] = True
        spectrum = welch_psd(pair[0] - pair[1], 250)
        relative = relative_band_powers(spectrum.freqs, spectrum.psd)
        features = features_from_window(SimpleNamespace(rel=relative))
        vectors[index] = [features[name] for name in FEATURE_NAMES]
    reference = list(clean_windows(raw, 500))
    if len(reference) != len(accepted):
        raise AssertionError("QC no longer matches the existing Monash helper")
    for original, repeated in zip(reference, accepted, strict=True):
        np.testing.assert_array_equal(original, repeated)
    return vectors, mask


def prepare(source, out):
    plans = json.loads((source / "synchronization.json").read_text(encoding="utf-8"))["included"]
    archived = json.loads((source / "features.json").read_text(encoding="utf-8"))
    indexed = {(row["person"], int(row["probe"])): row for row in archived}
    if len(archived) != 1560 or len(indexed) != 1560 or len(plans) != 26:
        raise ValueError("Expected 26 original participants and all 1,560 probes")
    # Prescribe groups before reading/replaying reference prediction values.
    splits_path, protocol = prescribe(out, [str(plan["person"]) for plan in plans], source)
    labelled = [row for row in archived if row["label"] is not None]
    labelled.sort(key=lambda row: (row["person"], int(row["probe"])))
    keys = [(row["person"], int(row["probe"])) for row in labelled]
    if len(keys) != 1368 or set(row["label"] for row in labelled) != {0, 1}:
        raise ValueError("The predeclared binary target roster changed")
    where = {key: index for index, key in enumerate(keys)}
    x = np.zeros((len(keys), 3, len(FEATURE_NAMES)), dtype=float)
    mask = np.zeros(x.shape[:2], dtype=bool)
    checked, raw_hashes, answer_counts = set(), {}, Counter()
    for plan in plans:
        person = str(plan["person"])
        if plan["rate"] != 500 or len(plan["at"]) != 60:
            raise ValueError("Unexpected original sample rate or probe roster")
        behavior = loadmat(source / "source" / plan["behavior"], variable_names=["probe_res"])["probe_res"]
        if behavior.shape[0] != 60:
            raise ValueError("Expected exactly 60 behavior probe rows")
        for ordinal in range(1, 61):
            key = person, ordinal
            archived_row = indexed[key]
            q2 = behavior[ordinal - 1, 31]
            label = int(q2 == 2) if q2 in (1, 2) else None
            if archived_row["label"] != label:
                raise AssertionError(f"Original self-report alignment changed: {key}")
            answer_counts[str(int(q2)) if np.isfinite(q2) else "nonfinite"] += 1
            path = source / "source" / ("p" + person) / f"{ordinal - 1:02}.npy"
            fingerprint = shared.sha(path)
            if fingerprint != archived_row["sample_sha256"]:
                raise ValueError(f"Raw epoch checksum changed: {key}")
            raw_hashes[f"{person}/{ordinal:02}"] = fingerprint
            if label is None:
                continue
            i = where[key]
            values, accepted = slot_features(np.load(path, allow_pickle=False))
            if int(accepted.sum()) != archived_row["valid_chunks"]:
                raise AssertionError(f"Existing QC eligibility differs: {key}")
            x[i], mask[i] = values, accepted
            checked.add(key)
    if checked != set(keys):
        raise AssertionError("A binary-labelled probe was omitted")
    y = np.array([row["label"] for row in labelled], dtype=int)
    groups = np.array([key[0] for key in keys])
    probes = np.array([f"probe-{key[1]:02}" for key in keys])
    if (not np.array_equal(np.unique([int(p.split("-")[-1]) for p in probes]), np.arange(1, 61))
            or np.any(~np.isfinite(x)) or not np.all(x[~mask] == 0)):
        raise AssertionError("Probe ordinal, feature finiteness or failed-slot masking changed")
    means = x.sum(axis=1) / np.maximum(mask.sum(axis=1, keepdims=True), 1)
    parity = reference_prediction_parity(source, keys, y, groups, means, mask.any(axis=1))
    np.savez_compressed(out / "sequences.npz", features=x, mask=mask, label=y,
                        participants=groups, probes=probes)
    provenance = {"raw_epoch_sha256": raw_hashes, "raw_epochs_verified": len(raw_hashes),
                  "source_features_sha256": shared.sha(source / "features.json"),
                  "source_synchronization_sha256": shared.sha(source / "synchronization.json"),
                  "source_answers": dict(answer_counts), "labelled_probes": len(keys),
                  "label_counts": {str(k): int((y == k).sum()) for k in (0, 1)},
                  "eligible_probes": int(mask.any(axis=1).sum()), "source_qc_reproduced": True,
                  "reference_project_six_parity": parity}
    shared.write_json(out / "input_provenance.json", provenance)
    return x, mask, y, groups, probes, splits_path, protocol, provenance


def reference_prediction_parity(source, keys, y, groups, means, eligible):
    """Replay pre-existing six-feature LR probabilities to verify aggregate features."""
    with (source / "project_six_predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
        previous = {(row["person"], int(row["probe"])): row for row in csv.DictReader(stream)}
    expected = {key for key, ok in zip(keys, eligible, strict=True) if ok}
    if set(previous) != expected:
        raise AssertionError("Six-feature reference eligibility differs from the recomputed slots")
    folds = json.loads((source / "results.json").read_text(encoding="utf-8"))["folds"]
    c = evaluation.Candidate("physical", "preprobe", "frontal", "logistic", .1)
    seen, errors = set(), []
    for fold in folds:
        train = eligible & np.isin(groups, fold["train"])
        test = eligible & np.isin(groups, fold["test"])
        original = evaluation.fit(means[train], y[train], groups[train], c)
        scores = evaluation.predict(original, means[test])
        for i, score in zip(np.flatnonzero(test), scores, strict=True):
            key = keys[i]
            if key in seen or int(previous[key]["label"]) != y[i]:
                raise AssertionError("Duplicated or mismatched reference prediction")
            errors.append(abs(float(score) - float(previous[key]["project_six_only"])))
            seen.add(key)
    if seen != expected or max(errors) > 1e-9:
        raise AssertionError("Original project six-feature predictions cannot be reproduced")
    return {"reference_file": "project_six_predictions.csv", "reference_sha256": shared.sha(source / "project_six_predictions.csv"),
            "probes": len(errors), "max_probability_error": max(errors),
            "reference_l2": .1, "new_comparison_l2": shared.LR.parameter,
            "interpretation": "Reference coefficients are refitted only to check same-window six-feature aggregation; reference scores do not select the new model settings or splits."}


def report(out, evidence):
    scores = evidence["evaluation"]["scores"]
    label = {"logistic": "L2 逻辑回归（六特征按窗均值）", "ordered_tcn": "小型 TCN（原时间顺序）",
             "shuffled_tcn": "同结构 TCN（打乱有效窗）", "probe_order": "探针序号 / 60"}
    coverage = scores["logistic"]
    lines = ["# Monash 双额部特征 TCN：独立第二队列对照", "",
             "在看到主队列 ATTLAPSE 的本次模型结果之前确定第二队列方案；固定模型、参数、随机种子和按人五折分组。",
             "26 人各 60 次探针，原始 Fp1/Fp2 为 500 Hz。Q2=1 为在任务上，Q2=2 为走神／off-task；其他状态不进入二分类。",
             f"原始 1,560 次中二分类共 1,368 次；相同物理质控后覆盖 {coverage['scored_labelled']}/1,368，{coverage['both_classes_scored_participants']} 人具备两类有效样本可计算个体内 AUROC。",
             "每次探针只观察 [-8,-1) 秒的已获得 EEG，中心化并降至 250 Hz、0.5–40 Hz 滤波，再保留三个 4 秒候选窗的位置与缺失掩码。每窗为项目原六特征，无行为、眼电或其他导联输入。",
             "全部模型采用相同探针与有效窗。逻辑回归取有效窗均值；TCN 读取特征序列。固定三个初始化分别训练，表中先按探针平均三个预测。", "",
             "| 方法 | 个体内平均 AUROC | AP | 平衡准确率（0.5 阈值） |", "|---|---:|---:|---:|"]
    for name in ("logistic", "ordered_tcn", "shuffled_tcn", "probe_order"):
        row = scores[name]
        lines.append(f"| {label[name]} | {row['macro_auc']:.4f} | {row['average_precision']:.4f} | {row['balanced_accuracy']:.4f} |")
    lines.extend(["", "AUROC 配对差与按参与者重采样的 95% 区间：", ""])
    for key, name in (("ordered_tcn_minus_logistic", "TCN − 逻辑回归"),
                      ("ordered_tcn_minus_shuffled_tcn", "TCN − 打乱时间顺序"),
                      ("ordered_tcn_minus_probe_order", "TCN − 探针序号")):
        row = evidence["evaluation"]["paired_macro_auc"][key]
        lo, hi = row["paired_ci95"]
        lines.append(f"- {name}：{row['difference']:+.4f} [{lo:+.4f}, {hi:+.4f}]。")
    parity = evidence["input_provenance"]["reference_project_six_parity"]
    lines.extend(["", f"既有项目六特征对照的 {parity['probes']} 个概率重算最大误差 {parity['max_probability_error']:.3g}；原始逐片段哈希及质控有效窗数均已核对。",
                  "这是在该队列内重新训练的探索性复现实验，不是 ATTLAPSE 模型直接迁移。该数据此前已分析，五折结果不是新采集的盲测。",
                  "Monash 的时间窗、自评措辞及物理质量规则不同；不能用两个队列间的 AUROC 差推断模型迁移效果。",
                  "三个有效短窗最多只有三个时间位置，且相互重叠；无额外严格频谱代理门控，残留眼动和肌电无法排除。",
                  "滤波仅使用探针前的完整已观察片段，但其中为正反向滤波，不是逐样点零延迟因果算法。",
                  "单个探针是一个评价样本。模型标准化和参数学习只使用训练者；未使用被试内随机切窗分组。",
                  "不改变生产模型、个人基线或设备质量门槛，不代表目标头环上的专注度识别有效性。", "",
                  "完整输入、固定分组、逐探针预测及统计见本目录的 sequences.npz、splits.json、predictions.csv 和 metrics.json。", ""])
    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    source, out = args.data.resolve(), args.output.resolve()
    out.relative_to(ROOT)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.json").exists():
        raise FileExistsError("Choose a fresh output; results are not overwritten")
    x, mask, y, groups, probes, splits, protocol, provenance = prepare(source, out)
    if args.prepare_only:
        print(json.dumps({"prepared": str(out / "sequences.npz"), "eligible": int(mask.any(axis=1).sum()),
                          "provenance": provenance["reference_project_six_parity"]}, ensure_ascii=False), flush=True)
        return
    predictions, seeds, run = shared.fit_and_predict(source, x, mask, y, groups, probes, mask.any(axis=1),
                                                    folds_path=splits, order_denominator=60)
    evidence = {"spec": "monash-matched-feature-tcn-v1", "source": "Andrillon et al., OSF ey3ca/w7gz5/6g2fh",
                "data_path": str(source), "input_shape": list(x.shape), "input_sha256": shared.sha(out / "sequences.npz"),
                "prescribed_protocol": protocol, "input_provenance": provenance,
                "tcn_settings": shared.TCN_SETTINGS, "tcn_seeds": list(shared.SEEDS),
                "logistic_l2": shared.LR.parameter, "run": run,
                "evaluation": shared.evaluate(y, groups, predictions, seeds, None)}
    with (out / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["participant", "probe", "label", "eligible", "valid_windows", *predictions])
        for i, (person, probe) in enumerate(zip(groups, probes, strict=True)):
            writer.writerow([person, probe, y[i], mask[i].any(), int(mask[i].sum()),
                             *(float(prob[i]) if np.isfinite(prob[i]) else "" for prob in predictions.values())])
    shared.write_json(out / "metrics.json", evidence)
    report(out, evidence)
    print(out / "RESULTS.md", flush=True)


if __name__ == "__main__":
    main()
