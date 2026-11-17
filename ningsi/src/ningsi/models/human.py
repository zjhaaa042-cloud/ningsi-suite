"""Reproducible participant-held-out training and evaluation of real probe data.

The primary model is deliberately small: six spectral features and weighted,
L2-regularised logistic regression. No simulated training fallback exists.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import csv
import html
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from ningsi.acquisition.timing import BIOMULTI_LSL_TIMING, BIOMULTI_REC_TIMING, PACKET_TIMING, STRICT_TIMING
from ningsi.signal import conditioning

from .human_data import LABELS, canonical, digest, summarize
from .logistic import FEATURE_NAMES, roc_auc
from .research_features import CONTRACT

MODEL_SPEC = "ningsi-human-attention-v1"
TRAINING_CONFIG = {"seed": 20261113, "folds": 5, "l2_candidates": [0.001, 0.01, 0.1],
                   "epochs": 900, "learning_rate": 0.15, "bootstrap_iterations": 2000,
                   "primary_min_per_class": 5, "evidence_min_test_participants": 20,
                   "target_macro_auc": 0.70, "target_ci_lower": 0.60, "target_coverage": 0.70,
                   "positive_label": 1, "positive_meaning": "internal_mind_wandering",
                   "model_selection": "development_group_oof_macro_auc",
                   "threshold_selection": "development_oof_participant_weighted_balanced_accuracy"}


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def sigmoid(x):
    return 1. / (1. + np.exp(-np.clip(x, -40, 40)))


def matrix(rows):
    return np.asarray([[float(r[k]) for k in FEATURE_NAMES] for r in rows], dtype=float)


def person_weights(rows, balance_classes=False):
    counts = Counter(r["participant"] for r in rows)
    w = np.asarray([1. / counts[r["participant"]] for r in rows])
    if balance_classes:
        y = np.asarray([int(r["label"]) for r in rows])
        for label in (0, 1):
            mask = y == label
            total = w[mask].sum()
            if total <= 0:
                raise ValueError("训练划分必须同时包含专注和走神标签。")
            w[mask] /= total
    return w / w.sum()


def fit(x, y, weights, l2, check=lambda: None):
    mean = np.sum(x * weights[:, None], axis=0)
    scale = np.sqrt(np.sum((x - mean) ** 2 * weights[:, None], axis=0))
    scale = np.where(scale < 1e-9, 1., scale)
    z = (x - mean) / scale
    coef, bias = np.zeros(x.shape[1]), 0.
    for epoch in range(TRAINING_CONFIG["epochs"]):
        if epoch % 30 == 0:
            check()
        error = (sigmoid(z @ coef + bias) - y) * weights
        coef -= TRAINING_CONFIG["learning_rate"] * (z.T @ error + l2 * coef)
        bias -= TRAINING_CONFIG["learning_rate"] * error.sum()
    return {"mean": mean.tolist(), "scale": scale.tolist(), "weights": coef.tolist(), "bias": float(bias)}


def logits(parameters, x):
    return ((x - np.asarray(parameters["mean"])) / np.asarray(parameters["scale"])) @ np.asarray(parameters["weights"]) + parameters["bias"]


def group_folds(rows, seed=TRAINING_CONFIG["seed"]):
    people = sorted({r["participant"] for r in rows})
    if len(people) < TRAINING_CONFIG["folds"]:
        raise ValueError("真人训练至少需要 5 位开发组参与者；请继续采集，不能用仿真数据补足。")
    rng = np.random.default_rng(seed)
    rng.shuffle(people)
    assignments = [sorted(people[i::TRAINING_CONFIG["folds"]]) for i in range(TRAINING_CONFIG["folds"])]
    folds = []
    groups = np.asarray([r["participant"] for r in rows])
    for index, heldout in enumerate(assignments):
        val = np.flatnonzero(np.isin(groups, heldout))
        train = np.flatnonzero(~np.isin(groups, heldout))
        if len({rows[i]["label"] for i in train}) != 2:
            raise ValueError("某交叉验证训练折缺少一种标签，请增加不同参与者的有效标签。")
        folds.append((train, val, {"fold": index + 1, "train_participants": sorted(set(groups[train])),
                                  "validation_participants": heldout}))
    return folds


def average_precision(y, scores):
    y, scores = np.asarray(y), np.asarray(scores)
    positives = int(y.sum())
    if not positives:
        return None
    order = np.argsort(-scores, kind="stable")
    y, scores = y[order], scores[order]
    tp, ap, index = 0, 0., 0
    while index < len(y):
        end = index + 1
        while end < len(y) and scores[end] == scores[index]:
            end += 1
        gained = int(y[index:end].sum())
        tp += gained
        ap += gained / positives * tp / end
        index = end
    return float(ap)


def scalar_metrics(y, p, threshold=.5, predictions=None, weights=None):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    pred = p >= threshold if predictions is None else np.asarray(predictions, dtype=bool)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    tp, tn, fp, fn = [float(w[mask].sum()) for mask in ((y == 1) & pred, (y == 0) & ~pred,
                                                       (y == 0) & pred, (y == 1) & ~pred)]
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    return {"n": len(y), "accuracy": (tp + tn) / w.sum() if len(y) else None,
            "sensitivity": sensitivity, "specificity": specificity,
            "balanced_accuracy": (sensitivity + specificity) / 2 if sensitivity is not None and specificity is not None else None,
            "auc": roc_auc(y, p) if len(y) else None, "average_precision": average_precision(y, p),
            "brier": float(np.average((p - y) ** 2, weights=w)) if len(y) else None,
            "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn}}


def evaluate(rows, p, threshold, *, predictions=None):
    y = np.asarray([int(r["label"]) for r in rows])
    p = np.asarray(p)
    overall = scalar_metrics(y, p, threshold, predictions)
    overall["participant_weighted"] = scalar_metrics(y, p, threshold, predictions, person_weights(rows))
    individuals = []
    for person in sorted({r["participant"] for r in rows}):
        indices = np.asarray([i for i, r in enumerate(rows) if r["participant"] == person])
        truth = y[indices]
        metric = scalar_metrics(truth, p[indices], threshold, None if predictions is None else np.asarray(predictions)[indices])
        metric.update(participant=person, on_task=int((truth == 0).sum()), mind_wandering=int(truth.sum()))
        metric["primary_eligible"] = min(metric["on_task"], metric["mind_wandering"]) >= TRAINING_CONFIG["primary_min_per_class"]
        individuals.append(metric)
    all_auc = [m["auc"] for m in individuals if m["auc"] is not None]
    primary = [m["auc"] for m in individuals if m["primary_eligible"]]
    overall.update(per_participant=individuals,
                   macro_auc=float(np.mean(all_auc)) if all_auc else None,
                   primary_macro_auc=float(np.mean(primary)) if primary else None,
                   primary_participants=len(primary), participants=len(individuals))
    # Calibration is descriptive on held-out predictions; no parameters fit here.
    bins = []
    for index in range(10):
        mask = (p >= index / 10) & ((p < (index + 1) / 10) if index < 9 else (p <= 1))
        bins.append({"low": index / 10, "high": (index + 1) / 10, "n": int(mask.sum()),
                     "mean_prediction": float(p[mask].mean()) if mask.any() else None,
                     "observed_fraction": float(y[mask].mean()) if mask.any() else None})
    overall["calibration_bins"] = bins
    return overall


def bootstrap_mean(values, seed, check=lambda: None):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    means = []
    for n in range(TRAINING_CONFIG["bootstrap_iterations"]):
        if n % 50 == 0:
            check()
        means.append(float(rng.choice(values, len(values), replace=True).mean()))
    return [float(v) for v in np.percentile(means, [2.5, 97.5])]


def train_development(data, progress=lambda stage, fraction: None):
    rows = data["development"]
    if not rows or len({r["label"] for r in rows}) != 2:
        raise ValueError("开发组缺少可用的两类真人标签。")
    folds, x = group_folds(rows), matrix(rows)
    y = np.asarray([int(r["label"]) for r in rows])
    candidates = []
    for c, l2 in enumerate(TRAINING_CONFIG["l2_candidates"]):
        oof = np.full(len(rows), np.nan)
        for f, (train, val, split) in enumerate(folds):
            fraction = .05 + .50 * (c * len(folds) + f) / (len(folds) * 3)
            check = lambda: progress("开发组按被试交叉验证", fraction)
            parameters = fit(x[train], y[train], person_weights([rows[i] for i in train], True), l2, check)
            oof[val] = logits(parameters, x[val])
        metric = evaluate(rows, sigmoid(oof), .5)
        score = metric["macro_auc"] if metric["macro_auc"] is not None else metric["auc"]
        candidates.append({"l2": l2, "selection_auc": score, "oof_logits": oof, "metrics": metric})
    best = max(candidates, key=lambda c: (c["selection_auc"], c["l2"]))
    progress("使用开发组折外预测校准与确定阈值", .58)
    cal = fit(best["oof_logits"][:, None], y, person_weights(rows), .01,
              lambda: progress("开发组概率校准", .60))
    slope = cal["weights"][0] / cal["scale"][0]
    offset = cal["bias"] - cal["mean"][0] * slope
    calibrated_oof = sigmoid(best["oof_logits"] * slope + offset)
    thresholds = sorted(set([.5, *np.clip(np.quantile(calibrated_oof, np.linspace(0, 1, 101)), .001, .999).tolist()]))
    weights = person_weights(rows)
    threshold = max(thresholds, key=lambda t: (scalar_metrics(y, calibrated_oof, t, weights=weights)["balanced_accuracy"], -abs(t - .5)))
    progress("在完整开发组拟合最终模型", .65)
    parameters = fit(x, y, person_weights(rows, True), best["l2"], lambda: progress("拟合最终模型", .65))
    sessions = data.get("session_metadata") or {}
    training_protocols = {sessions.get(row["session_id"], {}).get("protocol") for row in rows}
    bundle = {"schema": MODEL_SPEC, "created_utc": datetime.now(timezone.utc).isoformat(),
              "feature_contract": CONTRACT, "feature_names": list(FEATURE_NAMES), "labels": LABELS,
              "positive_label": 1, "threshold": float(threshold), "parameters": parameters,
              "calibration": {"method": "sigmoid_on_development_group_oof_logits", "slope": float(slope), "offset": float(offset)},
              "dataset_id": data["dataset_id"], "trained_participants": sorted({r["participant"] for r in rows}),
              "training_protocols": sorted(filter(None, training_protocols)),
              "engine_fingerprint": data["engine_fingerprint"], "source_profiles": data["source_profiles"],
              "implementation_fingerprint": data.get("implementation_fingerprint"),
              "training_config": {**TRAINING_CONFIG, "selected_l2": best["l2"]},
              "scope": "SART probe-related on-task vs internal mind-wandering; research use; not trait ability or diagnosis"}
    bundle["model_id"] = digest(bundle)
    cv = {"folds": [f[2] for f in folds], "candidates": [{k: v for k, v in c.items() if k != "oof_logits"} for c in candidates],
          "selected_l2": best["l2"], "threshold": float(threshold),
          "development_oof": evaluate(rows, calibrated_oof, threshold),
          "note": "开发组折外结果参与了模型、校准及阈值选择，不是独立测试成绩。"}
    return bundle, cv


def validate_model(bundle):
    if not isinstance(bundle, dict) or bundle.get("schema") != MODEL_SPEC:
        raise ValueError("仅支持真人训练器导出的 attention_model.json。")
    if bundle.get("feature_names") != list(FEATURE_NAMES) or bundle.get("feature_contract") != CONTRACT or bundle.get("labels") != LABELS or bundle.get("positive_label") != 1:
        raise ValueError("模型特征或标签方向不匹配。")
    if bundle.get("model_id") != digest({k: v for k, v in bundle.items() if k != "model_id"}):
        raise ValueError("模型完整性校验失败。")
    params = bundle.get("parameters", {})
    for name in ("weights", "mean", "scale"):
        values = np.asarray(params.get(name, []), dtype=float)
        if (values.shape != (6,) or not np.isfinite(values).all() or (np.abs(values) > 1e9).any()
                or name == "scale" and (values < 1e-9).any()):
            raise ValueError("模型参数无效。")
    scalars = [params.get("bias"), bundle.get("threshold"), bundle.get("calibration", {}).get("slope"), bundle.get("calibration", {}).get("offset")]
    if not all(v is not None and np.isfinite(float(v)) for v in scalars) or not 0 < bundle["threshold"] < 1:
        raise ValueError("模型阈值或校准参数无效。")
    if not isinstance(bundle.get("source_profiles"), list) or len(bundle["source_profiles"]) > 100:
        raise ValueError("模型设备清单无效。")
    expected = conditioning.POLICY if conditioning.FILE in (bundle.get("implementation_fingerprint") or {}) else None
    for profile in bundle["source_profiles"]:
        if (not isinstance(profile, dict) or not isinstance(profile.get("name"), str)
                or not isinstance(profile.get("channel_labels"), list)
                or len(profile["channel_labels"]) != profile.get("channels")
                or not 1 <= int(profile["channels"]) <= 256
                or not 100 <= float(profile["srate"]) <= 10000
                or not np.isfinite(float(profile["scale_to_uv"])) or float(profile["scale_to_uv"]) <= 0):
            raise ValueError("模型设备通道或单位定义无效。")
        if profile.get("timestamp_policy") not in (None, STRICT_TIMING, PACKET_TIMING, BIOMULTI_LSL_TIMING, BIOMULTI_REC_TIMING):
            raise ValueError("模型时间处理方式无效。")
        if profile.get("preprocessing_policy") != expected:
            raise ValueError("模型设备与特征预处理版本不一致。")
    return bundle


def predict(bundle, x):
    validate_model(bundle)
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != 6 or not np.isfinite(x).all():
        raise ValueError("推理输入必须是有限的六维特征。")
    raw = logits(bundle["parameters"], x)
    return sigmoid(raw * bundle["calibration"]["slope"] + bundle["calibration"]["offset"])


def explain(bundle, values):
    p = float(predict(bundle, np.asarray(values)[None, :])[0])
    params = bundle["parameters"]
    z = (np.asarray(values) - params["mean"]) / params["scale"]
    contributions = z * params["weights"] * bundle["calibration"]["slope"]
    return {"available": True, "model_id": bundle["model_id"], "mind_wandering_probability": p,
            "on_task_probability": 1 - p, "threshold": bundle["threshold"],
            "prediction": "internal_mind_wandering" if p >= bundle["threshold"] else "on_task",
            "contributions": [{"feature": name, "value": float(value), "log_odds_contribution": float(c)}
                              for name, value, c in zip(FEATURE_NAMES, values, contributions)],
            "intercept": params["bias"] * bundle["calibration"]["slope"] + bundle["calibration"]["offset"],
            "scope": bundle["scope"]}


def independent_test(data, bundle, progress=lambda stage, fraction: None):
    rows = data["test"]
    model_impl = bundle.get("implementation_fingerprint") or {}
    test_impl = data.get("implementation_fingerprint") or {}
    if (conditioning.FILE in model_impl or conditioning.FILE in test_impl) and model_impl != test_impl:
        raise ValueError("独立测试数据与模型使用不同的脑电预处理，请统一复算。")
    if not rows or len({r["label"] for r in rows}) != 2:
        raise ValueError("独立测试组需要可用的两类真人标签；请先补足封存测试数据。")
    if {r["participant"] for r in rows} & set(bundle["trained_participants"]):
        raise ValueError("独立测试参与者与模型训练参与者重叠。")
    progress("冻结模型后进行独立测试", .76)
    probabilities = predict(bundle, matrix(rows))
    test = evaluate(rows, probabilities, bundle["threshold"])
    original = np.asarray([1 - float(r["focus"]) for r in rows])
    reference = evaluate(rows, original, .6, predictions=[float(r["focus"]) < .4 for r in rows])
    reference["decision_rule"] = "focus < 0.4; AUROC uses 1-focus"
    test["threshold"] = bundle["threshold"]
    eligible = [m for m in test["per_participant"] if m["primary_eligible"]]
    ref_by_person = {m["participant"]: m for m in reference["per_participant"]}
    check = lambda: progress("按被试重采样计算置信区间", .85)
    ci = bootstrap_mean([m["auc"] for m in eligible], TRAINING_CONFIG["seed"], check)
    delta = [m["auc"] - ref_by_person[m["participant"]]["auc"] for m in eligible]
    paired_ci = bootstrap_mean(delta, TRAINING_CONFIG["seed"] + 1, check)
    reference["primary_auc_ci95"] = bootstrap_mean([ref_by_person[m["participant"]]["auc"] for m in eligible], TRAINING_CONFIG["seed"] + 2, check)
    test["primary_auc_ci95"] = ci
    all_test_probes = rows + [r for r in data["excluded"] if r["cohort"] == "test" and r["data_origin"] == "real" and r.get("practice") == "False"]
    labelled = [r for r in all_test_probes if r.get("binary_label_valid") == "True"]
    coverage = len(rows) / len(labelled) if labelled else 0.
    total_coverage = len(rows) / len(all_test_probes) if all_test_probes else 0.
    evidence = (len(eligible) >= TRAINING_CONFIG["evidence_min_test_participants"] and ci is not None)
    target_pass = bool(evidence and test["primary_macro_auc"] >= .70 and ci[0] > .60 and coverage >= .70)
    report = {"schema": "ningsi-human-evaluation-v1", "dataset_id": data["dataset_id"], "model_id": bundle["model_id"],
              "created_utc": datetime.now(timezone.utc).isoformat(), "data": summarize(data),
              "test": test, "original_index": reference,
              "paired_auc_difference": {"mean": float(np.mean(delta)) if delta else None, "ci95": paired_ci, "participants": len(delta)},
              "coverage": {"usable_binary_probes": len(rows), "all_formal_test_probes": len(all_test_probes),
                           "binary_labelled_probes": len(labelled), "eeg_coverage_given_binary_label": coverage,
                           "end_to_end_probe_coverage": total_coverage},
              "evidence_status": "independent_test_sufficient_size" if evidence else "exploratory_insufficient_participants",
              "internal_target_passed": target_pass,
              "limitations": ["标签为任务中的主观探针回答，不能解释为绝对专注能力或诊断。",
                              "当前仅验证 SART 中任务专注与内部走神，不涵盖所有分心类型。",
                              "概率校准仅拟合开发组；独立测试校准曲线为描述性检查。",
                              "相互关联的频带比值使单个系数不能解释为因果机制。",
                              "工程目标（AUC≥0.70、下界>0.60、覆盖≥70%、至少20位有效测试者）不是赛题或临床标准。",
                              "此实验不证明神经反馈训练有效，训练效果需要独立对照实验。",
                              *data["notices"]]}
    predictions = [{"participant": r["participant"], "session_id": r["session_id"], "visit": r["visit"],
                    "probe_id": r["probe_id"], "label": int(r["label"]), "mind_wandering_probability": float(p),
                    "prediction": int(p >= bundle["threshold"]), "original_focus": float(r["focus"])}
                   for r, p in zip(rows, probabilities)]
    return report, predictions


def _number(value):
    return "—" if value is None else f"{value:.3f}"


def _protocol_description(report):
    return "、".join(f"{p['label']} {p['sessions']} 次" for p in report["data"].get("research_protocols", []))


def report_markdown(report, cv):
    t, original, coverage = report["test"], report["original_index"], report["coverage"]
    lines = ["# 真人脑电注意状态模型独立测试报告", "", f"模型：`{report['model_id']}`", f"数据集：`{report['dataset_id']}`", "",
             "标签：0=当前任务专注；1=内部走神。开发组用于模型、校准和阈值选择，测试组在模型冻结后评估。", "",
             f"证据状态：{'正式规模的独立测试' if report['evidence_status'] == 'independent_test_sufficient_size' else '探索性结果：满足每类至少5个标签的测试参与者不足20人'}。", "",
             "| 指标 | 真人模型 | 原专注指数（走神分数=1-focus） |", "|---|---:|---:|"]
    for key, name in (("primary_macro_auc", "主要：被试等权 AUROC"), ("auc", "合并样本 AUROC（次要）"),
                      ("balanced_accuracy", "平衡准确率"), ("sensitivity", "走神灵敏度"), ("specificity", "专注特异度"), ("average_precision", "平均精确率 AP")):
        lines.append(f"| {name} | {_number(t[key])} | {_number(original[key])} |")
    lines.extend(["", f"主要 AUROC 的被试重采样 95% CI：{t['primary_auc_ci95']}。",
                  f"模型相对原指数的配对 AUROC 差异：{report['paired_auc_difference']}。", "",
                  f"独立测试：{t['participants']} 人 / {t['n']} 条；主要分析纳入 {t['primary_participants']} 人。",
                  f"二分类标签存在时的脑电覆盖率：{coverage['eeg_coverage_given_binary_label']:.1%}；全部正式探针最终可用率：{coverage['end_to_end_probe_coverage']:.1%}。",
                  f"冻结阈值：{cv['threshold']:.6f}；选择的 L2：{cv['selected_l2']}。",
                  "", "## 数据与排除", "", _protocol_description(report), "", f"排除探针：{report['data']['excluded_rows']}。原因计数（同一探针可有多种原因）：",
                  "", canonical(report['data']['excluded_reasons']), "", "## 每位测试参与者", "",
                  "| 参与者 | 专注标签 | 走神标签 | 模型 AUROC | 原指数 AUROC | 纳入主要分析 |", "|---|---:|---:|---:|---:|---|"])
    old = {r["participant"]: r for r in original["per_participant"]}
    for person in t["per_participant"]:
        lines.append(f"| {person['participant']} | {person['on_task']} | {person['mind_wandering']} | {_number(person['auc'])} | {_number(old[person['participant']]['auc'])} | {'是' if person['primary_eligible'] else '否'} |")
    lines.extend(["", "## 适用范围与限制", "", *[f"- {s}" for s in report["limitations"]], "",
                  "模型权重是统计预测依据，受特征相关性影响，不代表脑机制或因果关系。",
                  "测试结果不参与参数或阈值更新；需要修改模型时，应建立新的开发轮次与独立测试集。", ""])
    return "\n".join(lines)


def report_html(report, markdown):
    t = report["test"]
    cells = []
    for person in t["per_participant"]:
        cells.append("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in
                     (person["participant"], person["on_task"], person["mind_wandering"], _number(person["auc"]))) + "</tr>")
    # Standalone, printable report; no CDN or JavaScript required.
    bins = [b for b in t["calibration_bins"] if b["n"]]
    dots = "".join(f'<circle cx="{40+280*b["mean_prediction"]:.2f}" cy="{320-280*b["observed_fraction"]:.2f}" r="5" fill="#16776b"><title>n={b["n"]}</title></circle>' for b in bins)
    chart = f'<svg viewBox="0 0 360 355" role="img" aria-label="独立测试概率校准曲线"><path d="M40 40V320H320 M40 320L320 40" fill="none" stroke="#aaa"/><text x="70" y="347">预测走神概率（横）/实际比例（纵）</text>{dots}</svg>'
    summary = []
    for key, name in (("primary_macro_auc", "主要：被试等权 AUROC"), ("auc", "合并样本 AUROC（次要）"),
                      ("balanced_accuracy", "平衡准确率"), ("sensitivity", "走神灵敏度"),
                      ("specificity", "专注特异度"), ("average_precision", "平均精确率 AP")):
        summary.append(f"<tr><td>{name}</td><td>{_number(t[key])}</td><td>{_number(report['original_index'][key])}</td></tr>")
    limitation_list = "".join(f"<li>{html.escape(s)}</li>" for s in report["limitations"])
    exclusions = "".join(f"<li>{html.escape(reason)}：{count}</li>" for reason, count in report["data"]["excluded_reasons"].items()) or "<li>没有排除记录。</li>"
    exploratory = report["evidence_status"] != "independent_test_sufficient_size"
    coverage = report["coverage"]
    return ("<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>真人模型独立测试报告</title>"
            "<style>body{font:16px/1.7 system-ui;max-width:1050px;margin:36px auto;padding:0 24px;color:#18332e}h1{font-size:28px}h2{font-size:21px;margin-top:32px}.id{font:13px/1.7 monospace;overflow-wrap:anywhere}.notice{background:#eef5f3;padding:16px;border-left:4px solid #16776b}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #ccd8d5;padding:9px;text-align:left}th{background:#eef5f3}svg{width:360px;max-width:100%}@media print{body{margin:0}h2{break-after:avoid}tr{break-inside:avoid}}</style></head><body>"
            "<h1>真人脑电注意状态模型独立测试报告</h1>"
            f"<p class='id'>模型 {html.escape(report['model_id'])}<br>数据集 {html.escape(report['dataset_id'])}<br>生成时间 {html.escape(report['created_utc'])}</p>"
            "<p>标签 0=当前任务专注；1=内部走神。开发组用于选择模型、校准和阈值，模型文件冻结后才评估独立测试组。</p>"
            f"<p class='notice'>{'探索性结果：主要分析的有效测试参与者不足20人，暂不形成稳定性能结论。' if exploratory else '已达到预定的独立测试人数要求；性能和适用范围以本报告为准。'}</p>"
            "<h2>独立测试与原指数比较</h2><table><tr><th>指标</th><th>真人模型</th><th>原专注指数</th></tr>" + "".join(summary) + "</table>"
            f"<p>主要分析 {t['primary_participants']} 人；全部测试 {t['participants']} 人 / {t['n']} 条探针。主要 AUROC 的95%置信区间：{html.escape(str(t['primary_auc_ci95']))}。按被试重采样2000次。</p>"
            f"<p>模型相对原指数的配对 AUROC 差异：{html.escape(str(report['paired_auc_difference']))}。</p>"
            "<h2>数据覆盖与排除</h2>"
            f"<p>{html.escape(_protocol_description(report))}</p>"
            f"<p>二分类标签存在时的脑电覆盖率 {coverage['eeg_coverage_given_binary_label']:.1%}；全部正式探针最终可用率 {coverage['end_to_end_probe_coverage']:.1%}。排除记录 {report['data']['excluded_rows']} 条，同一条可有多种排除原因。</p><ul>{exclusions}</ul>"
            "<h2>概率校准检查</h2><p>蓝绿色点来自封存测试组，对角线表示理想校准。每个点的样本数可悬停查看；此检查未重新拟合模型。</p>" + chart +
            "<h2>每位测试参与者</h2><table><tr><th>参与者</th><th>专注标签</th><th>走神标签</th><th>模型 AUROC</th></tr>" + "".join(cells) + "</table>"
            "<h2>适用范围与限制</h2><ul>" + limitation_list + "</ul><p>模型参数、折分、阈值及逐探针预测见同包 JSON / CSV。开发组折外成绩参与模型选择，不能替代本报告的独立测试成绩。</p></body></html>")


def run_training(data, output, progress=lambda stage, fraction: None):
    """Freeze model files before test evaluation. Existing output is never overwritten."""
    if not data["test"] or len({r["label"] for r in data["test"]}) != 2:
        raise ValueError("一键训练评估需要独立测试组的两类有效标签，请先补齐测试数据。")
    output = Path(output)
    if output.exists():
        raise FileExistsError("训练输出目录已存在，请使用新的目录。")
    bundle, cv = train_development(data, progress)
    output.mkdir(parents=True)
    write_json(output / "attention_model.json", bundle)
    write_json(output / "cv_results.json", cv)
    write_json(output / "splits.json", {"development": bundle["trained_participants"],
                                       "test": sorted({r["participant"] for r in data["test"]}), "folds": cv["folds"]})
    write_json(output / "freeze.json", {"model_id": bundle["model_id"], "dataset_id": data["dataset_id"],
                                       "frozen_utc": datetime.now(timezone.utc).isoformat(), "test_opened": False})
    freeze = json.loads((output / "freeze.json").read_text(encoding="utf-8"))
    write_json(output / "freeze.json", {**freeze, "test_opened": True,
                                       "test_opened_utc": datetime.now(timezone.utc).isoformat()})
    report, predictions = independent_test(data, bundle, progress)
    report["frozen_before_test"] = True
    write_json(output / "evaluation.json", report)
    write_json(output / "training_config.json", bundle["training_config"])
    with (output / "test_predictions.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    md = report_markdown(report, cv)
    (output / "report.md").write_text(md, encoding="utf-8")
    (output / "report.html").write_text(report_html(report, md), encoding="utf-8")
    (output / "MODEL_CARD.md").write_text("# 模型说明\n\n" + bundle["scope"] + "\n\n" + "\n".join(report["limitations"]) +
                                         "\n\n标签 1=内部走神；实时采用探针前12～2秒定义的时间聚合。信号、单位、设备或引擎不匹配时拒绝输出。\n", encoding="utf-8")
    write_json(output / "SHA256.json", {p.name: __import__("hashlib").sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()})
    with zipfile.ZipFile(output / "model_bundle.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            if path.is_file() and path.name != "model_bundle.zip":
                archive.write(path, path.name)
    progress("模型、报告与复现材料已导出", 1.)
    return {"model_id": bundle["model_id"], "dataset_id": data["dataset_id"], "evaluation": report,
            "selected_l2": cv["selected_l2"], "threshold": cv["threshold"], "output": str(output)}
