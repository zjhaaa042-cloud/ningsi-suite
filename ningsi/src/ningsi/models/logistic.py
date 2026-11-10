"""逻辑回归基线模型：特征可解释、可在本地训练与部署（文档 6.4）。

模型只使用频带相对功率与比值特征；训练/验证/测试按被试划分，指标含准确率与 AUC；
权重以 JSON 保存，便于版本化与本地推断。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ningsi import config

FEATURE_NAMES = ("rel_theta", "rel_alpha", "rel_beta", "log_beta_theta", "log_alpha_beta", "log_theta_alpha")


def features_from_window(window) -> dict:
    """从单个窗特征中取模型输入；与报告中的指标口径一致。"""
    rel = window.rel
    def ratio(top: str, bottom: str) -> float:
        return math.log(max(rel.get(top, 0.0), config.INDEX_FLOOR) / max(rel.get(bottom, 0.0), config.INDEX_FLOOR))
    return {
        "rel_theta": float(rel.get("theta", 0.0)),
        "rel_alpha": float(rel.get("alpha", 0.0)),
        "rel_beta": float(rel.get("beta", 0.0)),
        "log_beta_theta": ratio("beta", "theta"),
        "log_alpha_beta": ratio("alpha", "beta"),
        "log_theta_alpha": ratio("theta", "alpha"),
    }


def feature_matrix(windows) -> tuple:
    import numpy as np
    rows = [features_from_window(window) for window in windows]
    matrix = np.array([[row[name] for name in FEATURE_NAMES] for row in rows], dtype=float)
    return matrix, list(FEATURE_NAMES)


@dataclass
class LinearModel:
    weights: np.ndarray
    bias: float
    mean: np.ndarray
    scale: np.ndarray
    feature_names: tuple = FEATURE_NAMES
    classes: tuple = (0, 1)
    spec: str = "logistic-v1"
    trained_subjects: tuple = ()
    metrics: dict = field(default_factory=dict)

    def predict_proba(self, matrix) -> np.ndarray:
        x = (np.asarray(matrix, dtype=float) - self.mean) / self.scale
        logits = x @ self.weights + self.bias
        return 1.0 / (1.0 + np.exp(-logits))

    def predict(self, matrix, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(matrix) >= threshold).astype(int)

    def as_dict(self) -> dict:
        return {
            "spec": self.spec,
            "feature_names": list(self.feature_names),
            "classes": list(self.classes),
            "weights": [float(value) for value in self.weights],
            "bias": float(self.bias),
            "mean": [float(value) for value in self.mean],
            "scale": [float(value) for value in self.scale],
            "trained_subjects": list(self.trained_subjects),
            "metrics": self.metrics,
        }

    def save(self, path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path) -> "LinearModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            weights=np.array(payload["weights"], dtype=float),
            bias=float(payload["bias"]),
            mean=np.array(payload["mean"], dtype=float),
            scale=np.array(payload["scale"], dtype=float),
            feature_names=tuple(payload["feature_names"]),
            classes=tuple(payload["classes"]),
            spec=payload["spec"],
            trained_subjects=tuple(payload.get("trained_subjects", ())),
            metrics=payload.get("metrics", {}),
        )


def _binary_labels(labels):
    y = np.asarray(labels, dtype=float)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    if y.ndim != 1 or not y.size:
        raise ValueError("标签必须是非空一维数组或列向量")
    if not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("标签只能为有限的 0 或 1")
    return y


def _validated_xy(matrix, labels):
    x = np.asarray(matrix, dtype=float)
    y = _binary_labels(labels)
    if x.ndim != 2 or min(x.shape) == 0 or x.shape[0] != y.size:
        raise ValueError("特征矩阵与标签数量不一致")
    if not np.isfinite(x).all():
        raise ValueError("特征不得含 NaN/Inf")
    return x, y


def fit_logistic(matrix, labels, epochs: int = 800, lr: float = 0.35, l2: float = 1e-3,
                 trained_subjects=()) -> LinearModel:
    x, y = _validated_xy(matrix, labels)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    scale = np.where(std < 1e-9, 1.0, std)
    xs = (x - mean) / scale
    weights = np.zeros(x.shape[1], dtype=float)
    bias = 0.0
    n = xs.shape[0]
    for _ in range(epochs):
        logits = xs @ weights + bias
        probs = 1.0 / (1.0 + np.exp(-logits))
        error = probs - y
        grad_w = xs.T @ error / n + l2 * weights
        grad_b = float(error.mean())
        weights -= lr * grad_w
        bias -= lr * grad_b
    return LinearModel(weights=weights, bias=bias, mean=mean, scale=scale, trained_subjects=tuple(trained_subjects))


def roc_auc(labels, probabilities) -> float | None:
    """秩统计 AUC；只有一种标签时不可计算，返回 None。"""
    truth = _binary_labels(labels)
    scores = np.asarray(probabilities, dtype=float)
    if scores.shape != truth.shape or not np.isfinite(scores).all():
        raise ValueError("概率必须是与标签等长的有限一维数组")
    if np.any((scores < 0) | (scores > 1)):
        raise ValueError("概率必须在 [0, 1] 范围内")
    pairs = sorted(zip(scores, truth), key=lambda item: item[0])
    positives = sum(1 for _, label in pairs if label == 1)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return None
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index
        while end + 1 < len(pairs) and pairs[end + 1][0] == pairs[index][0]:
            end += 1
        average_rank = (index + end) / 2.0 + 1
        for position in range(index, end + 1):
            if pairs[position][1] == 1:
                rank_sum += average_rank
        index = end + 1
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def evaluate(model: LinearModel, matrix, labels) -> dict:
    x, truth = _validated_xy(matrix, labels)
    probabilities = np.asarray(model.predict_proba(x), dtype=float)
    auc = roc_auc(truth, probabilities)
    predicted = (probabilities >= 0.5).astype(int)
    tp = int(np.sum((predicted == 1) & (truth == 1)))
    tn = int(np.sum((predicted == 0) & (truth == 0)))
    fp = int(np.sum((predicted == 1) & (truth == 0)))
    fn = int(np.sum((predicted == 0) & (truth == 1)))
    total = max(1, truth.size)
    return {
        "n": int(truth.size),
        "accuracy": round((tp + tn) / total, 4),
        "sensitivity": round(tp / (tp + fn), 4) if (tp + fn) else 0.0,
        "specificity": round(tn / (tn + fp), 4) if (tn + fp) else 0.0,
        "auc": round(auc, 4) if auc is not None else None,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def subject_split(participants, seed: int = 7, ratios=(0.6, 0.2, 0.2)) -> dict:
    """按被试划分，保证同一被试不跨集合（避免窗口泄漏）。"""
    unique = sorted(set(participants))
    state = seed or 1
    ordered = list(unique)
    for index in range(len(ordered) - 1, 0, -1):
        state = (1103515245 * state + 12345) % (2 ** 31)
        swap = state % (index + 1)
        ordered[index], ordered[swap] = ordered[swap], ordered[index]
    n = len(ordered)
    n_train = max(1, int(round(n * ratios[0])))
    n_val = max(1, int(round(n * ratios[1]))) if n - n_train > 1 else 0
    return {
        "train": ordered[:n_train],
        "validation": ordered[n_train:n_train + n_val],
        "test": ordered[n_train + n_val:],
    }
