"""专注度、放松度、认知负荷三个可解释指标（文档 6.7、8.3）。

每个指标先由频带功率比值（β/θ、α/β、θ/α）定义，再相对个体基线做 z 标准化，
最后映射到 0–1 评分。报告同时保留原始频带功率、基线统计量与 z 值，任一评分都可回算。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ningsi import config
from ningsi.signal.baseline import Baseline, index_values


@dataclass(frozen=True)
class IndicatorResult:
    scores: dict = field(default_factory=dict)     # focus / relax / load -> 0..1
    index_z: dict = field(default_factory=dict)    # 比值指标的 z 值
    band_z: dict = field(default_factory=dict)     # 各频带相对功率的 z 值（报告用）
    basis: dict = field(default_factory=dict)      # 指标 -> (分子频带, 分母频带)
    baseline_spec: str = config.BASELINE_SPEC
    spec: str = config.INDICATOR_SPEC
    available: bool = True
    reason: str = ""

    def score(self, name: str) -> float | None:
        return self.scores.get(name)

    def as_dict(self) -> dict:
        return {
            "spec": self.spec,
            "baseline_spec": self.baseline_spec,
            "available": self.available,
            "reason": self.reason,
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "index_z": {k: round(v, 4) for k, v in self.index_z.items()},
            "band_z": {k: round(v, 4) for k, v in self.band_z.items()},
            "basis": {k: list(v) for k, v in self.basis.items()},
        }


def score_from_z(value: float) -> float:
    """逻辑映射：z=0 → 0.5；输出裁剪到 [0,1]。"""
    raw = 1.0 / (1.0 + math.exp(-config.SCORE_GAIN * float(value)))
    return min(config.SCORE_MAX, max(config.SCORE_MIN, raw))


def compute_indicators(window, baseline: Baseline) -> IndicatorResult:
    """window: 质量合格的 WindowFeatures；baseline: 同设备同配置的个体基线。"""
    if window is None or not window.usable:
        return IndicatorResult(available=False, reason="window_unusable")
    if baseline is None or not baseline.valid:
        return IndicatorResult(available=False, reason="baseline_invalid")

    index_z: dict[str, float] = {}
    missing: list[str] = []
    for name, value in index_values(window).items():
        z = baseline.z_index(name, value)
        if z is None:
            missing.append(name)
        else:
            index_z[name] = z
    if missing:
        return IndicatorResult(available=False, reason="baseline_index_missing:" + ",".join(missing))

    band_z = {k: baseline.z_band(k, v) for k, v in sorted(window.rel.items())}
    scores = {name: score_from_z(z) for name, z in index_z.items()}
    return IndicatorResult(
        scores=scores,
        index_z=index_z,
        band_z={k: v for k, v in band_z.items() if v is not None},
        basis={k: tuple(v) for k, v in config.INDICATORS.items()},
        baseline_spec=baseline.spec,
    )


def aggregate(windows, results=None) -> dict:
    """Run 级汇总：可用窗比例、各指标均值/波动、不可用原因计数。"""
    total = len(windows)
    usable = [w for w in windows if w.usable]
    reasons: dict[str, int] = {}
    for w in windows:
        for reason in w.quality.reasons:
            reasons[reason] = reasons.get(reason, 0) + 1
    summary = {
        "windows_total": total,
        "windows_usable": len(usable),
        "valid_ratio": round(len(usable) / total, 4) if total else 0.0,
        "unusable_reasons": reasons,
    }
    if results:
        summary["indicators"] = summarize_indicators(results)
    return summary


def summarize_indicators(results) -> dict:
    """对指标序列做均值/标准差汇总；无可用结果时返回空。"""
    usable = [r for r in results if r.available]
    if not usable:
        return {}
    out = {}
    for name in sorted(usable[0].scores):
        values = [r.scores[name] for r in usable]
        mean = sum(values) / len(values)
        if len(values) > 1:
            std = (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5
        else:
            std = 0.0
        out[name] = {"mean": round(mean, 4), "std": round(std, 4), "n": len(values)}
    return out
