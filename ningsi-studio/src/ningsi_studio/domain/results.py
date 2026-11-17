"""Completion-page EEG summaries with an explicit phase and scoring scope."""

import math
from collections import Counter

from ningsi.signal.indicators import IndicatorResult, summarize_indicators


def training_indicator_overview(windows):
    """Summarize saved, unsmoothed A-grade training scores only.

    This descriptive result is separate from monitor metrics and joint assessment.
    Saved diagnostic windows also let pre-fix sessions recover their real scores.
    """
    rows = [row for row in windows if row.get("phase") == "training"]
    results = []
    missing = Counter()
    for row in rows:
        scores = row.get("scores") or {}
        if not row.get("usable"):
            missing["window_unusable"] += 1
        elif (row.get("graded") or {}).get("grade") != "A":
            missing["grade_not_a"] += 1
        elif not row.get("indicator_available"):
            missing[row.get("indicator_reason") or "indicator_unavailable"] += 1
        elif not all(isinstance(scores.get(key), (int, float))
                     and not isinstance(scores[key], bool)
                     and math.isfinite(scores[key]) and 0 <= scores[key] <= 1
                     for key in ("focus", "relax", "load")):
            missing["invalid_scores"] += 1
        else:
            results.append(IndicatorResult(scores={key: scores[key] for key in ("focus", "relax", "load")}))
    total, scored = len(rows), len(results)
    strict = sum(bool(row.get("usable")) for row in rows)
    reason = ""
    if not scored:
        if not rows:
            reason = "没有保存可复算的训练逐窗指数；请查看已有训练记录或重新采集。"
        elif missing.get("baseline_invalid"):
            reason = "睁眼基线无效，训练脑电指数不可评分；请检查基线覆盖和质量后重测。"
        else:
            reason = "没有同时满足A级、有效基线与完整指数的训练窗口；请查看采集质量。"
    return {
        "source_phase": "training",
        "source_label": "训练阶段",
        "summary": summarize_indicators(results),
        "quality": {"windows_total": total, "windows_usable": strict,
                    "windows_scored": scored,
                    "valid_ratio": round(strict / total, 4) if total else 0.0,
                    "scored_ratio": round(scored / total, 4) if total else 0.0},
        "excluded_reasons": dict(missing),
        "available": bool(results),
        "reason": reason,
        "note": "训练阶段A级窗口的未平滑指数均值（0–1），相对训练前睁眼基线；含反馈升级等待窗，"
                "与平滑后的奖励分数口径不同。重叠窗不是独立样本，本结果不进入独立测评或联合评估。",
    }
