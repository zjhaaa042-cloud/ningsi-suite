"""联合评估与报告：直接复用引擎的判定逻辑与报告模板，保证三端结论一致。"""

from __future__ import annotations

import math

from ningsi import config
from ningsi.assessment.joint import assess as joint_assess
from ningsi.assessment.report import SessionReport


def eeg_summary(*, baseline, quality, indicator_summary) -> dict:
    """构造联合评估需要的脑电侧输入（缺一不可，缺失时明确标注不可用）。"""
    def mean_of(name):
        item = (indicator_summary or {}).get(name) or {}
        return item.get("mean")

    values = {name: mean_of(name) for name in ("focus", "relax", "load")}
    baseline_valid = bool(baseline is not None and baseline.valid)
    scores_valid = all(isinstance(values[name], (int, float)) and math.isfinite(values[name])
                       for name in ("focus", "load"))
    if not baseline_valid:
        reason = "睁眼基线不可用"
        reason_code = "baseline_invalid"
    elif not (quality or {}).get("windows_total") and not indicator_summary:
        reason = "未采集独立监测阶段脑电；训练指数单独展示"
        reason_code = "monitor_not_collected"
    elif not scores_valid:
        reason = "监测阶段缺少有效的专注或负荷指数"
        reason_code = "missing_indicators"
    else:
        reason = ""
        reason_code = ""
    return {
        "available": baseline_valid and scores_valid,
        "unavailable_reason": reason,
        "unavailable_code": reason_code,
        "valid_ratio": (quality or {}).get("valid_ratio", 0.0),
        **values,
        "spec": config.INDICATOR_SPEC,
        "baseline_spec": config.BASELINE_SPEC,
    }


def assess(eeg, scale_objects=None, behavior=None):
    return joint_assess(eeg, scale_objects or {}, behavior or {})


def compose_report(*, participant: str, session: str, run: str, indicators: dict,
                   quality: dict, scales: dict, behavior: dict, assessment,
                   extras: dict | None = None) -> SessionReport:
    return SessionReport(
        participant=participant,
        session=session,
        run=run,
        indicators=indicators,
        quality=quality,
        scales=scales,
        behavior=behavior,
        assessment=assessment,
        extras=extras or {},
    )


def write_report(report: SessionReport, reports_dir, stem: str) -> dict:
    """写 Markdown 与 JSON 两份报告，返回路径字典。"""
    from pathlib import Path

    target = Path(reports_dir)
    target.mkdir(parents=True, exist_ok=True)
    md_path = target / f"{stem}_report.md"
    json_path = target / f"{stem}_report.json"
    html_path = target / f"{stem}_report.html"
    md_path.write_text(report.to_markdown(), encoding="utf-8")
    json_path.write_text(report.to_json(), encoding="utf-8")
    from ningsi.assessment.diagnostics import render_html
    html_path.write_text(render_html(report.as_dict()), encoding="utf-8")
    return {"report_md": md_path, "report_json": json_path, "report_html": html_path}
