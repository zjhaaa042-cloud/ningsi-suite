"""Heatmap availability is separate from attention score and quality rejection.

This module only describes existing window evidence. It never repairs signals,
relaxes quality checks, creates a baseline or computes a replacement score.
"""
from __future__ import annotations

from collections import Counter
from html import escape
import json
import math

from ningsi.assessment.diagnostics import REASONS
from ningsi.monitoring.heatmap import MISSING_COLOR, band_of, legend

SPEC = "heatmap-availability-v1"
STATUS_META = {
    "baseline_pending": ("信号可用 · 基线待确认", "#79aaa2"),
    "signal_limited": ("信号受限", "#d9ab71"),
    "no_data": ("等待完整数据", "#d7dee3"),
    "timing_issue": ("时标待核对", "#a8aac6"),
    "unscored": ("暂无状态指数", "#b7c1c9"),
}
REASON_LABELS = {
    **REASONS,
    "baseline_pending": "个人基线尚未建立",
    "baseline_invalid": "个人基线尚未合格",
    "baseline_configuration_mismatch": "基线与当前配置不一致",
    "window_unusable": "信号窗口未通过质量检查",
    "clock_unsynchronized": "时标尚未同步",
    "preprocessing_warmup": "信号处理预热",
    "unknown_quality": "旧记录未保存逐窗质量依据",
    "score_unavailable": "尚无可用状态指数",
}
NOTE = ("状态颜色仅用于已有合格分数；信号可用、等待基线、信号受限与无数据分别展示。"
        "统计覆盖已保存的质检、基线、监测及训练窗；未采集的行为任务间隔不补窗。"
        "窗口有重叠，覆盖率按窗口计数，不代表独立样本数或连续有效时长。")


def _number(value):
    return (float(value) if isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) else None)


def _codes(value):
    return list(dict.fromkeys(str(item) for item in value if isinstance(item, str) and item)) if isinstance(value, (tuple, list)) else []


def reason_label(code):
    if code.startswith("baseline_index_missing"):
        return "个人基线缺少所需指标"
    return REASON_LABELS.get(code, code)


def cell_from_payload(payload):
    """Describe a recorded WindowStep payload, or a legacy metric adaptation."""
    usable = payload.get("usable")
    usable = usable if isinstance(usable, bool) else None
    score = _number((payload.get("scores") or {}).get("focus"))
    if score is not None and not 0 <= score <= 1:
        score = None
    reasons = _codes(payload.get("reasons", (payload.get("quality") or {}).get("reasons", [])))
    indicator_reason = str(payload.get("indicator_reason") or "")
    # A rejected window is never promoted by a stray/stale numerical score.
    if usable is False or payload.get("indicator_available") is False:
        score = None
    if score is not None:
        status = "scored"
        usable = True
    elif "no_data" in reasons or "signal_too_short" in reasons:
        status = "no_data"
    elif any(code.startswith(("clock_", "timestamp_")) for code in reasons):
        status = "timing_issue"
    elif usable is False:
        status = "signal_limited"
    elif usable is True and indicator_reason.startswith("baseline_"):
        status = "baseline_pending"
    else:
        status = "unscored"
    band = band_of(score)
    if status != "scored":
        label, color = STATUS_META[status]
        band.update(label=label, color=color)
        if indicator_reason and indicator_reason != "window_unusable":
            reasons = _codes([*reasons, indicator_reason])
        if not reasons:
            reasons = ["window_unusable" if usable is False else
                       "unknown_quality" if usable is None else "score_unavailable"]
    t = _number(payload.get("t_end")) or 0.0
    phase = payload.get("phase") if isinstance(payload.get("phase"), str) else None
    window_id = payload.get("window_id")
    if window_id is None and type(payload.get("index")) is int:
        window_id = f"{phase or 'unknown'}:{payload['index']}"
    return {**band, "t": round(t, 3), "phase": phase,
            "window_id": str(window_id) if window_id is not None else None,
            "status": status, "signal_usable": usable, "reason_codes": reasons,
            "reason_labels": [reason_label(code) for code in reasons],
            "indicator_reason": indicator_reason, "elapsed_sec": _number(payload.get("elapsed_sec"))}


def provenance(payload):
    """Keep only the evidence needed to replay the timeline; no raw EEG/labels."""
    return {"spec": SPEC, "window_id": payload.get("window_id"),
            "index": payload.get("index"), "phase": payload.get("phase"),
            "usable": payload.get("usable"), "indicator_reason": payload.get("indicator_reason"),
            "reasons": _codes(payload.get("reasons", [])), "elapsed_sec": payload.get("elapsed_sec")}


def cells_from_metrics(rows):
    cells = []
    for record in rows:
        row = dict(record)
        if row.get("t_sec") is None:
            continue
        try:
            quality = json.loads(row.get("quality") or "{}")
        except (ValueError, TypeError):
            quality = {}
        if not isinstance(quality, dict):
            quality = {}
        usable = quality.get("usable")
        if not isinstance(usable, bool):
            ratio = _number(row.get("valid_ratio"))
            usable = ratio == 1 if ratio in (0., 1.) else None
        cells.append(cell_from_payload({**quality, "usable": usable,
            "t_end": row["t_sec"], "scores": {"focus": row.get("value")},
            "window_id": quality.get("window_id") or f"metric:{row['id']}"}))
    return cells


def response(uuid, cells):
    cells = list(cells)
    counts = Counter(cell["status"] for cell in cells)
    reasons = Counter(code for cell in cells for code in set(cell["reason_codes"]))
    phases = {}
    for cell in cells:
        key = cell.get("phase") or "unknown"
        phase = phases.setdefault(key, {"phase": key, "total": 0, "signal_usable": 0, "scored": 0})
        phase["total"] += 1
        phase["signal_usable"] += cell["signal_usable"] is True
        phase["scored"] += cell["status"] == "scored"
    total, scored = len(cells), counts["scored"]
    usable = sum(cell["signal_usable"] is True for cell in cells)
    summary = {"total": total, "scored": scored, "missing": total-scored,
        "signal_usable": usable, "unknown_quality": sum(cell["signal_usable"] is None for cell in cells),
        "signal_coverage": round(usable/total, 4) if total else None,
        "score_coverage": round(scored/total, 4) if total else None,
        "counts": {key: counts[key] for key in ("scored", *STATUS_META)},
        "reason_counts": [{"code": code, "label": reason_label(code), "count": count}
                          for code, count in reasons.most_common()],
        "phase_counts": list(phases.values()), "coverage_unit": "windows"}
    return {"uuid": uuid, "spec": SPEC, "columns": 30, "cells": cells,
            "legend": legend(), "availability_legend": [
                {"status": key, "label": label, "color": color} for key, (label, color) in STATUS_META.items()],
            "missing_color": MISSING_COLOR, "scored": scored, "missing": total-scored,
            "summary": summary, "note": NOTE}


def render_svg(cells, *, title="状态与信号可用性", columns=40):
    """Portable two-track artifact using the same cells as REST/SSE.

    A null score stays empty on the state track, while its real availability is
    visible on a separate track. Input labels are escaped, never SVG markup.
    """
    cells = list(cells)
    columns = max(1, min(60, int(columns)))
    summary = response("", cells)["summary"]
    groups = [cells[i:i+columns] for i in range(0, len(cells), columns)]
    width, cell_size, left = max(780, 90+columns*15), 14, 90
    height = 126 + max(1, len(groups))*66 + 94
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'viewBox="0 0 {width} {height}" font-family="sans-serif" font-size="11">',
             '<rect width="100%" height="100%" fill="#fff"/>',
             f'<text x="20" y="28" font-size="17" fill="#18383c">{escape(str(title))}</text>',
             f'<text x="20" y="51">状态可评分 {summary["scored"]}/{summary["total"]} 窗 · '
             f'信号合格 {summary["signal_usable"]}/{summary["total"]} 窗</text>',
             '<text x="20" y="73" fill="#52686c">上轨为已有专注指数，下轨为采集可用性；基线准备窗不等于信号不合格。</text>',
             '<text x="20" y="92" fill="#52686c">窗口有重叠，窗数不是独立样本数或连续有效时长；未采集的间隔不补窗。</text>']
    if not groups:
        parts.append('<text x="20" y="135" fill="#52686c">尚无已保存的脑电窗口。</text>')
    for group_index, group in enumerate(groups):
        top = 115 + group_index*66
        parts.extend([f'<text x="20" y="{top+11}">专注指数</text>',
                      f'<text x="20" y="{top+31}">信号可用性</text>'])
        for column, cell in enumerate(group):
            x = left + column*15
            scored = cell.get("status") == "scored"
            color = cell["color"] if scored else "#f0f3f4"
            signal_color = STATUS_META["baseline_pending"][1] if cell.get("signal_usable") is True else cell["color"]
            hint = (f'{cell.get("phase") or "阶段未记录"} / t={cell["t"]:g}s / {cell["label"]} / '
                    f'指数={cell["score"] if scored else "未评分"} / ' + '、'.join(cell.get("reason_labels", [])))
            parts.extend([
                f'<rect x="{x}" y="{top}" width="{cell_size}" height="14" rx="2" '
                f'fill="{escape(str(color), quote=True)}" stroke="#d7e0e2"><title>{escape(hint)}</title></rect>',
                f'<rect x="{x}" y="{top+20}" width="{cell_size}" height="10" rx="2" '
                f'fill="{escape(str(signal_color), quote=True)}"><title>{escape(hint)}</title></rect>'])
        parts.append(f'<text x="{left}" y="{top+46}" fill="#52686c">'
                     f'第 {group_index*columns+1}–{group_index*columns+len(group)} 窗</text>')
    y = 125+max(1, len(groups))*66
    parts.append(f'<text x="20" y="{y}" fill="#52686c">专注指数（相对个人基线，0–1）</text>')
    for index, item in enumerate(legend()[:-1]):
        x = 20+index*148
        parts.extend([f'<rect x="{x}" y="{y+10}" width="12" height="12" rx="2" fill="{item["color"]}"/>',
                      f'<text x="{x+17}" y="{y+20}">{escape(item["range"])}</text>'])
    signal_legend = [("信号可用", STATUS_META["baseline_pending"][1]),
                     *[STATUS_META[key] for key in ("signal_limited", "no_data", "timing_issue", "unscored")]]
    for index, (label, color) in enumerate(signal_legend):
        x = 20+index*148
        parts.extend([f'<rect x="{x}" y="{y+40}" width="12" height="12" rx="2" fill="{color}"/>',
                      f'<text x="{x+17}" y="{y+50}">{escape(label)}</text>'])
    parts.append('</svg>')
    return '\n'.join(parts)
