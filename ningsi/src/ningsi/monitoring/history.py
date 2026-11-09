"""历史记录与周/月趋势（文档 7.2.11）。

只对同一被试、同一设备配置的记录做趋势；设备或佩戴方式变化时明确标注不可比。
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from ningsi.signal.baseline import can_compare


def append_record(root, record: dict) -> Path:
    target = Path(root) / "history" / "sessions.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record}
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return target


def load_records(path) -> list:
    target = Path(path)
    if target.is_dir():
        target = target / "history" / "sessions.jsonl"
    if not target.exists():
        return []
    return [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]


def comparable(records, device: str, srate: float, channels: int, participant: str | None = None) -> tuple[list, list]:
    """返回 (可用记录, [(记录, 不可比原因), ...])。"""
    usable, rejected = [], []
    for record in records:
        if participant and record.get("participant") != participant:
            reject = "participant_mismatch"
        elif record.get("device") and device and record["device"] != device:
            reject = "device_changed"
        elif record.get("srate") and srate and abs(float(record["srate"]) - float(srate)) > 1e-6:
            reject = "sample_rate_changed"
        elif record.get("channels") and channels and int(record["channels"]) != int(channels):
            reject = "channel_count_changed"
        else:
            reject = ""
        (rejected.append((record, reject)) if reject else usable.append(record))
    return usable, rejected


def _period_key(stamp: str, period: str) -> str:
    moment = datetime.fromisoformat(stamp.replace("Z", "+00:00")) if stamp else datetime.now(timezone.utc)
    if period == "month":
        return f"{moment.year:04d}-{moment.month:02d}"
    year, week, _ = moment.isocalendar()
    return f"{year:04d}-W{week:02d}"


def aggregate(records, field: str, period: str = "week") -> list:
    buckets: dict[str, list] = {}
    for record in records:
        value = (record.get("indicators") or {}).get(field)
        if value is None:
            value = record.get(field)
        if value is None:
            continue
        buckets.setdefault(_period_key(record.get("recorded_at", ""), period), []).append(float(value))
    return [
        {"period": key, "mean": round(statistics.fmean(values), 4), "n": len(values),
         "std": round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0}
        for key, values in sorted(buckets.items())
    ]


def render_svg_trend(points, width: int = 480, height: int = 200, title: str = "趋势", y_label: str = "评分") -> str:
    pad = 36
    plotted = [p for p in points if p.get("mean") is not None]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" font-family="sans-serif" font-size="11">',
        f'<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{pad}" y="18" font-size="13">{title}</text>',
        f'<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="#999"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="#999"/>',
        f'<text x="6" y="{pad}" >1.0</text><text x="6" y="{height - pad}">0.0</text>',
        f'<text x="{pad}" y="{height - 8}">{y_label}</text>',
    ]
    if plotted:
        span = width - 2 * pad
        step = span / max(1, len(plotted) - 1)
        coords = []
        for index, point in enumerate(plotted):
            x = pad + index * step
            y = height - pad - (height - 2 * pad) * min(1.0, max(0.0, float(point["mean"])))
            coords.append(f"{x:.1f},{y:.1f}")
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="#2b6cb0"/>')
            parts.append(f'<text x="{x - 12:.1f}" y="{height - pad + 14:.1f}">{point["period"]}</text>')
        parts.append(f'<polyline points="{" ".join(coords)}" fill="none" stroke="#2b6cb0" stroke-width="2"/>')
    else:
        parts.append(f'<text x="{pad}" y="{height / 2}">暂无可比记录</text>')
    parts.append("</svg>")
    return "\n".join(parts)
