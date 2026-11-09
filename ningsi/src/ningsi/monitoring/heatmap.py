"""状态热力图：0–1 评分分档、颜色映射与 SVG 渲染（文档 6.6）。

低质量窗口不出数值，用独立的"缺失"色块表示，避免把伪迹窗画成状态变化。
"""

from __future__ import annotations

from ningsi import config

MISSING_COLOR = "#6b6b6b"
MISSING_LABEL = "低质量缺失"


def band_of(score) -> dict:
    """返回评分所属档位；score 为 None 表示该窗质量不合格。"""
    if score is None:
        return {"index": -1, "label": MISSING_LABEL, "color": MISSING_COLOR, "low": None, "high": None, "score": None}
    value = min(1.0, max(0.0, float(score)))
    for index, (low, high, label, color) in enumerate(config.HEATMAP_BANDS):
        if value < high or index == len(config.HEATMAP_BANDS) - 1:
            return {"index": index, "label": label, "color": color, "low": low, "high": high, "score": round(value, 4)}
    return {"index": len(config.HEATMAP_BANDS) - 1, "label": config.HEATMAP_BANDS[-1][2],
            "color": config.HEATMAP_BANDS[-1][3], "low": 0.0, "high": 1.0, "score": round(value, 4)}


def legend() -> list:
    return [{"label": label, "color": color, "range": f"{low:.2f}–{high:.2f}"}
            for low, high, label, color in config.HEATMAP_BANDS] + [
        {"label": MISSING_LABEL, "color": MISSING_COLOR, "range": "—"}
    ]


def to_rows(series, columns: int = 30) -> list:
    """series: [(t, score 或 None), ...] → 按列数折行的档位矩阵。"""
    cells = [band_of(score) | {"t": float(t)} for t, score in series]
    return [cells[index:index + columns] for index in range(0, len(cells), columns)]


def render_text(series, columns: int = 30) -> str:
    rows = to_rows(series, columns)
    lines = []
    for row in rows:
        lines.append("".join("■" if cell["index"] >= 0 else "□" for cell in row))
    lines.append("档位：" + "  ".join(f"{item['range']} {item['label']}" for item in legend()))
    return "\n".join(lines)


def render_svg(series, cell: float = 14.0, columns: int = 30, title: str = "状态热力图") -> str:
    rows = to_rows(series, columns)
    width = max(1, max((len(row) for row in rows), default=0)) * cell + 2 * cell
    height = len(rows) * cell + 5 * cell
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
        f'viewBox="0 0 {width:.0f} {height:.0f}" font-family="sans-serif" font-size="11">',
        f'<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{cell:.0f}" y="{cell:.0f}" font-size="13">{title}</text>',
    ]
    for row_index, row in enumerate(rows):
        for column_index, cell_data in enumerate(row):
            x = cell + column_index * cell
            y = 2 * cell + row_index * cell
            parts.append(
                f'<rect x="{x:.0f}" y="{y:.0f}" width="{cell - 1:.0f}" height="{cell - 1:.0f}" '
                f'fill="{cell_data["color"]}"><title>{cell_data["t"]:.1f}s '
                f'{cell_data["label"]}</title></rect>'
            )
    legend_y = 2 * cell + len(rows) * cell + cell
    for index, item in enumerate(legend()):
        x = cell + index * 6 * cell
        parts.append(f'<rect x="{x:.0f}" y="{legend_y:.0f}" width="{cell - 2:.0f}" height="{cell - 2:.0f}" fill="{item["color"]}"/>')
        parts.append(f'<text x="{x + cell:.0f}" y="{legend_y + cell - 2:.0f}">{item["range"]} {item["label"]}</text>')
    parts.append("</svg>")
    return "\n".join(parts)
