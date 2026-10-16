"""生成项目简介 PPT 配图与 PPTX。

配色取自 Okabe–Ito 色盲友好色板（Nature Methods 2011 推荐的 8 色方案），
叠加中性灰阶与浅色卡片底；热力图分档色与应用同源（ningsi.config.HEATMAP_BANDS），
界面、文档与答辩材料保持一致。
绘图与版式参考顶会论文插图的惯例：图内不重复标题、统一基线、细网格与参考线、
类别色不超过四种、顺序色用于热力图、字号与正文匹配（约 8–12 pt）。
配图用 Pillow 直接绘制（不依赖 matplotlib），PPT 用 python-pptx 生成，含逐页演讲备注。
"""

from __future__ import annotations

import math
import os
import random
import re
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml import parse_xml
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[2]          # D:/Desktop/ningsi
OUT = ROOT / "deliverables"
FIGS = OUT / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# ---- 配色：Okabe–Ito 色盲友好色板 + 中性灰阶 ----
INK = "#0E1B33"       # 主文字与深色底
NAVY = "#1B3B6F"      # 结构主色
BLUE = "#2E6DA4"
SKY = "#56B4E9"
TEAL = "#009E73"
AMBER = "#E69F00"
VERM = "#D55E00"
ROSE = "#CC79A7"
GREY = "#5B6B7C"
HAIR = "#DCE3EC"      # 细边框
RULE = "#B7C2D0"      # 连接线
PANEL = "#F5F8FC"     # 卡片底
FAINT = "#EEF2F8"     # 背景大字
WHITE = "#FFFFFF"

PPT_FONT = "微软雅黑"
FONT_CANDIDATES = {
    False: (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\simsun.ttc"),
    True: (r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\simsun.ttc"),
}
# 配图按 5.5 英寸宽摆放：1 px ≈ 0.264 pt，正文级字号取 40–46 px
TEXT = 40
LABEL_SIZE = 46
NOTE = 28


def rgb(value):
    """'#RRGGBB' 或 (r, g, b) → (r, g, b)。"""
    if isinstance(value, tuple):
        return value
    return tuple(int(value[index:index + 2], 16) for index in (1, 3, 5))


def mix(first, second, ratio):
    """按比例混合两色，ratio=0 取第一色。"""
    left, right = rgb(first), rgb(second)
    return tuple(round(left[index] + (right[index] - left[index]) * ratio) for index in range(3))


@lru_cache(maxsize=None)
def font(size: int, bold: bool = False):
    for path in FONT_CANDIDATES[bold]:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


class Canvas:
    """绘图画布：先登记卡片投影，统一模糊合成后再画主体，避免逐块重采样。"""

    def __init__(self, width, height, background=WHITE):
        self.image = Image.new("RGB", (width, height), rgb(background))
        self.pending = []

    def shadow(self, box, radius=16, dy=7, alpha=30):
        self.pending.append((box, radius, dy, alpha))

    def flush(self):
        if self.pending:
            layer = Image.new("RGBA", self.image.size, (0, 0, 0, 0))
            pen = ImageDraw.Draw(layer)
            for box, radius, dy, alpha in self.pending:
                pen.rounded_rectangle((box[0], box[1] + dy, box[2], box[3] + dy),
                                      radius=radius, fill=rgb(INK) + (alpha,))
            layer = layer.filter(ImageFilter.GaussianBlur(9))
            self.image = Image.alpha_composite(self.image.convert("RGBA"), layer).convert("RGB")
            self.pending = []
        return self

    @property
    def draw(self):
        return ImageDraw.Draw(self.image)

    def layer(self):
        return Image.new("RGBA", self.image.size, (0, 0, 0, 0))

    def merge(self, layer):
        self.image = Image.alpha_composite(self.image.convert("RGBA"), layer).convert("RGB")
        return self

    def save(self, path):
        self.image.save(path)
        return path


def label(draw, xy, text, size, color=INK, bold=False, anchor="la", spacing=0):
    """spacing 为字间距（像素），用于西文小标签的疏排。"""
    if spacing:
        x, y = xy
        for char in text:
            draw.text((x, y), char, font=font(size, bold), fill=rgb(color), anchor=anchor)
            x += draw.textlength(char, font=font(size, bold)) + spacing
        return
    draw.text(xy, text, font=font(size, bold), fill=rgb(color), anchor=anchor)


def width_of(draw, text, size, bold=False):
    return draw.textlength(text, font=font(size, bold))


def wrap(draw, text, size, limit, bold=False):
    """按像素宽度断行；优先在空格、·、→ 处断开，超长片段再按字断开。"""
    tokens = [token for token in re.split(r"(?<=[ ·→])", text) if token]
    lines, current = [], ""
    for token in tokens:
        candidate = current + token
        if current and width_of(draw, candidate, size, bold) > limit:
            lines.append(current.rstrip())
            current = token
        else:
            current = candidate
        while width_of(draw, current, size, bold) > limit and len(current) > 1:
            cut = len(current) - 1
            while cut > 1 and width_of(draw, current[:cut], size, bold) > limit:
                cut -= 1
            lines.append(current[:cut].rstrip())
            current = current[cut:]
    if current.strip():
        lines.append(current.rstrip())
    return lines or [""]


def paragraphs(draw, box, text, size, limit, color=INK, bold=False, leading=1.32, anchor="lm"):
    lines = wrap(draw, text, size, limit, bold)
    step = round(size * leading)
    height = step * len(lines)
    top = box[1] + (box[3] - box[1] - height) / 2 if anchor == "lm" else box[1]
    for offset, line in enumerate(lines):
        label(draw, (box[0], top + offset * step), line, size, color, bold)
    return len(lines)


def card(draw, box, radius=16, fill=WHITE, outline=HAIR, line=2):
    draw.rounded_rectangle(box, radius=radius, fill=rgb(fill), outline=rgb(outline), width=line)


def arrow(draw, start, end, color=RULE, width=4, head=13):
    draw.line([start, end], fill=rgb(color), width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    spread = math.radians(25)
    draw.polygon([end,
                  (end[0] - head * math.cos(angle - spread), end[1] - head * math.sin(angle - spread)),
                  (end[0] - head * math.cos(angle + spread), end[1] - head * math.sin(angle + spread))],
                 fill=rgb(color))


def dashed(draw, start, end, color=RULE, width=2, dash=10, gap=8):
    length = math.hypot(end[0] - start[0], end[1] - start[1])
    if not length:
        return
    step_x = (end[0] - start[0]) / length
    step_y = (end[1] - start[1]) / length
    position = 0.0
    while position < length:
        stop = min(position + dash, length)
        draw.line([(start[0] + step_x * position, start[1] + step_y * position),
                   (start[0] + step_x * stop, start[1] + step_y * stop)], fill=rgb(color), width=width)
        position = stop + gap


def draw_cover() -> Path:
    """封面底图：深色渐变 + 疏网格 + 频谱条与波形，文字由 PPTX 叠加（便于填团队名）。"""
    width, height = 1920, 1080
    canvas = Canvas(width, height, INK)
    pen = canvas.draw
    for y in range(height):
        pen.line([(0, y), (width, y)], fill=mix(INK, NAVY, min(1.0, y / height * 1.2)))
    art_layer = canvas.layer()
    art = ImageDraw.Draw(art_layer)
    for x in range(0, width, 48):
        art.line([(x, 0), (x, height)], fill=rgb(WHITE) + (10,), width=1)
    for y in range(0, height, 48):
        art.line([(0, y), (width, y)], fill=rgb(WHITE) + (10,), width=1)
    rng = random.Random(7)
    for index in range(38):
        amplitude = 40 + 210 * abs(math.sin(index * 0.63)) * (0.45 + rng.random() * 0.55)
        x = 1120 + index * 20
        art.rounded_rectangle((x, 520 - amplitude, x + 11, 520), radius=5,
                              fill=rgb(mix(SKY, TEAL, index / 37)) + (165,))
    art.line([(1120, 520), (1878, 520)], fill=rgb(SKY) + (90,), width=3)
    rng = random.Random(21)
    phase = rng.random() * 6
    points = []
    for index in range(0, 981, 5):
        moment = index / 980 * 11 + phase
        value = math.sin(moment * 2.1) * 30 + math.sin(moment * 5.7 + 1.2) * 14 + (rng.random() - 0.5) * 10
        points.append((1110 + index, 790 + value))
    art.line(points, fill=rgb(SKY) + (110,), width=3, joint="curve")
    art.line([(1110, 940), (1890, 940)], fill=rgb(WHITE) + (26,), width=2)
    for index, color in enumerate((AMBER, ROSE, TEAL, SKY)):
        art.ellipse((1122 + index * 30, 966, 1134 + index * 30, 978), fill=rgb(color) + (200,))
    canvas.merge(art_layer)
    return canvas.save(FIGS / "fig_cover.png")


def draw_architecture() -> Path:
    """四层架构 + 数据安全横条：左侧色块标层名，右侧要点断行，层间用短连线串联。"""
    width, height = 1500, 900
    canvas = Canvas(width, height)
    layers = [
        ("用户层", "USER", "学生与职场人群 · 个人用户 · 心理教师与辅导员 · 员工关怀与研究人员", SKY),
        ("交互层", "INTERFACE", "设备接入与质检 · 评估与量表 · 实时监测 · 训练主界面 · 报告与趋势", BLUE),
        ("智能层", "INTELLIGENCE", "四级处理链 → Welch 频谱 → 频带特征 → 三指标 → 质量门控 → 联合评估 → 训练建议", TEAL),
        ("支撑层", "PLATFORM", "采集抽象 · 口径与版本管理 · 证据回填 · 历史与趋势（JSONL + SVG）", GREY),
    ]
    left, right, chip = 70, 1430, 210
    box_h, gap, top = 152, 22, 66
    for index in range(len(layers)):
        y = top + index * (box_h + gap)
        canvas.shadow((left, y, right, y + box_h))
    canvas.flush()
    pen = canvas.draw
    for index, (name, latin, detail, color) in enumerate(layers):
        y = top + index * (box_h + gap)
        card(pen, (left, y, right, y + box_h))
        pen.rounded_rectangle((left, y, left + chip, y + box_h), radius=16, fill=rgb(color))
        pen.rectangle((left + chip - 40, y, left + chip, y + box_h), fill=rgb(color))
        label(pen, (left + 28, y + box_h / 2 - 26), name, LABEL_SIZE, WHITE, True)
        label(pen, (left + 30, y + box_h / 2 + 14), latin, 18, mix(WHITE, color, 0.42), False, spacing=1)
        paragraphs(pen, (left + chip + 34, y, right - 10, y + box_h), detail, TEXT,
                   right - (left + chip + 34) - 10)
        if index < len(layers) - 1:
            pen.line([(left + chip / 2, y + box_h + 4), (left + chip / 2, y + box_h + gap - 4)],
                     fill=rgb(RULE), width=3)
    bar_top = top + len(layers) * (box_h + gap) + 26
    card(pen, (left, bar_top, right, bar_top + 86), radius=14, fill="#FDF4E3", outline=AMBER)
    pen.rounded_rectangle((left, bar_top, left + 9, bar_top + 86), radius=4, fill=rgb(AMBER))
    label(pen, ((left + right) / 2, bar_top + 43),
          "数据安全与合规：匿名编号 · 受限目录 · 不作医疗诊断 · 结论只给操作建议",
          36, INK, True, anchor="mm")
    return canvas.save(FIGS / "fig_architecture.png")


def draw_flow() -> Path:
    """一次会话九步流程：3×3 卡片 + 行间回流箭头，深色卡片为决定数据可信度的三个锚点。"""
    width, height = 1500, 860
    canvas = Canvas(width, height)
    steps = ["设备接入与质检", "量表填写", "静息基线", "行为任务", "实时指标与热力图",
             "联合评估报告", "神经反馈训练", "训练后复评", "趋势对比"]
    anchors = {0, 2, 6}
    left, card_w, card_h = 90, 430, 140
    gap_x, gap_y, top = 30, 105, 88
    for index in range(len(steps)):
        row, column = divmod(index, 3)
        if index not in anchors:
            x = left + column * (card_w + gap_x)
            y = top + row * (card_h + gap_y)
            canvas.shadow((x, y, x + card_w, y + card_h), radius=14, dy=6, alpha=24)
    canvas.flush()
    pen = canvas.draw
    for index, text in enumerate(steps):
        row, column = divmod(index, 3)
        x = left + column * (card_w + gap_x)
        y = top + row * (card_h + gap_y)
        anchor_step = index in anchors
        card(pen, (x, y, x + card_w, y + card_h), radius=14,
             fill=NAVY if anchor_step else WHITE, outline=NAVY if anchor_step else HAIR)
        badge = AMBER if anchor_step else mix(BLUE, WHITE, 0.84)
        pen.ellipse((x + 24, y + card_h / 2 - 24, x + 72, y + card_h / 2 + 24), fill=rgb(badge))
        label(pen, (x + 48, y + card_h / 2), str(index + 1), 26, INK if anchor_step else NAVY,
              True, anchor="mm")
        label(pen, (x + 92, y + card_h / 2), text, 34, WHITE if anchor_step else INK, anchor="lm")
        if column != 2:
            arrow(pen, (x + card_w + 6, y + card_h / 2), (x + card_w + gap_x - 8, y + card_h / 2))
    for row in range(2):
        bottom = top + row * (card_h + gap_y) + card_h
        turn = bottom + gap_y / 2
        start = (left + 2 * (card_w + gap_x) + card_w / 2, bottom + 6)
        end = (left + card_w / 2, top + (row + 1) * (card_h + gap_y) - 10)
        pen.line([start, (start[0], turn), (end[0], turn), end], fill=rgb(RULE), width=4, joint="curve")
        arrow(pen, (end[0], turn), end)
    legend_y = top + 3 * card_h + 2 * gap_y + 30
    pen.rounded_rectangle((left, legend_y, left + 34, legend_y + 34), radius=9, fill=rgb(NAVY))
    label(pen, (left + 48, legend_y + 17),
          "深色为关键锚点：质检决定数据能否使用，基线决定指标与谁比，训练闭环决定效果可复评",
          30, GREY, False, anchor="lm")
    return canvas.save(FIGS / "fig_flow.png")


def draw_indicators(summary: dict) -> Path:
    """三个可解释指标：0–1 横向条 + 0.5 基线参考线 + 0/0.5/1.0 刻度。"""
    width, height = 1500, 1000
    canvas = Canvas(width, height)
    metrics = [("focus", "专注度", "β/θ", BLUE), ("relax", "放松度", "α/β", TEAL), ("load", "认知负荷", "θ/α", VERM)]
    pen = canvas.draw
    bar_left, bar_right = 430, 1290
    span = bar_right - bar_left
    bar_h, row_gap, top = 86, 240, 200
    for index in range(len(metrics)):
        y = top + index * row_gap
        card(pen, (60, y - 36, 1440, y + bar_h + 36), radius=20, fill=PANEL, outline=PANEL)
    axis_y = top + 2 * row_gap + bar_h + 74
    pen.line([(bar_left, axis_y), (bar_right, axis_y)], fill=rgb(RULE), width=3)
    for fraction, text in ((0.0, "0.0"), (0.5, "0.5"), (1.0, "1.0")):
        x = bar_left + span * fraction
        pen.line([(x, axis_y), (x, axis_y + 12)], fill=rgb(RULE), width=3)
        label(pen, (x, axis_y + 26), text, NOTE, GREY, anchor="ma")
    label(pen, (bar_left + span * 0.5, axis_y + 74), "0.5 = 与个体基线持平", NOTE, NAVY, False, anchor="ma")
    dashed(pen, (bar_left + span * 0.5, top - 56), (bar_left + span * 0.5, top + 2 * row_gap + bar_h + 40),
           color=NAVY, width=3, dash=12, gap=9)
    for index, (key, name, ratio, color) in enumerate(metrics):
        y = top + index * row_gap
        value = max(0.0, min(1.0, float(summary.get(key, {}).get("mean", 0.5))))
        pen.rounded_rectangle((bar_left, y, bar_right, y + bar_h), radius=18, fill="#E4EBF3")
        pen.rounded_rectangle((bar_left, y, bar_left + max(30, span * value), y + bar_h), radius=18,
                              fill=rgb(color))
        label(pen, (100, y + 6), name, LABEL_SIZE, INK, True)
        label(pen, (102, y + 58), ratio, 30, mix(color, INK, 0.15), False)
        label(pen, (1420, y + bar_h / 2), f"{value:.2f}", TEXT, INK, True, anchor="rm")
        pen.ellipse((bar_left + span * value - 10, y + bar_h / 2 - 10,
                     bar_left + span * value + 10, y + bar_h / 2 + 10),
                    fill=rgb(WHITE), outline=rgb(color), width=4)
    return canvas.save(FIGS / "fig_indicators.png")


def draw_heatmap(series) -> Path:
    """状态热力图：上方为专注度轨迹，下方为同一时间轴上的分档色带，缺失窗单独成色。"""
    from ningsi.config import HEATMAP_BANDS
    from ningsi.monitoring.heatmap import MISSING_COLOR, MISSING_LABEL, band_of
    width, height = 1500, 880
    canvas = Canvas(width, height)
    pen = canvas.draw
    columns, cell = 30, 44
    rows = max(1, (len(series) + columns - 1) // columns)
    left = 90
    grid_right = left + columns * cell
    plot_top, plot_bottom = 104, 470
    for fraction, text in ((0.0, "0.0"), (0.5, "0.5"), (1.0, "1.0")):
        y = plot_bottom - (plot_bottom - plot_top) * fraction
        dashed(pen, (left, y), (grid_right, y), color=HAIR, width=2, dash=6, gap=8)
        label(pen, (left - 16, y), text, NOTE, NAVY if fraction == 0.5 else GREY,
              fraction == 0.5, anchor="rm")
    if series:
        usable = [(index, score) for index, (_, score) in enumerate(series) if score is not None]
        points = []
        for index, score in usable:
            x = left + index * cell + cell / 2
            y = plot_bottom - (plot_bottom - plot_top) * max(0.0, min(1.0, float(score)))
            points.append((x, y))
        if len(points) >= 2:
            pen.polygon([(points[0][0], plot_bottom), *points, (points[-1][0], plot_bottom)],
                        fill=rgb("#E8F1F9"))
            pen.line(points, fill=rgb(BLUE), width=5, joint="curve")
        for x, y in points:
            pen.ellipse((x - 8, y - 8, x + 8, y + 8), fill=rgb(WHITE), outline=rgb(BLUE), width=4)
    strip_top = 800 - 138 - rows * cell
    for index, (_, score) in enumerate(series):
        row, column = divmod(index, columns)
        x = left + column * cell
        y = strip_top + row * cell
        pen.rectangle((x + 2, y + 2, x + cell - 4, y + cell - 4), fill=rgb(band_of(score)["color"]))
    ruler_y = strip_top + rows * cell + 40
    for column in range(0, min(columns, len(series)), 5):
        x = left + column * cell + cell / 2
        pen.line([(x, strip_top + rows * cell + 8), (x, ruler_y - 10)], fill=rgb(RULE), width=2)
        label(pen, (x, ruler_y), f"{int(series[column][0])}s", NOTE, GREY, anchor="ma")
    label(pen, (left, 46), "每格 2 秒｜上：专注度轨迹　下：分档色带", NOTE, GREY)
    bands = list(HEATMAP_BANDS) + [(None, None, MISSING_LABEL, MISSING_COLOR)]
    for index, (low, high, text, color) in enumerate(bands):
        row, column = divmod(index, 3)
        x = left + column * 444
        y = 762 + row * 62
        pen.rounded_rectangle((x, y, x + 30, y + 30), radius=6, fill=rgb(color))
        caption = f"{low:.2f}–{high:.2f} {text}" if low is not None else text
        label(pen, (x + 42, y + 15), caption, NOTE, GREY, anchor="lm")
    return canvas.save(FIGS / "fig_heatmap.png")


def draw_trend(points) -> Path:
    """专注度周趋势：细网格 + 0.5 参考线 + 单点状态说明，不硬凑曲线。"""
    width, height = 1500, 880
    canvas = Canvas(width, height)
    pen = canvas.draw
    left, right, upper, bottom = 190, 1420, 120, 640
    for fraction in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        y = bottom - (bottom - upper) * fraction
        dashed(pen, (left, y), (right, y), color=HAIR, width=2, dash=6, gap=8)
        label(pen, (left - 22, y), f"{fraction:.1f}", NOTE, NAVY if fraction == 0.5 else GREY,
              fraction == 0.5, anchor="rm")
    dashed(pen, (left, (upper + bottom) / 2), (right, (upper + bottom) / 2),
           color=NAVY, width=2, dash=12, gap=9)
    label(pen, (left, upper - 34), "专注度评分（0–1）", NOTE, GREY)
    pen.line([(left, bottom), (right, bottom)], fill=rgb(RULE), width=3)
    pen.line([(left, upper), (left, bottom)], fill=rgb(RULE), width=3)
    if len(points) >= 2:
        step = (right - left) / (len(points) - 1)
        coords = []
        for index, point in enumerate(points):
            x = left + index * step
            y = bottom - (bottom - upper) * max(0.0, min(1.0, float(point["mean"])))
            coords.append((x, y))
        pen.polygon([(coords[0][0], bottom), *coords, (coords[-1][0], bottom)], fill=rgb("#E8F1F9"))
        pen.line(coords, fill=rgb(BLUE), width=5, joint="curve")
        for index, point in enumerate(points):
            x, y = coords[index]
            pen.ellipse((x - 11, y - 11, x + 11, y + 11), fill=rgb(WHITE), outline=rgb(BLUE), width=5)
            label(pen, (x, bottom + 38), str(point["period"]), NOTE, GREY, anchor="ma")
    else:
        if points:
            x = left + 0.16 * (right - left)
            y = bottom - (bottom - upper) * max(0.0, min(1.0, float(points[0]["mean"])))
            pen.ellipse((x - 13, y - 13, x + 13, y + 13), fill=rgb(WHITE), outline=rgb(BLUE), width=6)
            label(pen, (x + 36, y), f"{float(points[0]['mean']):.2f}", 32, INK, True, anchor="lm")
            label(pen, (x, bottom + 38), str(points[0]["period"]), NOTE, GREY, anchor="ma")
        note = "首次会话记录已写入，趋势随复评次数累积"
        box_width = width_of(pen, note, 32) + 88
        card(pen, ((left + right - box_width) / 2, 430, (left + right + box_width) / 2, 534),
             radius=16, fill=PANEL, outline=HAIR)
        label(pen, ((left + right) / 2, 482), note, 32, GREY, anchor="mm")
    return canvas.save(FIGS / "fig_trend.png")


SLIDES = [
    {
        "title": "凝思（Ningsi）",
        "subtitle": "AI+便携脑电设备的专注力强化训练系统\n每一段专注，都看得见",
        "bullets": ["赛题编号：A09 ｜ 命题企业：杭州金扬智能科技有限公司",
                    "团队：胆double天队（张嘉辉、李骞恒、黄柯语、叶瑞洋）",
                    "汇报人：张嘉辉（队长）　｜　2026 年 10 月"],
        "notes": "开场：凝思是一套基于便携脑电设备的专注力强化训练与心理状态评估系统，把脑电从能采到，做成能判、能练、能复评。",
    },
    {
        "title": "一、背景与痛点",
        "bullets": [
            "传统评估依赖问卷与访谈：主观、离散，反映不了当时的状态",
            "便携脑电设备已普及，但把设备变成产品还缺三件事",
            "缺一：把原始信号变成可信指标（伪迹与质量门控）",
            "缺二：把指标与量表、行为任务放进同一套证据体系",
            "缺三：把指标变成可执行的训练反馈与可比较的复评",
        ],
        "notes": "痛点的关键不在采不到信号，而在于采到看起来正常、实际已被伪迹污染的信号；同时指标也无法与量表、行为任务互相印证。",
    },
    {
        "title": "二、项目目标与闭环",
        "bullets": [
            "目标：构建可解释、可复算、边界清晰的专注力评估与训练系统",
            "闭环：设备接入与质检 → 量表与基线 → 任务或训练 → 指标反馈 → 趋势复评",
            "三类证据：脑电指标（客观连续）、标准化量表（可横向比较）、行为任务（直接反映表现）",
            "一致时给出确定结论，不一致时明确建议复测，不强行合并成单一分数",
        ],
        "notes": "闭环可以概括为：同一套指标、同一套相对基线，训练前后的状态可以对比，训练效果不依赖主观回忆。",
    },
    {
        "title": "三、创意：四条",
        "bullets": [
            "让采集更可信：逐窗质检 + 不可用原因记录，伪迹窗不计入指标与计时",
            "让特征有依据：只用 θ/α/β 频带功率与比值，每个评分都能回算",
            "让评估成体系：SAS/SDS + SART/PVT-B + 脑电指标联合判断",
            "让训练有反馈：实时专注度驱动反馈，目标随表现自适应",
        ],
        "notes": "四条创意各对应一个环节，每条都能指到具体机制与可观测效果。",
    },
    {
        "title": "四、产品架构：四层 + 数据安全横条",
        "bullets": [
            "用户层：四类角色，结论只对本人或授权角色可见",
            "交互层：设备接入与质检、评估与量表、实时监测、训练、报告与趋势",
            "智能层：四级处理链 → 频谱 → 频带特征 → 三指标 → 质量门控 → 联合评估 → 训练建议",
            "支撑层：采集抽象、口径与版本管理、证据回填、历史与趋势",
        ],
        "image": "fig_architecture.png",
        "notes": "架构图与正文一一对应：图中每个模块都能在代码仓库找到对应文件，报告里每个结论都能指回模块与口径版本。",
    },
    {
        "title": "五、用户使用流程：九步",
        "bullets": [
            "会话设置 → 设备接入与质检 → 量表填写 → 睁眼与闭眼基线",
            "行为任务 → 实时指标与热力图 → 联合评估报告 → 神经反馈训练 → 趋势复评",
            "以现有实现为例：一条命令可跑通全链路并产出报告、热力图与趋势图",
        ],
        "image": "fig_flow.png",
        "notes": "第 ② 步质检与第 ④ 步基线是可信度的两个锚点；九步在同一窗口内完成，不需要在多个程序之间搬运数据。",
    },
    {
        "title": "六、技术路线：四级处理链与频谱口径",
        "bullets": [
            "处理链：0.5 Hz 基线漂移校正 → 50/60 Hz 工频陷波 → 0.5–45 Hz 带通，全部零相位",
            "频谱：Welch 法，4 秒窗 / 2 秒分段 / 50% 重叠 / 汉宁窗 / NFFT 512（Δf 0.488 Hz）",
            "特征：θ/α/β/γ 相对功率、频带比值、波幅与通道相关系数",
            "每一步参数随窗留痕，同一段原始数据可被重新复算",
        ],
        "notes": "参数固定且可复算：频谱与滤波参数是同一份定义，文档、代码与报告三处一致；实现不依赖 SciPy，便于现场部署。",
    },
    {
        "title": "七、三个可解释指标",
        "bullets": [
            "专注度 = β/θ：β 相对增强且 θ 受抑制",
            "放松度 = α/β；认知负荷 = θ/α",
            "先相对个体基线做 z 标准化，再映射到 0–1；0.5 表示与基线持平",
            "同时输出原始频带功率、基线统计量与 z 值，避免黑箱分数",
        ],
        "image": "fig_indicators.png",
        "notes": "选择比值而非相对功率之差的原因：相对功率此消彼长会带来串扰；比值型指标同时符合生理含义与统计稳定性。",
    },
    {
        "title": "八、质量门控与实时预警",
        "bullets": [
            "单窗判定：幅度 >150 µV、恒定通道、通道跨度、30–45 Hz 肌电代理、0.5–4 Hz 眼电代理",
            "Run 级门槛：可用窗比例 <60% 或连续 3 个不可用窗即提示质量不足",
            "低专注预警：评分 <0.35 持续 20 秒（解除：≥0.45 持续 10 秒）",
            "高负荷预警：评分 ≥0.75 持续 30 秒（解除：≤0.65 持续 15 秒）",
            "质量不合格窗不参与计时，等于把预警时钟暂停，避免因动作伪迹误报",
        ],
        "notes": "质量门控要说明的是如何避免把伪迹当成状态：逐窗判定 + 记录不可用原因 + 伪迹窗停表。",
    },
    {
        "title": "九、三类证据联合评估",
        "bullets": [
            "流程：证据独立计算 → 质量前置判定 → 一致性检查 → 结论输出 → 边界提示 → 证据回填",
            "一致：输出压力或专注维度的明确结论与建议",
            "不一致：标注「结果不一致，建议复测」，不强行合并分数",
            "缺失：明确输出「数据不足」，并指出缺哪一个环节",
            "报告每条结论都附证据引用（指标、口径版本、序列集与种子）",
        ],
        "notes": "真实运行示例：脑电提示专注偏低而行为任务表现正常，系统输出不一致并建议复测，不合并成单一分数。",
    },
    {
        "title": "十、神经反馈训练闭环",
        "bullets": [
            "初始目标取自当次基线中位数对应的评分（0.5），保证近期可达",
            "分段训练：每段记录平均专注度、达标时间占比与波动幅度",
            "自适应：达标占比 ≥70% 且波动小 → 目标 +0.05、保持时长 +2 秒；占比 <40% 或波动大 → 下调",
            "训练前后各采一次静息基线，复评与训练同口径，可比性有明确判定",
        ],
        "notes": "训练过程全程留痕：目标、达标占比、波动与前后基线都被记录下来，训练效果不依赖主观回忆。",
    },
    {
        "title": "十一、可视化与趋势",
        "bullets": [
            "六个界面页签：状态指标、实时波形与频谱、状态热力图、训练、报告、趋势",
            "热力图五档着色，低质量窗用独立「缺失」色块，不把伪迹画成状态变化",
            "趋势按周/月聚合；设备、采样率或通道变化时明确标注不可比并排除",
        ],
        "images": ["ui_live.png", "ui_history.png"],
        "notes": "本页使用实机界面截图：左图为实时监测（含热力图分档色带与质量缺失色块），右图为历史与趋势。趋势按可比条件过滤后聚合，设备或口径变化时断开曲线并标注不可比；首次会话只有一个数据点属正常状态，它随复评次数累积，因此首次会话不给趋势结论。",
    },
    {
        "title": "十二、工程实现与测试证据",
        "bullets": [
            "代码：38 个模块 / 约 2.6 千行，覆盖信号、量表、行为、评估、训练、监测、模型与界面",
            "测试：51 个自动化用例，覆盖处理链、频谱、伪迹、基线、指标、量表、行为、评估、预警、趋势、训练、模型、端到端",
            "模型：逻辑回归基线，6 维频带特征，按被试 6:2:2 划分，输出准确率与 AUC，JSON 版本化保存",
            "一键复现：scripts/verify.ps1 跑测试并生成报告、热力图、趋势与模型文件",
        ],
        "notes": "全部承诺都有可运行证据；模型为基线模型，深度模型列入下一阶段，验证结论以实测数据为限。",
    },
    {
        "title": "十三、应用对象与应用场景",
        "bullets": [
            "主要对象：需要提升专注力的学生与职场人群；具备评估需求的个人用户",
            "次要对象：学校心理教师与辅导员、企业与机构的员工关怀负责人、认知研究人员",
            "场景一：考前与任务前的状态自查与调节（约 10 分钟）",
            "场景二：按周的专注力训练周期，形成周/月趋势",
            "场景三：校园与机构团体活动；场景四：认知负荷与疲劳的观察性研究",
        ],
        "notes": "对次要对象只输出群体趋势，不涉及个体诊断与个人评价。",
    },
    {
        "title": "十四、边界声明与迭代方向",
        "bullets": [
            "边界：不提供医疗诊断；量表结果只作提示；不得用于处罚或自动上岗决策",
            "边界：真实设备链路与效度验证仍需扩大样本，本轮已完成 BSense-R 实机演示与联调",
            "迭代一：真实设备多品牌适配与实测验收",
            "迭代二：扩大样本，验证重复测量信度与跨天稳定性",
            "迭代三：压力与情绪标签采集与模型训练；深度模型替换基线模型",
        ],
        "notes": "结尾：系统的可信度随实测数据积累提升。主动说明适用边界，是这套系统的基本立场。",
    },
    {
        "title": "十五、团队与分工：三条工作线与协作节奏",
        "bullets": [
            "团队 4 人：队长张嘉辉（项目负责人），队员李骞恒、黄柯语、叶瑞洋；"
            "按信号与算法 / 产品与开发 / 评估与交付三条工作线分工",
            "每周一次信号复核；每阶段末一次交付评审，逐项核对文档、代码与产物是否一致",
        ],
        "image": "fig_team_photos.png",
        "layout": "wide",
        "bullets_h": 1.45,
        "bullet_size": 17,
        "notes": "本页为团队赛期内的现场照片：左图为设备接入与会话流程的联调现场，右图为文档、代码与报告口径的方案讨论。团队协作以每周信号复核与阶段末交付评审两条固定节奏推进，这是各项口径能在文档、代码与报告之间保持一致的原因。",
    },
    {
        "title": "十六、进度与关键路径：16 周甘特图",
        "bullets": [
            "P1 W1–W3 硬件接入与采集 → P2 W4–W6 预处理与特征 → P3 W7–W9 指标与模型 → "
            "P4 W10–W13 训练与可视化 → P5 W14–W16 验证与交付",
            "关键路径为 P2 的伪迹处理与 P4 的训练闭环；贯穿任务为文档口径维护、每周信号复核与真实数据采集",
            "五个里程碑各有可核对的判据：设备接入与完整会话录制 → 处理链回归用例 → 指标与评估可复算 → "
            "训练闭环与界面可用 → 51 个用例通过且五项材料齐备",
        ],
        "image": "fig_gantt.png",
        "layout": "wide",
        "bullets_h": 1.95,
        "bullet_size": 16,
        "notes": "若设备到货延迟，P1 可先用仿真信号推进 P2 的算法开发，但 P2 的验收必须使用真实设备数据；甘特图下方已注明该条件，排期按可核对的里程碑表述。",
    },
    {
        "title": "十七、界面一览：Web 主界面四个视图",
        "bullets": [
            "概览（状态指标）· 实时监测（波形、频谱与质量分档）· 训练中心（专注度与分段记录）· 报告与产物（结论与证据回填）",
        ],
        "image": "fig_ui_row.png",
        "layout": "wide",
        "bullets_h": 0.95,
        "bullet_size": 17,
        "notes": "四个视图对应现场演示的操作路径：从状态指标到实时监测，再到训练与报告，一步不断链；每个视图上的数字都能在报告与质量记录里找到来源。",
    },
]
def collect_demo_data():
    import sys
    src = ROOT / "ningsi" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from ningsi.app.pipeline import SessionConfig, SessionRunner
    from ningsi.monitoring import history

    runner = SessionRunner(SessionConfig(
        root=str(ROOT / "var" / "ppt_demo"), participant="p01",
        task_plan=(("rest", 6), ("focused", 12), ("drowsy", 12), ("loaded", 8)),
    ))
    baselines = runner.collect_baselines()
    task = runner.run_task(baselines["eyes_open"])
    records = history.load_records(ROOT / "ningsi" / "var" / "verify")
    usable, _ = history.comparable(records, "sim-bsense", 250.0, 1)
    points = history.aggregate(usable, "focus", "week")
    return task["indicators"], task["series"], points
def write_speaker_notes() -> Path:
    lines = ["# 项目简介 PPT 逐页讲稿", "", "> 与 A09-凝思-项目简介.pptx 一一对应；每页建议时长按 10–12 分钟汇报排练。", ""]
    for index, spec in enumerate(SLIDES, start=1):
        lines.append(f"## 第 {index} 页 · {spec['title']}")
        if spec.get("subtitle"):
            lines.append("")
            lines.append(spec["subtitle"].replace("\n", " ／ "))
        lines.append("")
        for bullet in spec.get("bullets", []):
            lines.append(f"- {bullet}")
        lines.append("")
        lines.append(f"**讲稿**：{spec.get('notes', '')}")
        lines.append("")
    target = OUT / "PPT逐页讲稿.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    return target
def _cn_number(text: str) -> int:
    digits = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    if text == "十":
        return 10
    if text.startswith("十"):
        return 10 + digits.get(text[1:], 0)
    if "十" in text:
        head, _, tail = text.partition("十")
        return digits.get(head, 0) * 10 + (digits.get(tail, 0) if tail else 0)
    return digits.get(text, 0)


def split_title(title: str):
    """“四、产品架构：四层 + 数据安全横条” → (“04”, “产品架构”, “四层 + 数据安全横条”)。"""
    if "、" not in title:
        return "", "", title
    prefix, rest = title.split("、", 1)
    number = f"{_cn_number(prefix):02d}"
    section, _, heading = rest.partition("：")
    if heading:
        return number, section, heading
    return number, "", rest


def section_accent(index: int) -> str:
    """按章节分组切换强调色：同一章内保持一致，四组色对应 Okabe–Ito 的四个色相。"""
    if index <= 4:
        return NAVY          # 背景 · 目标 · 创意
    if index <= 8:
        return TEAL          # 架构 · 流程 · 技术 · 指标
    if index <= 12:
        return BLUE          # 质量 · 评估 · 训练 · 可视化
    if index <= 14:
        return VERM          # 工程 · 场景 · 边界
    return {15: NAVY, 16: TEAL, 17: BLUE}.get(index, VERM)   # 团队 · 进度 · 界面


def split_bullet(text: str):
    """把“标签：说明”拆成标签与说明，标签用强调色加粗，正文保持墨色。"""
    for mark in ("：", ":"):
        head, sep, tail = text.partition(mark)
        if sep and 0 < len(head) <= 12 and tail:
            return head + sep, tail
    return "", text


FIGURE_CAPTIONS = {
    "fig_architecture.png": "图 1　凝思产品架构（四层 + 数据安全横条）",
    "fig_flow.png": "图 2　一次会话九步流程",
    "fig_indicators.png": "图 3　三个可解释指标（示例会话）",
    "ui_live.png": "图 4　实时监测与质量分档（实机界面）",
    "ui_history.png": "图 5　历史记录与趋势分析（实机界面）",
    "fig_team_photos.png": "图 6　团队协作过程记录（设备联调与方案讨论现场）",
    "fig_gantt.png": "图 7　项目主计划甘特图（16 周，含关键路径与里程碑）",
    "fig_ui_grid.png": "图 8　Web 主界面四个视图",
    "fig_ui_row.png": "图 8　Web 主界面四个视图",
}

MARGIN = 0.72
CONTENT_WIDTH = round(13.333 - 2 * MARGIN, 3)
CONTENT_TOP, CONTENT_BOTTOM = 1.80, 6.80
RIGHT_EDGE = round(13.333 - MARGIN, 3)
COVER_KICKER = "A09 ｜ 项目简介"
CAPTION_HEIGHT = 0.24


def hex_of(color):
    """把 (r, g, b) 或 '#RRGGBB' 统一成 '#RRGGBB'。"""
    if isinstance(color, tuple):
        return "#%02X%02X%02X" % color
    return color


def style_run(run, size, color, bold=False, spacing=None):
    color = hex_of(color)
    face = run.font
    face.size = Pt(size)
    face.bold = bold
    face.name = PPT_FONT
    face.color.rgb = RGBColor.from_string(color.lstrip("#"))
    properties = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        element = properties.find(qn(tag))
        if element is None:
            element = parse_xml('<%s xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
                                'typeface="%s"/>' % (tag, PPT_FONT))
            properties.append(element)
        element.set("typeface", PPT_FONT)
    if spacing:
        properties.set("spc", str(int(spacing * 100)))


def add_text(slide, x, y, width, height, text, size, color, bold=False,
             anchor=MSO_ANCHOR.TOP, align=PP_ALIGN.LEFT, spacing=None):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(width), Inches(height))
    frame = box.text_frame
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
    frame.vertical_anchor = anchor
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    run = paragraph.add_run()
    run.text = text
    style_run(run, size, color, bold, spacing)
    return frame


@lru_cache(maxsize=None)
def line_factor():
    """微软雅黑的行高系数：PPT 的倍数行距乘的是字体行高而非字号。"""
    ascent, descent = font(100).getmetrics()
    return (ascent + descent) / 100.0


def text_lines(text, size, width_in):
    """用与 PPT 同一字体（微软雅黑）的真实字宽估算换行后的行数。"""
    measure = font(100)
    limit = width_in * 72.0 / size * 100.0
    lines, current = 1, 0.0
    for char in text:
        advance = measure.getlength(char)
        if current > 0 and current + advance > limit:
            lines, current = lines + 1, advance
        else:
            current += advance
    return lines


def add_bullets(slide, x, y, width, height, bullets, max_size, accent):
    """按可用高度自动选字号；标签加粗着色，悬挂缩进对齐正文。"""
    for size in range(max_size, 14, -1):
        leading = 1.44 if size >= 19 else 1.36
        gap = round(size)
        rows = sum(text_lines(text, size, width - 0.26) for text in bullets)
        needed = (rows * 1.06 * size * leading * line_factor() + (len(bullets) - 1) * gap) / 72.0
        if needed <= height:
            break
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(width), Inches(height))
    frame = box.text_frame
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    for index, text in enumerate(bullets):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.line_spacing = leading
        paragraph.space_after = Pt(gap)
        properties = paragraph._p.get_or_add_pPr()
        properties.set("marL", str(int(0.24 * 914400)))
        properties.set("indent", str(-int(0.24 * 914400)))
        marker = paragraph.add_run()
        marker.text = "▪  "
        style_run(marker, size * 0.82, accent, True)
        lead, body = split_bullet(text)
        if lead:
            head = paragraph.add_run()
            head.text = lead + " "
            style_run(head, size, accent, True)
        content = paragraph.add_run()
        content.text = body
        style_run(content, size, INK)
    return frame


def add_rule(slide, y, width=CONTENT_WIDTH, x=MARGIN, color=HAIR, thickness=0.012):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(width), Inches(thickness))
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor.from_string(hex_of(color).lstrip("#"))
    shape.line.fill.background()
    shape.shadow.inherit = False
    return shape


def add_figure(slide, path, x, y, width, pad=0.10):
    """配图外套一张细边框白卡，与正文留白对齐；图内不重复标题。"""
    with Image.open(path) as image:
        height = width * image.height / image.width
    board = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x - pad), Inches(y - pad),
                                   Inches(width + 2 * pad), Inches(height + 2 * pad))
    board.fill.solid()
    board.fill.fore_color.rgb = RGBColor.from_string(WHITE.lstrip("#"))
    board.line.color.rgb = RGBColor.from_string(HAIR.lstrip("#"))
    board.line.width = Pt(0.75)
    board.shadow.inherit = False
    board.adjustments[0] = 0.02
    slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(width))
    return height


def add_caption(slide, x, y, width, name, accent):
    """图注重建“图 N”编号：编号用章节强调色，说明用灰阶。"""
    number, _, text = FIGURE_CAPTIONS[name].partition("　")
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(width), Inches(CAPTION_HEIGHT))
    frame = box.text_frame
    frame.word_wrap = False
    frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
    paragraph = frame.paragraphs[0]
    head = paragraph.add_run()
    head.text = number + "　"
    style_run(head, 10, accent, True)
    body = paragraph.add_run()
    body.text = text
    style_run(body, 10, GREY)
    return frame


def figure_height(path, width):
    with Image.open(path) as image:
        return width * image.height / image.width


def cover_slide(prs, spec, paths):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.shapes.add_picture(str(paths["fig_cover.png"]), 0, 0, width=Inches(13.333), height=Inches(7.5))
    add_text(slide, 1.0, 1.40, 6.4, 0.32, COVER_KICKER, 13, SKY, True, spacing=2.4)
    add_text(slide, 1.0, 1.82, 7.4, 1.05, spec["title"], 56, WHITE, True)
    lines = spec["subtitle"].split("\n")
    add_text(slide, 1.0, 3.16, 7.4, 0.42, lines[0], 21, "#B9C7DA")
    if len(lines) > 1:
        add_text(slide, 1.0, 3.62, 7.4, 0.42, lines[1], 21, AMBER, True)
    add_rule(slide, 4.30, x=1.03, width=1.5, color=AMBER, thickness=0.045)
    add_rule(slide, 5.34, x=1.0, width=5.9, color="#2B4570", thickness=0.01)
    for index, line in enumerate(spec.get("bullets", [])):
        add_text(slide, 1.0, 5.52 + index * 0.38, 9.6, 0.32, line, 14, "#B4C2D6")
    return slide


def content_slide(prs, spec, index, total, paths):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    number, section, heading = split_title(spec["title"])
    accent = section_accent(index)
    images = spec.get("images") or ([spec["image"]] if spec.get("image") else [])
    add_rule(slide, 0, width=13.333, x=0, color=accent, thickness=0.055)
    if not images:
        add_text(slide, 8.55, 2.9, 4.1, 3.6, number, 200, mix(accent, WHITE, 0.9), True,
                 anchor=MSO_ANCHOR.BOTTOM, align=PP_ALIGN.RIGHT)
    add_text(slide, MARGIN, 0.46, 8.0, 0.3, f"{number} — {section}" if section else number,
             13, GREY, True, spacing=1.6)
    add_text(slide, MARGIN, 0.78, 11.6, 0.62, heading, 30, INK, True)
    add_rule(slide, 1.44)
    add_rule(slide, 1.432, width=1.35, color=accent, thickness=0.04)
    bullets = spec.get("bullets", [])
    if len(images) == 2:
        add_bullets(slide, MARGIN, 1.70, CONTENT_WIDTH, 1.60, bullets, 18, accent)
        width = 5.4
        for position, name in enumerate(images):
            x = MARGIN + position * (width + 1.09)
            add_figure(slide, paths[name], x, 3.34, width)
            add_caption(slide, x, 3.34 + figure_height(paths[name], width) + 0.10, width, name, accent)
    elif images:
        name = images[0]
        if spec.get("layout") == "wide":
            # 通栏配图（甘特图 / 界面一览 / 团队照片）：要点在上，图按可用高度等比放大
            bullets_h = spec.get("bullets_h", 1.20)
            add_bullets(slide, MARGIN, 1.62, CONTENT_WIDTH, bullets_h, bullets,
                        spec.get("bullet_size", 17), accent)
            top = 1.62 + bullets_h + 0.10
            avail_h = CONTENT_BOTTOM - top - CAPTION_HEIGHT - 0.10
            with Image.open(paths[name]) as image:
                ratio = image.width / image.height
            width = min(CONTENT_WIDTH, avail_h * ratio)
            height = width / ratio
            x = round((13.333 - width) / 2, 3)
            add_figure(slide, paths[name], x, top, width)
            add_caption(slide, x, top + height + 0.10, width, name, accent)
        else:
            add_bullets(slide, MARGIN, CONTENT_TOP, 5.95, CONTENT_BOTTOM - CONTENT_TOP, bullets, 17, accent)
            width = 5.55
            height = figure_height(paths[name], width)
            top = CONTENT_TOP + (CONTENT_BOTTOM - CONTENT_TOP - height - CAPTION_HEIGHT - 0.08) / 2
            add_figure(slide, paths[name], RIGHT_EDGE - width, top, width)
            add_caption(slide, RIGHT_EDGE - width, top + height + 0.10, width, name, accent)
    else:
        add_bullets(slide, MARGIN, CONTENT_TOP, 8.05, CONTENT_BOTTOM - CONTENT_TOP, bullets, 22, accent)
    add_rule(slide, 6.98, thickness=0.01)
    add_text(slide, MARGIN, 7.06, 7.0, 0.28, "凝思 Ningsi ｜ A09 ｜ 项目简介", 9.5, GREY)
    add_text(slide, RIGHT_EDGE - 2.5, 7.06, 2.5, 0.28, f"{index:02d} / {total}", 9.5, GREY, align=PP_ALIGN.RIGHT)
    return slide


def build_pptx(paths) -> Path:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    total = len(SLIDES)
    for index, spec in enumerate(SLIDES, start=1):
        slide = cover_slide(prs, spec, paths) if index == 1 else content_slide(prs, spec, index, total, paths)
        slide.notes_slide.notes_text_frame.text = spec.get("notes", "")
    target = OUT / "A09-凝思-项目简介.pptx"
    prs.save(target)
    return target


def main() -> int:
    import sys
    sys.path.insert(0, str(ROOT / "ningsi" / "src"))
    summary, series, points = collect_demo_data()
    ui = FIGS / "ui"
    paths = {
        "fig_cover.png": draw_cover(),
        "fig_architecture.png": draw_architecture(),
        "fig_flow.png": draw_flow(),
        "fig_indicators.png": draw_indicators(summary),
        "fig_heatmap.png": draw_heatmap(series),
        "fig_trend.png": draw_trend(points),
        # 实机界面截图与新增图示（由 _analysis/prep_ui_assets.py 与
        # _analysis/fig_*.py 生成；重跑本脚本不会覆盖它们）
        "ui_live.png": ui / "ui_live.png",
        "ui_history.png": ui / "ui_history.png",
        "fig_team_photos.png": FIGS / "fig_team_photos.png",
        "fig_ui_grid.png": FIGS / "fig_ui_grid.png",
        "fig_ui_row.png": FIGS / "fig_ui_row.png",
        "fig_gantt.png": ROOT / "deliverables" / "figures_v2" / "fig_gantt.png",
    }
    missing = [str(p) for p in paths.values() if not Path(p).exists()]
    if missing:
        raise SystemExit("缺少素材，请先生成：\n  " + "\n  ".join(missing))
    for name, path in paths.items():
        print(f"  配图：{path} （{round(os.path.getsize(path) / 1024)} KB）")
    pptx = build_pptx(paths)
    print(f"  PPT：{pptx}")
    notes = write_speaker_notes()
    print(f"  讲稿：{notes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
