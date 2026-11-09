"""优化交付 docx 内所有插图的版式：给标题带补内边距、拉开标题与副标题行距、整图加白色外边距。

与上一版的区别：标题带往往是渐变底色，因此不再用"单一底色填充"，而是
**复制边行/边列像素**来扩展标题带（渐变、纹理都能自然延续），行墨迹则以"该行中位色"为参照，
因此在渐变带上也能稳定区分文字行与空白行。

处理步骤：
1) 检测顶部标题带（连续"非白为主"的行）；
2) 在带内拉开标题行与副标题行的距离（不足 min_gap 时在其间复制一行带底像素）；
3) 标题带上下左右补内边距（复制边行/边列）；
4) 整幅图加白色外边距，内容不再贴边；
5) 用补白还原原始宽高比，避免 Word 按原框拉伸变形；
6) 用优化后的图片替换 docx 包内 word/media/*，其余包内容原样保留。
"""

from __future__ import annotations

import io
import shutil
import time
import zipfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DELIVERABLES = ROOT / "deliverables"
BACKUP = ROOT / "_analysis" / "backup"

WHITE_LEVEL = 214          # 非白判定：任一通道低于该值即视为"有色"
BAND_NONWHITE_RATIO = 0.70
OUTER_MARGIN_RATIO = 0.018
MIN_OUTER_MARGIN = 12
HEADER_PAD_RATIO = 0.045
MIN_HEADER_PAD = 24
HEADER_PAD_Y_RATIO = 0.42
MIN_TEXT_GAP_RATIO = 0.026
MIN_TEXT_GAP = 26
SUBTITLE_ROWS_RATIO = 0.35   # 副标题文字高度占"标题+副标题"总高度的比例（同模板稳定）
TITLE_TRACKING = 0.6         # 标题字距微调（像素），贴近原设计稿的疏朗排版
INK_TOLERANCE = 60
STRONG_INK = 140          # 强墨迹阈值：低于此值视为抗锯齿淡边，在空隙边界处抹平


def is_white(pixel) -> bool:
    return min(pixel[:3]) >= WHITE_LEVEL


def has_white_frame(image: Image.Image) -> bool:
    """判断图片四边是否已经是白边（用于避免重复优化时再次叠加外边距）。"""
    width, height = image.size
    step = max(1, width // 60)
    top = [image.getpixel((x, 0))[:3] for x in range(0, width, step)]
    bottom = [image.getpixel((x, height - 1))[:3] for x in range(0, width, step)]
    left = [image.getpixel((0, y))[:3] for y in range(0, height, max(1, height // 60))]
    right = [image.getpixel((width - 1, y))[:3] for y in range(0, height, max(1, height // 60))]
    edges = top + bottom + left + right
    white = sum(1 for pixel in edges if is_white(pixel))
    return white / len(edges) > 0.96


def detect_header_band(image: Image.Image) -> int:
    """返回标题带高度（连续"非白为主"的行数）；找不到返回 0。"""
    width, height = image.size
    step = max(1, width // 200)
    limit = min(height, max(24, int(height * 0.25)))
    band = 0
    for y in range(limit):
        row = [image.getpixel((x, y))[:3] for x in range(0, width, step)]
        non_white = sum(1 for pixel in row if not is_white(pixel))
        if non_white / len(row) < BAND_NONWHITE_RATIO:
            break
        band = y + 1
    return band if band >= 12 else 0


def row_reference(image: Image.Image, y: int, step: int):
    """以该行采样像素的中位色作为底色参照，适配渐变带。"""
    values = [image.getpixel((x, y))[:3] for x in range(0, image.size[0], step)]
    channels = [sorted(pixel[i] for pixel in values)[len(values) // 2] for i in range(3)]
    return tuple(channels)


def ink_profile(image: Image.Image):
    """逐行统计与"该行底色"差异明显的像素数。"""
    width, height = image.size
    step = max(1, width // 240)
    profile = []
    for y in range(height):
        reference = row_reference(image, y, step)
        count = 0
        for x in range(0, width, step):
            pixel = image.getpixel((x, y))[:3]
            if sum(abs(pixel[i] - reference[i]) for i in range(3)) > INK_TOLERANCE:
                count += 1
        profile.append(count)
    return profile


def ink_metrics(header: Image.Image):
    """逐行返回 (墨迹像素数, 墨迹横向跨度)。标题行较窄、副标题行通常更宽，可据此切分粘连的两行。"""
    width, height = header.size
    step = max(1, width // 240)
    counts, spans, lefts, rights = [], [], [], []
    edge = max(4, step)          # 忽略图片左右边缘的抗锯齿/描边像素，避免被当成"文字"
    for y in range(height):
        reference = row_reference(header, y, step)
        count = 0
        first = None
        last = None
        for x in range(edge, width - edge, step):
            pixel = header.getpixel((x, y))[:3]
            if sum(abs(pixel[i] - reference[i]) for i in range(3)) > INK_TOLERANCE:
                count += 1
                first = x if first is None else first
                last = x
        counts.append(count)
        spans.append((last - first) if (first is not None and last is not None) else 0)
        lefts.append(first)
        rights.append(last)
    return counts, spans, lefts, rights


def widen_gap(header: Image.Image, min_gap: int, title_text: str | None = None) -> Image.Image:
    """把标题行与副标题行之间的距离拉到 min_gap 像素；粘连时在墨迹低谷处切分。"""
    if title_text:
        return rebuild_band(header, title_text, min_gap)

    profile = ink_profile(header)
    rows = [y for y, count in enumerate(profile) if count > 0]
    if not rows:
        return header
    blocks = []
    start = previous = rows[0]
    for value in rows[1:]:
        if value - previous > 2:
            blocks.append((start, previous))
            start = value
        previous = value
    blocks.append((start, previous))

    counts, _, lefts, rights = ink_metrics(header)

    if len(blocks) >= 2:
        gap_end = blocks[0][1]                      # 两行之间本来就有空白行
    else:
        # 两行粘连：这批插图由同一模板生成，副标题文字高度稳定（约 16 行），
        # 因此从文字块底部往上量固定行数即可准确落在"标题末行"，比密度低谷稳得多。
        ink_rows = [y for y, count in enumerate(counts) if count > 4]
        if len(ink_rows) < 12:
            return header
        text_start, text_end = ink_rows[0], ink_rows[-1]
        subtitle_rows = max(12, round(SUBTITLE_ROWS_RATIO * (text_end - text_start)))
        gap_end = text_end - subtitle_rows
        if gap_end <= text_start + 6 or gap_end >= text_end - 3:
            gap_end = text_start + round(0.64 * (text_end - text_start))

    if gap_end + 1 >= header.size[1]:
        return header
    extra = min_gap
    # 两行紧贴时，切分点前后各有一两行是"副标题的顶边/标题的底边"这种模糊行：
    # 直接保留会在空隙里留下孤立笔画。这里把这几行一并抹成底色（顶边多为抗锯齿淡边，去掉最不显眼），
    # 再补足空隙，因此标题完整、副标题只损失一点顶端淡边。
    trim_above, trim_below = 0, 0    # 不裁剪标题/副标题本身，保留完整字形；零星淡边交给下面清理
    top_end = max(0, gap_end - trim_above)
    bottom_start = min(header.size[1], gap_end + 1 + trim_below)
    filler = extra + (gap_end + 1 + trim_below) - top_end
    strip = background_strip(header, gap_end, filler)
    out = Image.new("RGB", (header.size[0], top_end + filler + (header.size[1] - bottom_start)))
    out.paste(header.crop((0, 0, header.size[0], top_end)), (0, 0))
    out.paste(strip, (0, top_end))
    out.paste(header.crop((0, bottom_start, header.size[0], header.size[1])), (0, top_end + filler))
    return out    # 切分点已按模板量准，不再做边界抹边（抹边反而会把标题底部笔画削成孤立小点）


def _has_support(image: Image.Image, x: int, rows, step: int, tolerance: int = INK_TOLERANCE) -> bool:
    for y in rows:
        if not 0 <= y < image.size[1]:
            continue
        reference = row_reference(image, y, step)
        for candidate in (x - step, x, x + step):
            if not 0 <= candidate < image.size[0]:
                continue
            pixel = image.getpixel((candidate, y))[:3]
            if sum(abs(pixel[i] - reference[i]) for i in range(3)) > tolerance:
                return True
    return False


def clean_boundary_rows(image: Image.Image, filler_top: int, filler_height: int) -> Image.Image:
    """清除切分处残留的半个字形：只在紧邻空隙的一两行里删除"上下没有笔画支撑"的孤立墨点。

    标题末行与副标题首行在像素上紧贴时，切分难免带上 1–2 行零星笔画（多为抗锯齿淡边）；
    这一步把它们抹成底色，真正属于字形的笔画（有上下支撑）保留。
    """
    width, height = image.size
    step = max(1, width // 240)
    strip_top = filler_top
    strip_bottom = filler_top + filler_height - 1
    targets = [
        (list(range(max(0, filler_top - 2), filler_top)), list(range(max(0, filler_top - 6), filler_top - 1)), strip_top),
        (list(range(strip_bottom + 1, min(height, strip_bottom + 3))),
         list(range(strip_bottom + 3, min(height, strip_bottom + 7))), strip_bottom),
    ]
    for rows, support_rows, strip_row in targets:
        if not rows or not support_rows:
            continue
        reference_fill = [image.getpixel((x, strip_row))[:3] for x in range(width)]
        for y in rows:
            reference = row_reference(image, y, step)
            for x in range(width):
                pixel = image.getpixel((x, y))[:3]
                contrast = sum(abs(pixel[i] - reference[i]) for i in range(3))
                if contrast <= INK_TOLERANCE:
                    continue
                # 弱墨迹（抗锯齿淡边）直接抹掉；强墨迹仅在没有上下支撑时才清
                if contrast < STRONG_INK or not _has_support(image, x, support_rows, step):
                    image.putpixel((x, y), reference_fill[x])
    return image


def row_background_fit(header: Image.Image, y: int, step: int):
    """用该行的"非文字"像素按 x 做线性拟合，得到这一行的底色函数（不受字形笔画干扰）。"""
    reference = row_reference(header, y, step)
    points = []
    edge = max(4, step)
    for x in range(edge, header.size[0] - edge, step):
        pixel = header.getpixel((x, y))[:3]
        if sum(abs(pixel[i] - reference[i]) for i in range(3)) <= INK_TOLERANCE:
            points.append((float(x), pixel))
    if len(points) < 4:
        return None
    count = len(points)
    sx = sum(point[0] for point in points)
    sxx = sum(point[0] ** 2 for point in points)
    denominator = count * sxx - sx * sx
    fits = []
    for channel in range(3):
        sy = sum(point[1][channel] for point in points)
        sxy = sum(point[0] * point[1][channel] for point in points)
        if abs(denominator) < 1e-9:
            fits.append((0.0, sy / count))
        else:
            slope = (count * sxy - sx * sy) / denominator
            fits.append((slope, (sy - slope * sx) / count))
    return fits


def background_strip(header: Image.Image, split: int, height: int) -> Image.Image:
    """在标题行与副标题行之间重建一条纯净底色带。

    做法：取空隙上下两行的"非文字像素"各拟合一条水平渐变（线性），再按行做垂直插值。
    这样既不会把字形笔画拉成竖条纹理（早期逐列向上取色的做法会），也不受文字密度影响。
    """
    width = header.size[0]
    step = max(1, width // 240)
    upper = row_background_fit(header, max(0, split - 1), step) or row_background_fit(header, split, step)
    lower = row_background_fit(header, min(header.size[1] - 1, split + 1), step) or upper
    if upper is None:
        upper = lower
    strip = Image.new("RGB", (width, height))
    target = strip.load()
    samples = list(range(0, width, step)) + [width - 1]
    for x in samples:
        top = tuple(upper[channel][0] * x + upper[channel][1] for channel in range(3))
        bottom = tuple(lower[channel][0] * x + lower[channel][1] for channel in range(3))
        for dy in range(height):
            ratio = (dy + 0.5) / height
            color = tuple(
                int(max(0, min(255, round(top[channel] + (bottom[channel] - top[channel]) * ratio))))
                for channel in range(3)
            )
            target[x, dy] = color
        if step > 1 and x + step <= width - 1:
            for offset in range(1, step):
                target[x + offset, 0] = target[x, 0]
    # 用最近采样列填充中间列，保证整条底色平滑
    for x in range(width):
        if x % step == 0 or x == width - 1:
            continue
        left = (x // step) * step
        right = min(width - 1, left + step)
        for dy in range(height):
            a = target[left, dy]
            b = target[right, dy]
            ratio = (x - left) / max(1, right - left)
            target[x, dy] = tuple(int(round(a[channel] + (b[channel] - a[channel]) * ratio)) for channel in range(3))
    return strip


TITLE_FONTS = (r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf")


def _font_path() -> str | None:
    for path in TITLE_FONTS:
        if Path(path).exists():
            return path
    return None


def ink_left_x(header: Image.Image, rows, step: int) -> int:
    """取指定行内墨迹像素 x 坐标的低分位，作为文字左边界（排除边缘描边干扰）。"""
    positions = []
    for y in rows:
        if not 0 <= y < header.size[1]:
            continue
        reference = row_reference(header, y, step)
        for x in range(0, header.size[0], step):
            pixel = header.getpixel((x, y))[:3]
            if sum(abs(pixel[i] - reference[i]) for i in range(3)) > INK_TOLERANCE:
                positions.append(x)
    if not positions:
        return 0
    positions.sort()
    return positions[max(0, int(len(positions) * 0.03))]


def sample_text_colour(header: Image.Image, rows, step: int):
    """取文字像素中最亮的一部分求均值，作为文字颜色。"""
    samples = []
    for y in rows:
        if not 0 <= y < header.size[1]:
            continue
        reference = row_reference(header, y, step)
        for x in range(0, header.size[0], step):
            pixel = header.getpixel((x, y))[:3]
            if sum(abs(pixel[i] - reference[i]) for i in range(3)) > INK_TOLERANCE:
                samples.append(pixel)
    if not samples:
        return (255, 255, 255)
    samples.sort(key=lambda colour: sum(colour), reverse=True)
    top = samples[: max(1, len(samples) // 5)]
    return tuple(sum(colour[i] for colour in top) // len(top) for i in range(3))


def _legacy_paint_background(header: Image.Image, rows, step: int) -> None:
    """用左右边距拟合出的水平渐变覆盖指定行（抹掉原有文字）。"""
    fallback = None
    for y in rows:
        if not 0 <= y < header.size[1]:
            continue
        fit = row_background_fit(header, y, step) or fallback
        if fit is None:
            continue
        fallback = fit
        edge = max(4, step)
        for x in range(edge, header.size[0] - edge):   # 图片左右边缘保持原样，避免渐变外推出现色块
            header.putpixel((x, y), tuple(
                max(0, min(255, round(fit[channel][0] * x + fit[channel][1]))) for channel in range(3)
            ))


def _legacy_draw_title(header: Image.Image, text: str, top: int, bottom: int, left_x: int, colour) -> None:
    """把标题按目标行高与左边界重绘到标题带上（微软雅黑 Bold，尺寸自动匹配）。"""
    path = _font_path()
    if path is None or not text:
        return
    from PIL import ImageDraw, ImageFont

    target_height = bottom - top + 1
    centre = (top + bottom) / 2
    size = target_height
    chosen = None
    for _ in range(5):
        font = ImageFont.truetype(path, size)
        scratch = Image.new("L", (max(64, header.size[0]), target_height * 4), 0)
        draw = ImageDraw.Draw(scratch)
        draw.text((8, 8), text, font=font, fill=255)
        box = scratch.getbbox()
        if box is None:
            return
        height = box[3] - box[1]
        chosen = (font, box, height)
        if abs(height - target_height) <= 1:
            break
        size = max(8, round(size * target_height / max(1, height)))
    if chosen is None:
        return
    font, box, height = chosen
    layer = Image.new("RGBA", header.size, (0, 0, 0, 0))
    ImageDraw.Draw(layer).text((left_x - box[0], centre - height / 2 - box[1]), text, font=font,
                               fill=tuple(colour) + (255,))
    header.paste(layer, (0, 0), layer)


def _legacy_rebuild_title(header: Image.Image, title_text: str, gap_end: int, text_start: int, gap: int) -> Image.Image:
    """抹掉原标题并重绘：标题用文档图题文字，副标题保持原始像素（整体下移）。"""
    step = max(1, header.size[0] // 240)
    counts, _, _, _ = ink_metrics(header)
    ink_rows = [y for y, count in enumerate(counts) if count > 4]
    if len(ink_rows) < 12:
        return header
    text_start, text_end = ink_rows[0], ink_rows[-1]
    subtitle_rows = max(12, round(SUBTITLE_ROWS_RATIO * (text_end - text_start)))
    gap_end = text_end - subtitle_rows
    title_rows = list(range(text_start, gap_end + 1))
    left_x = ink_left_x(header, title_rows, step)
    colour = sample_text_colour(header, title_rows, step)
    work = header.copy()
    paint_background(work, range(max(0, text_start - 4), gap_end + 2), step)
    draw_title(work, title_text, text_start, gap_end, left_x, colour)

    out = Image.new("RGB", (header.size[0], header.size[1] + gap))
    out.paste(work.crop((0, 0, header.size[0], gap_end + 1)), (0, 0))
    filler = work.crop((0, gap_end, header.size[0], gap_end + 1)).resize((header.size[0], gap))
    out.paste(filler, (0, gap_end + 1))
    out.paste(work.crop((0, gap_end + 1, header.size[0], header.size[1])), (0, gap_end + 1 + gap))
    return out


def clean_rows_background(header: Image.Image, text_start: int, text_end: int):
    """用文字区上下两侧的"干净行"逐列取色，再按纵向线性插值，得到 (x, y) -> 底色。

    比"逐行拟合"更稳：不再从文字行内部外推，因此不会出现浅色块或色带。
    """
    width, height = header.size
    top_rows = list(range(1, max(2, text_start - 3)))
    bottom_rows = list(range(min(height - 2, text_end + 3), height - 1))
    if not top_rows:
        top_rows = [0]
    if not bottom_rows:
        bottom_rows = [height - 1]

    def column_mean(rows, x):
        return [sum(header.getpixel((x, y))[channel] for y in rows) / len(rows) for channel in range(3)]

    top = [column_mean(top_rows, x) for x in range(width)]
    bottom = [column_mean(bottom_rows, x) for x in range(width)]
    y_top = sum(top_rows) / len(top_rows)
    y_bottom = sum(bottom_rows) / len(bottom_rows)
    span = max(1.0, y_bottom - y_top)

    def colour(x, y):
        ratio = min(1.3, max(-0.3, (y - y_top) / span))
        return tuple(int(round(top[x][channel] + (bottom[x][channel] - top[x][channel]) * ratio)) for channel in range(3))

    return colour


def rebuild_band(header: Image.Image, title_text: str, gap: int) -> Image.Image:
    """整条标题带重建：底色按上下干净行插值，标题重绘，副标题以"原像素减原底色"的差值贴回。

    不切割任何字形，因此不会留下碎片、条纹或孤立小点；副标题保持原始像素，清晰度与原图一致。
    """
    from PIL import ImageDraw, ImageFont

    width, height = header.size
    counts, _, _, _ = ink_metrics(header)
    ink_rows = [y for y, count in enumerate(counts) if count > 4]
    if len(ink_rows) < 12:
        return header
    text_start, text_end = ink_rows[0], ink_rows[-1]
    subtitle_rows = max(12, round(SUBTITLE_ROWS_RATIO * (text_end - text_start)))
    gap_end = text_end - subtitle_rows
    colour = clean_rows_background(header, text_start, text_end)

    new_height = height + gap
    out = Image.new("RGB", (width, new_height))
    for y in range(new_height):
        source_y = y if y <= gap_end else (y - gap if y - gap <= text_end else gap_end)
        for x in range(width):
            out.putpixel((x, y), colour(x, source_y))

    for y in range(gap_end + 1, text_end + 1):                 # 副标题：差值贴回，保留抗锯齿
        for x in range(width):
            original = header.getpixel((x, y))[:3]
            base = colour(x, y)
            new_base = colour(x, gap_end)
            out.putpixel((x, y + gap), tuple(
                max(0, min(255, int(round(new_base[channel] + (original[channel] - base[channel])))))
                for channel in range(3)
            ))

    target_height = gap_end - text_start + 1
    left_x = ink_left_x(header, range(text_start, gap_end + 1), max(1, width // 240))
    font_path = _font_path()
    if font_path and title_text:
        size = target_height
        font = ImageFont.truetype(font_path, size)
        box = None
        for _ in range(5):
            scratch = Image.new("L", (width, 260), 0)
            draw = ImageDraw.Draw(scratch)
            if TITLE_TRACKING:
                cursor = 4
                for character in title_text:
                    draw.text((cursor, 4), character, font=font, fill=255)
                    cursor += draw.textlength(character, font=font) + TITLE_TRACKING
            else:
                draw.text((4, 4), title_text, font=font, fill=255)
            box = scratch.getbbox()
            if box is None:
                break
            ink_height = box[3] - box[1]
            if abs(ink_height - target_height) <= 1:
                break
            size = max(8, round(font.size * target_height / max(1, ink_height)))
            font = ImageFont.truetype(font_path, size)
        if box is not None:
            ink_height = box[3] - box[1]
            layer = Image.new("RGBA", (width, new_height), (0, 0, 0, 0))
            painter = ImageDraw.Draw(layer)
            centre = (text_start + gap_end) / 2 + 2
            if TITLE_TRACKING:
                cursor = left_x - box[0]
                for character in title_text:
                    painter.text((cursor, centre - ink_height / 2 - box[1]), character, font=font,
                                 fill=(255, 255, 255, 255))
                    cursor += painter.textlength(character, font=font) + TITLE_TRACKING
            else:
                painter.text((left_x - box[0], centre - ink_height / 2 - box[1]), title_text, font=font,
                             fill=(255, 255, 255, 255))
            out.paste(layer, (0, 0), layer)
    return out


def pad_header(header: Image.Image, pad_x: int, pad_y: int) -> Image.Image:
    """标题带补内边距：上下复制边行、左右延展边列，保持渐变连续。"""
    width, height = header.size
    out = Image.new("RGB", (width + 2 * pad_x, height + 2 * pad_y))
    out.paste(header, (pad_x, pad_y))
    if pad_y:
        out.paste(header.crop((0, 0, width, 1)).resize((width, pad_y)), (pad_x, 0))
        out.paste(header.crop((0, height - 1, width, height)).resize((width, pad_y)), (pad_x, pad_y + height))
    if pad_x:
        pixels = out.load()
        for y in range(out.size[1]):
            left = pixels[pad_x, y]
            right = pixels[pad_x + width - 1, y]
            for offset in range(pad_x):
                pixels[offset, y] = left
                pixels[pad_x + width + offset, y] = right
    return out


def optimize_image(image: Image.Image, title_text: str | None = None) -> Image.Image:
    original_size = image.size
    width, height = original_size
    image = image.convert("RGB")
    band_height = detect_header_band(image)
    outer = max(MIN_OUTER_MARGIN, round(min(width, height) * OUTER_MARGIN_RATIO))

    if band_height:
        header = image.crop((0, 0, width, band_height))
        body = image.crop((0, band_height, width, height))
        header = widen_gap(header, max(MIN_TEXT_GAP, round(height * MIN_TEXT_GAP_RATIO)), title_text)
        pad_x = max(MIN_HEADER_PAD, round(width * HEADER_PAD_RATIO))
        pad_y = max(12, round(header.size[1] * HEADER_PAD_Y_RATIO))
        header = pad_header(header, pad_x, pad_y)
    else:
        header = None
        body = image
        pad_x = 0

    inner_width = width + 2 * pad_x
    total_height = (header.size[1] if header else 0) + body.size[1]
    canvas = Image.new("RGB", (inner_width + 2 * outer, total_height + 2 * outer), (255, 255, 255))
    top = outer
    if header is not None:
        canvas.paste(header, (outer, top))
        top += header.size[1]
    canvas.paste(body, (outer + pad_x, top))

    target_ratio = original_size[0] / original_size[1]
    current_ratio = canvas.size[0] / canvas.size[1]
    if current_ratio > target_ratio:
        extra = round(canvas.size[0] / target_ratio) - canvas.size[1]
        grown = Image.new("RGB", (canvas.size[0], canvas.size[1] + extra), (255, 255, 255))
        grown.paste(canvas, (0, 0))
        canvas = grown
    elif current_ratio < target_ratio:
        extra = round(canvas.size[1] * target_ratio) - canvas.size[0]
        grown = Image.new("RGB", (canvas.size[0] + extra, canvas.size[1]), (255, 255, 255))
        grown.paste(canvas, (0, 0))
        canvas = grown
    return canvas


def encode(image: Image.Image, suffix: str) -> bytes:
    buffer = io.BytesIO()
    if suffix.lower() in (".jpg", ".jpeg"):
        image.save(buffer, format="JPEG", quality=95, subsampling=1)
    else:
        image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def caption_map(path: Path) -> dict:
    """从文档中取出"图 X ..."图题，建立 media 文件名 -> 图题 的映射（用于重绘标题）。"""
    from docx import Document
    from docx.oxml.ns import qn

    document = Document(str(path))
    relation_targets = {rid: rel.target_ref for rid, rel in document.part.rels.items()}
    mapping = {}
    pending = None
    for paragraph in document.paragraphs:
        blips = paragraph._p.findall(".//" + qn("a:blip"))
        rids = [blip.get(qn("r:embed")) for blip in blips if blip.get(qn("r:embed"))]
        text = paragraph.text.strip()
        if rids:
            pending = rids[-1]
        elif pending and text.startswith("图"):
            target = relation_targets.get(pending)
            if target:
                mapping["word/" + target.lstrip("/")] = text
            pending = None
    return mapping


def optimize_docx(source: Path, target: Path) -> dict:
    BACKUP.mkdir(parents=True, exist_ok=True)
    if source.exists() and source.resolve() != target.resolve():
        shutil.copy2(source, BACKUP / source.name)
    captions = caption_map(source)
    stats = {"images": 0, "optimized": 0, "skipped": 0, "banded": 0, "titled": 0}
    print(f"    图题识别：{len(captions)} 条")
    with zipfile.ZipFile(source) as archive:
        entries = archive.infolist()
        payloads = {entry.filename: archive.read(entry.filename) for entry in entries}
    for name in list(payloads):
        if not name.startswith("word/media/"):
            continue
        suffix = Path(name).suffix.lower()
        if suffix not in (".png", ".jpg", ".jpeg"):
            continue
        stats["images"] += 1
        try:
            with Image.open(io.BytesIO(payloads[name])) as raw:
                raw.load()
                if max(raw.size) > 1600:            # 整页封面等大图：不做版式改造
                    stats["skipped"] += 1
                    print(f"    跳过 {name}：整页级大图（{raw.size[0]}×{raw.size[1]}），保持原样")
                    continue
                if has_white_frame(raw.convert("RGB")):   # 已优化过：避免重复叠加外边距
                    stats["skipped"] += 1
                    print(f"    跳过 {name}：四边已是白边（疑似已优化），不重复处理")
                    continue
                title_text = captions.get(name)
                if title_text:
                    stats["titled"] += 1
                if detect_header_band(raw.convert("RGB")):
                    stats["banded"] += 1
                optimized = optimize_image(raw, title_text)
            payloads[name] = encode(optimized, suffix)
            stats["optimized"] += 1
        except Exception as error:
            stats["skipped"] += 1
            print(f"    跳过 {name}：{error}")

    for attempt in range(20):
        try:
            if target.exists():
                target.unlink()
            break
        except PermissionError:
            if attempt == 0:
                print(f"    {target.name} 正被占用，等待释放（最多 100 秒）…")
            time.sleep(5)
    else:
        target = target.with_name(f"{target.stem}-插图优化{target.suffix}")
        print(f"    原文件被占用（可能已在 Word 中打开），改写到 {target.name}")
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for entry in entries:
            archive.writestr(entry, payloads[entry.filename])
    return stats


def main() -> int:
    import sys

    # 默认禁用：对作者位图设计稿做像素级改造会引入条纹/碎片，已整体回滚。
    # 只有在明确知道风险、并拿到设计源文件或确认可接受位图位移时，才加 --force 运行。
    if "--force" not in sys.argv:
        print("插图像素处理默认已禁用（上一轮改造产生的条纹/碎片已回滚，文档内插图与原始设计稿逐字节一致）。")
        print("确需运行请显式加 --force，并先备份文档；更安全的做法是提供设计源文件（PPT/Figma/AI）改行距后重新导出。")
        return 0

    names = [a for a in sys.argv[1:] if not a.startswith("--")] or (
        "A09-凝思-项目概要介绍-v2.docx", "A09-凝思-项目详细方案-v2.docx")
    for name in names:
        path = Path(name) if Path(name).is_absolute() else DELIVERABLES / name
        if not path.exists():
            print(f"找不到 {path}")
            continue
        print(f"== {path.name} ==")
        stats = optimize_docx(path, path)
        print(f"   图片 {stats['images']} 张，识别到标题带 {stats['banded']} 张，"
              f"重绘标题 {stats['titled']} 张，优化 {stats['optimized']} 张，跳过 {stats['skipped']} 张"
              f"（原文件已备份到 _analysis/backup）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
