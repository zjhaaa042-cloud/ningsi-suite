
import zipfile, io, os, sys
sys.path.insert(0, r'D:\Desktop\ninsi\ningsi\tools')
from PIL import Image, ImageDraw, ImageFont
import relayout_figures as R

FONTS = {
    'Bold': r'C:\Windows\Fonts\msyhbd.ttc',
    'Regular': r'C:\Windows\Fonts\msyh.ttc',
    'SimHei': r'C:\Windows\Fonts\simhei.ttf',
}

def band_background(header, edge=10, smooth=3):
    """逐行用左右边缘像素线性拟合底色，并做纵向平滑，返回 (x, y) -> color。"""
    w, h = header.size
    rows = []
    for y in range(h):
        left = [header.getpixel((x, y))[:3] for x in range(2, edge)]
        right = [header.getpixel((x, y))[:3] for x in range(w - edge, w - 2)]
        def mean(samples):
            n = len(samples)
            return [sum(s[i] for s in samples) / n for i in range(3)]
        rows.append((mean(left), mean(right)))
    # 纵向平滑
    smoothed = []
    for y in range(h):
        lo, hi = max(0, y - smooth), min(h - 1, y + smooth)
        left = [sum(rows[i][0][c] for i in range(lo, hi + 1)) / (hi - lo + 1) for c in range(3)]
        right = [sum(rows[i][1][c] for i in range(lo, hi + 1)) / (hi - lo + 1) for c in range(3)]
        smoothed.append((left, right))
    def colour(x, y):
        yy = min(max(y, 0), h - 1)
        left, right = smoothed[yy]
        ratio = x / max(1, w - 1)
        return tuple(int(round(left[c] + (right[c] - left[c]) * ratio)) for c in range(3))
    return colour

def band_row_background(header, y, edge=10):
    w = header.size[0]
    left = [header.getpixel((x, y))[:3] for x in range(2, edge)]
    right = [header.getpixel((x, y))[:3] for x in range(w - edge, w - 2)]
    def mean(samples):
        n = len(samples)
        return [sum(s[i] for s in samples) / n for i in range(3)]
    return mean(left), mean(right)

def rebuild(header, title_text, gap, font_key, size_scale=1.0, tracking=0.0):
    counts, _, _, _ = R.ink_metrics(header)
    ink_rows = [y for y, c in enumerate(counts) if c > 4]
    ts, te = ink_rows[0], ink_rows[-1]
    sub_rows = max(12, round(0.35 * (te - ts)))
    gap_end = te - sub_rows
    w, h = header.size
    colour = band_background(header)
    H = h + gap
    out = Image.new('RGB', (w, H))
    for y in range(H):
        src = y if y <= gap_end else gap_end
        for x in range(w):
            out.putpixel((x, y), colour(x, src))
    # 副标题：用"原像素 - 原行底色"的差值，叠加到新底色上（保留抗锯齿）
    for y in range(gap_end + 1, te + 1):
        left, right = band_row_background(header, y)
        target_y = y + gap
        for x in range(w):
            orig = header.getpixel((x, y))[:3]
            ratio = x / max(1, w - 1)
            base = [left[c] + (right[c] - left[c]) * ratio for c in range(3)]
            new_base = colour(x, gap_end)
            value = tuple(max(0, min(255, int(round(new_base[c] + (orig[c] - base[c]))))) for c in range(3))
            out.putpixel((x, target_y), value)
    # 标题重绘
    font = ImageFont.truetype(FONTS[font_key], max(8, round((gap_end - ts + 1) * size_scale * 1.32)))
    scratch = Image.new('L', (w, 200), 0)
    d = ImageDraw.Draw(scratch)
    if tracking:
        x = 4
        for ch in title_text:
            d.text((x, 4), ch, font=font, fill=255)
            x += d.textlength(ch, font=font) + tracking
        box = scratch.getbbox()
    else:
        d.text((4, 4), title_text, font=font, fill=255)
        box = scratch.getbbox()
    ink_h = box[3] - box[1]
    target_h = gap_end - ts + 1
    # 高度校正
    for _ in range(4):
        if abs(ink_h - target_h) <= 1:
            break
        scale = target_h / max(1, ink_h)
        size = max(8, round(font.size * scale))
        font = ImageFont.truetype(FONTS[font_key], size)
        scratch = Image.new('L', (w, 200), 0); d = ImageDraw.Draw(scratch)
        if tracking:
            x = 4
            for ch in title_text:
                d.text((x, 4), ch, font=font, fill=255); x += d.textlength(ch, font=font) + tracking
            box = scratch.getbbox()
        else:
            d.text((4, 4), title_text, font=font, fill=255); box = scratch.getbbox()
        ink_h = box[3] - box[1]
    left_x = R.ink_left_x(header, range(ts, gap_end + 1), max(1, w // 240))
    layer = Image.new('RGBA', (w, H), (0, 0, 0, 0))
    dl = ImageDraw.Draw(layer)
    centre = (ts + gap_end) / 2 + 2
    if tracking:
        x = left_x - box[0]
        for ch in title_text:
            dl.text((x, centre - ink_h / 2 - box[1]), ch, font=font, fill=(255, 255, 255, 255))
            x += dl.textlength(ch, font=font) + tracking
    else:
        dl.text((left_x - box[0], centre - ink_h / 2 - box[1]), title_text, font=font, fill=(255, 255, 255, 255))
    out.paste(layer, (0, 0), layer)
    return out

src = zipfile.ZipFile(r'D:\Desktop\ninsi\A09-凝思-项目概要介绍-最终版.docx')
img = Image.open(io.BytesIO(src.read('word/media/image4.png'))).convert('RGB')
band = R.detect_header_band(img)
header = img.crop((0, 0, img.size[0], band))
variants = [
    ('A Bold gap24', dict(gap=24, font_key='Bold', size_scale=1.0, tracking=0.0)),
    ('B Bold gap30 track1', dict(gap=30, font_key='Bold', size_scale=1.0, tracking=1.0)),
    ('C Regular gap24', dict(gap=24, font_key='Regular', size_scale=1.0, tracking=0.0)),
]
tiles = [header] + [rebuild(header, '图 3 特色总结图', **kwargs) for _, kwargs in variants]
scale = 2
sheet = Image.new('RGB', (header.size[0] * scale, sum(t.size[1] for t in tiles) * scale + 20 * len(tiles)), (255, 255, 255))
y = 0
for tile in tiles:
    big = tile.resize((tile.size[0] * scale, tile.size[1] * scale), Image.LANCZOS)
    sheet.paste(big, (0, y)); y += big.size[1] + 20
sheet.save(r'D:\Desktop\ninsi\_analysis\variants.png')
print('variants saved', sheet.size, [name for name, _ in variants])
