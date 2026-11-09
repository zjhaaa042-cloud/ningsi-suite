
import zipfile, io, os, sys
sys.path.insert(0, r'D:\Desktop\ninsi\ningsi\tools')
from PIL import Image, ImageDraw, ImageFont
import relayout_figures as R

FONTS = {'Bold': r'C:\Windows\Fonts\msyhbd.ttc', 'Regular': r'C:\Windows\Fonts\msyh.ttc'}

def clean_rows_background(header, ts, te):
    """用文字区上方与下方的"干净行"逐列取色，再按纵向线性插值，得到 (x, y) -> 底色。"""
    w, h = header.size
    top_rows = list(range(max(0, 1), max(1, ts - 3)))
    bottom_rows = list(range(min(h - 1, te + 3), h))
    if not top_rows: top_rows = [0]
    if not bottom_rows: bottom_rows = [h - 1]
    def column_mean(rows, x):
        return [sum(header.getpixel((x, y))[c] for y in rows) / len(rows) for c in range(3)]
    top = [column_mean(top_rows, x) for x in range(w)]
    bottom = [column_mean(bottom_rows, x) for x in range(w)]
    y_top = sum(top_rows) / len(top_rows)
    y_bottom = sum(bottom_rows) / len(bottom_rows)
    span = max(1.0, y_bottom - y_top)
    def colour(x, y):
        ratio = min(1.4, max(-0.4, (y - y_top) / span))
        return tuple(int(round(top[x][c] + (bottom[x][c] - top[x][c]) * ratio)) for c in range(3))
    return colour

def rebuild(header, title_text, gap, font_key, tracking=0.0):
    counts, _, _, _ = R.ink_metrics(header)
    ink_rows = [y for y, c in enumerate(counts) if c > 4]
    ts, te = ink_rows[0], ink_rows[-1]
    sub_rows = max(12, round(0.35 * (te - ts)))
    gap_end = te - sub_rows
    w, h = header.size
    colour = clean_rows_background(header, ts, te)
    H = h + gap
    out = Image.new('RGB', (w, H))
    for y in range(H):
        src = y if y <= gap_end else (y - gap if y - gap <= te else gap_end)
        for x in range(w):
            out.putpixel((x, y), colour(x, src))
    for y in range(gap_end + 1, te + 1):
        target_y = y + gap
        for x in range(w):
            orig = header.getpixel((x, y))[:3]
            base = colour(x, y)
            new_base = colour(x, gap_end)
            out.putpixel((x, target_y), tuple(max(0, min(255, int(round(new_base[c] + (orig[c] - base[c]))))) for c in range(3)))
    target_h = gap_end - ts + 1
    left_x = R.ink_left_x(header, range(ts, gap_end + 1), max(1, w // 240))
    size = target_h
    font = ImageFont.truetype(FONTS[font_key], size)
    for _ in range(5):
        scratch = Image.new('L', (w, 240), 0); d = ImageDraw.Draw(scratch)
        if tracking:
            x = 4
            for ch in title_text:
                d.text((x, 4), ch, font=font, fill=255); x += d.textlength(ch, font=font) + tracking
        else:
            d.text((4, 4), title_text, font=font, fill=255)
        box = scratch.getbbox()
        ink_h = box[3] - box[1]
        if abs(ink_h - target_h) <= 1: break
        size = max(8, round(font.size * target_h / max(1, ink_h)))
        font = ImageFont.truetype(FONTS[font_key], size)
    layer = Image.new('RGBA', (w, H), (0, 0, 0, 0)); dl = ImageDraw.Draw(layer)
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
tiles = [header,
         rebuild(header, '图 3 特色总结图', 24, 'Bold', 0.0),
         rebuild(header, '图 3 特色总结图', 28, 'Bold', 0.6),
         rebuild(header, '图 3 特色总结图', 24, 'Regular', 0.0)]
sheet = Image.new('RGB', (header.size[0] * 2, sum(t.size[1] for t in tiles) * 2 + 16 * len(tiles)), (255, 255, 255))
y = 0
for tile in tiles:
    big = tile.resize((tile.size[0] * 2, tile.size[1] * 2), Image.LANCZOS)
    sheet.paste(big, (0, y)); y += big.size[1] + 16
sheet.save(r'D:\Desktop\ninsi\_analysis\variants2.png')
print('saved', sheet.size)
