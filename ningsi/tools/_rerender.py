
import zipfile, io, sys
sys.path.insert(0, r'D:\Desktop\ninsi\ningsi\tools')
from PIL import Image, ImageDraw, ImageFont
import relayout_figures as R

FONTS = [r'C:\Windows\Fonts\msyhbd.ttc', r'C:\Windows\Fonts\msyh.ttc', r'C:\Windows\Fonts\simhei.ttf']

def margin_background(header, y, lefts, rights, step):
    width = header.size[0]
    left_bound = lefts[y]; right_bound = rights[y]
    if left_bound is None:
        colours = [header.getpixel((x, y))[:3] for x in range(0, width, step)]
        colours.sort(key=lambda c: sum(c))
        mid = colours[len(colours) // 2]
        return lambda x, mid=mid: mid
    left_stop = max(1, left_bound - step)
    right_start = min(width - 1, (right_bound or 0) + step)
    left_samples = [header.getpixel((x, y))[:3] for x in range(0, left_stop, max(1, step // 2))] or [header.getpixel((0, y))[:3]]
    right_samples = [header.getpixel((x, y))[:3] for x in range(right_start, width, max(1, step // 2))] or left_samples
    def mean(samples):
        n = len(samples)
        return tuple(sum(s[i] for s in samples) // n for i in range(3))
    left_colour, right_colour = mean(left_samples), mean(right_samples)
    def colour(x):
        ratio = x / max(1, width - 1)
        return tuple(int(round(left_colour[i] + (right_colour[i] - left_colour[i]) * ratio)) for i in range(3))
    return colour

def ink_bbox(img):
    return img.getbbox()

def draw_title(header, text, top, bottom, left_x, colour):
    target_h = bottom - top + 1
    target_center = (top + bottom) / 2
    font_path = next((p for p in FONTS if __import__('os').path.exists(p)), None)
    size = target_h
    best = None
    for _ in range(5):
        font = ImageFont.truetype(font_path, size)
        scratch = Image.new('L', (header.size[0], target_h * 3), 0)
        d = ImageDraw.Draw(scratch)
        d.text((4, 4), text, font=font, fill=255)
        box = scratch.getbbox()
        if box is None: break
        h = box[3] - box[1]
        best = (font, box, h)
        if abs(h - target_h) <= 1: break
        size = max(8, round(size * target_h / max(1, h)))
    if best is None:
        return header
    font, box, h = best
    layer = Image.new('RGBA', header.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.text((left_x - box[0], target_center - h / 2 - box[1]), text, font=font, fill=colour + (255,))
    header.paste(layer, (0, 0), layer)
    return header

for member, text, tag in [('word/media/image4.png', '图 3 特色总结图', 'gaiyao'),
                          ('word/media/image24.png', '图 10.1 迭代路线图', 'xiangan')]:
    src_path = r'D:\Desktop\ninsi\A09-凝思-项目概要介绍-最终版.docx' if tag == 'gaiyao' else r'D:\Desktop\ninsi\A09-凝思-项目详细方案-最终版.docx'
    z = zipfile.ZipFile(src_path)
    img = Image.open(io.BytesIO(z.read(member))).convert('RGB')
    band = R.detect_header_band(img)
    header = img.crop((0, 0, img.size[0], band))
    counts, _, lefts, rights = R.ink_metrics(header)
    step = max(1, header.size[0] // 240)
    ink_rows = [y for y, c in enumerate(counts) if c > 4]
    ts, te = ink_rows[0], ink_rows[-1]
    sub_rows = max(12, round(0.35 * (te - ts)))
    gap = te - sub_rows
    title_left = min(l for l in lefts[ts:gap + 1] if l is not None)
    # 采样标题文字颜色
    samples = []
    ref_rows = range(ts, gap + 1)
    for y in ref_rows:
        reference = R.row_reference(header, y, step)
        for x in range(0, header.size[0], step):
            px = header.getpixel((x, y))[:3]
            if sum(abs(px[i] - reference[i]) for i in range(3)) > R.INK_TOLERANCE:
                samples.append(px)
    samples.sort(key=lambda c: sum(c), reverse=True)
    colour = samples[: max(1, len(samples) // 5)]
    colour = tuple(sum(c[i] for c in colour) // len(colour) for i in range(3))
    # 擦除标题区并重画
    work = header.copy()
    fallback = None
    for y in range(max(0, ts - 4), gap + 2):
        fit = R.row_background_fit(work, y, step) or fallback
        if fit is None:
            continue
        fallback = fit
        for x in range(header.size[0]):
            value = tuple(max(0, min(255, round(fit[c][0] * x + fit[c][1]))) for c in range(3))
            work.putpixel((x, y), value)
    work = draw_title(work, text, ts, gap, title_left, colour)
    # 拼上副标题（原样下移 extra 行）
    extra = 20
    out = Image.new('RGB', (header.size[0], header.size[1] + extra))
    out.paste(work.crop((0, 0, header.size[0], gap + 1)), (0, 0))
    filler_row = work.crop((0, gap + 1, header.size[0], gap + 2)).resize((header.size[0], extra))
    out.paste(filler_row, (0, gap + 1))
    out.paste(work.crop((0, gap + 1, header.size[0], header.size[1])), (0, gap + 1 + extra))
    band_o = header.resize((header.size[0] * 2, header.size[1] * 5), Image.NEAREST)
    band_n = out.crop((0, max(0, ts - 10), header.size[0], min(out.size[1], te + 20))).resize((header.size[0] * 2, (min(out.size[1], te + 20) - max(0, ts - 10)) * 5), Image.NEAREST)
    canvas = Image.new('RGB', (header.size[0] * 2, band_o.size[1] + band_n.size[1] + 16), (255, 255, 255))
    canvas.paste(band_o, (0, 0)); canvas.paste(band_n, (0, band_o.size[1] + 16))
    canvas.save(r'D:\Desktop\ninsi\_analysis\rerender_' + tag + '.png')
    print(tag, 'title colour', colour, 'left', title_left, 'rows', ts, gap, '->', canvas.size)
