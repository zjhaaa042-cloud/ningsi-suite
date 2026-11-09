"""去掉插图顶部标题带（"图 X XXX" 标题行与其下方副标题行）——直接裁掉整条色带。

为什么用裁剪而不是擦字：标题带是**带斜纹理的设计底**，任何"擦掉文字再补底色"的做法都会留下色块或纹理断层；
裁掉整条带则**一个像素都不用重绘**，天然无瑕疵。图题在文档正文里本来就有（"图 3 特色总结图"段落），
因此图内标题属于重复信息。

同时修正图片在文档中的显示框：裁剪后高宽比变化，脚本会把每个引用该图的 drawing 的 cy 按新比例改小，
避免 Word 把图片拉伸变形。
"""

from __future__ import annotations

import io
import sys
import time
import zipfile
from pathlib import Path

from lxml import etree
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import relayout_figures as R

ROOT = Path(__file__).resolve().parents[2]
DELIVERABLES = ROOT / "deliverables"
BACKUP = ROOT / "_analysis" / "backup_titles"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def crop_band(image: Image.Image, keep_bottom_rule: int = 0) -> tuple:
    width, height = image.size
    band = R.detect_header_band(image)
    if not band or band >= height - 10:
        return image, 0
    top = max(0, band - keep_bottom_rule)
    return image.crop((0, top, width, height)), band


def fix_extents(document_xml: bytes, rels_xml: bytes, media_sizes: dict) -> bytes:
    """按裁剪后的高宽比调整 drawing 的显示高度（宽度不变），避免 Word 拉伸变形。"""
    document = etree.fromstring(document_xml)
    rels = etree.fromstring(rels_xml)
    rel_map = {node.get("Id"): node.get("Target") for node in rels}
    changed = 0
    for drawing in document.iter("{%s}drawing" % W):
        blip = drawing.find(".//{%s}blip" % A)
        if blip is None:
            continue
        target = rel_map.get(blip.get("{%s}embed" % REL))
        if not target:
            continue
        member = "word/" + target.lstrip("/")
        if member not in media_sizes:
            continue
        new_width, new_height = media_sizes[member]
        extent = drawing.find(".//{%s}extent" % WP)
        if extent is None or not extent.get("cx"):
            continue
        cx = int(extent.get("cx"))
        cy = int(round(cx * new_height / new_width))
        extent.set("cy", str(cy))
        for node in drawing.iter("{%s}ext" % A):
            if node.get("cx") and node.get("cy"):
                node.set("cy", str(cy))
        changed += 1
    return etree.tostring(document, xml_declaration=True, encoding="UTF-8", standalone=True), changed


def process_docx(source: Path, target: Path) -> dict:
    BACKUP.mkdir(parents=True, exist_ok=True)
    (BACKUP / (source.stem + "_before.docx")).write_bytes(source.read_bytes())
    with zipfile.ZipFile(source) as archive:
        entries = archive.infolist()
        payloads = {entry.filename: archive.read(entry.filename) for entry in entries}

    stats = {"images": 0, "cropped": 0, "skipped": 0, "bands": []}
    media_sizes = {}
    for name in list(payloads):
        if not name.startswith("word/media/"):
            continue
        if Path(name).suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        stats["images"] += 1
        with Image.open(io.BytesIO(payloads[name])) as raw:
            raw.load()
            if max(raw.size) > 1600:                 # 整页封面
                stats["skipped"] += 1
                continue
            canvas = raw.convert("RGB")
            cropped, band = crop_band(canvas)
        if not band:
            stats["skipped"] += 1
            continue
        payloads[name] = R.encode(cropped, Path(name).suffix)
        media_sizes[name] = cropped.size
        stats["cropped"] += 1
        stats["bands"].append(band)

    document_xml, changed = fix_extents(payloads["word/document.xml"],
                                        payloads["word/_rels/document.xml.rels"], media_sizes)
    payloads["word/document.xml"] = document_xml
    stats["extents"] = changed

    for attempt in range(20):
        try:
            if target.exists():
                target.unlink()
            break
        except PermissionError:
            if attempt == 0:
                print(f"    {target.name} 正被占用，等待释放…")
            time.sleep(5)
    else:
        target = target.with_name(f"{target.stem}-去标题带{target.suffix}")
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for entry in entries:
            archive.writestr(entry, payloads[entry.filename])
    return stats


def main() -> int:
    names = [a for a in sys.argv[1:] if not a.startswith("--")]
    names = names or ["A09-凝思-项目概要介绍-v2.docx", "A09-凝思-项目详细方案-v2.docx"]
    for name in names:
        path = Path(name) if Path(name).is_absolute() else DELIVERABLES / name
        if not path.exists():
            print(f"找不到 {path}")
            continue
        print(f"== {path.name} ==")
        stats = process_docx(path, path)
        print(f"   图片 {stats['images']} 张：裁掉标题带 {stats['cropped']} 张，跳过 {stats['skipped']} 张，"
              f"显示框同步修正 {stats['extents']} 处（裁剪前备份在 _analysis/backup_titles）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
