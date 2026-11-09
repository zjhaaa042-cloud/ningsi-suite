"""合成 A09 凝思演示视频：PPT 页 + 真实界面录屏 + 真实会话产物 + 终端输出 + 中文旁白。

分阶段执行（在 D:\\Desktop\\ningsi 下运行）：

    python ningsi\\tools\\build_demo_video.py stage --stage panels      # 渲染片头、量表、行为、预警静态面板
    python ningsi\\tools\\build_demo_video.py stage --stage ui          # 把界面帧序列编码成 ui_full.mp4
    python ningsi\\tools\\build_demo_video.py stage --stage terminal    # 渲染终端输出动画帧
    powershell -File ningsi\\tools\\build_demo_voice.ps1               # 生成逐镜中文旁白 wav + 字幕 srt
    python ningsi\\tools\\build_demo_video.py stage --stage assemble    # 逐镜合成、拼接、烧字幕、压到 150MB 内

前置：
    1. deliverables\\A09-凝思-项目简介.pptx 已导出到 var\\slides（PowerPoint 另存为 PNG）；
    2. python ningsi\\tools\\capture_ui.py 已录出 var\\video_frames\\ui_long；
    3. ningsi repo 下已跑过 demo 与 verify，产物在 ningsi\\var\\video 与 ningsi\\var\\verify.log。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]            # D:/Desktop/ningsi
REPO = ROOT / "ningsi"
BUILD = ROOT / "var" / "video_build"
SLIDES = BUILD / "slides"
VOICE = BUILD / "voice"
PANELS = BUILD / "panels"
TERMINAL = BUILD / "terminal"
SCENE_DIR = BUILD / "scenes"
UI_FRAMES = REPO / "var" / "video_frames" / "ui_long"
VERIFY_LOG = REPO / "var" / "verify.log"
REPORT_JSON = REPO / "var" / "video" / "reports" / "sub-p01_ses-01_run-001_report.json"
OUT_VIDEO = ROOT / "deliverables" / "A09凝思_实机演示视频.mp4"
OUT_SRT = ROOT / "deliverables" / "A09凝思_实机演示视频.srt"

W, H, FPS = 1920, 1080, 30
NAVY, BLUE, TEAL, AMBER, GREY, HAIR = "#1B3B6F", "#2E6DA4", "#009E73", "#E69F00", "#5B6B7C", "#DCE3EC"
INK, WHITE, PANEL = "#0E1B33", "#FFFFFF", "#F5F8FC"

FONT_REG = r"C:\Windows\Fonts\msyh.ttc"
FONT_BOLD = r"C:\Windows\Fonts\msyhbd.ttc"
FONT_MONO = r"C:\Windows\Fonts\consola.ttf"
FONT_MONO_BOLD = r"C:\Windows\Fonts\consolab.ttf"

TEAM = "胆double天队"

# --------------------------------------------------------------------------- 工具


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REG, size)


def mono(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_MONO_BOLD if bold else FONT_MONO, size)


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("  $", " ".join(str(item) for item in cmd[:8]), "..." if len(cmd) > 8 else "")
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    if result.returncode != 0:
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        raise SystemExit(f"命令失败：{cmd[0]}")


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def base_canvas(title: str, subtitle: str = "") -> Image.Image:
    image = Image.new("RGB", (W, H), WHITE)
    pen = ImageDraw.Draw(image)
    pen.rectangle((0, 0, W, 12), fill=NAVY)
    pen.text((110, 78), title, font=font(58, True), fill=NAVY)
    if subtitle:
        pen.text((112, 162), subtitle, font=font(28), fill=GREY)
    pen.line((110, 226, W - 110, 226), fill=HAIR, width=3)
    pen.text((110, H - 74), f"凝思 Ningsi ｜ A09 ｜ {TEAM}", font=font(24), fill=GREY)
    return image


def card(image: Image.Image, box, label: str, value: str, accent: str = BLUE, hint: str = "") -> None:
    pen = ImageDraw.Draw(image)
    left, top, right, bottom = box
    pen.rounded_rectangle((left, top, right, bottom), radius=18, fill=PANEL, outline=HAIR, width=2)
    pen.rectangle((left, top, left + 10, bottom), fill=accent)
    pen.text((left + 40, top + 26), label, font=font(30, True), fill=INK)
    pen.text((left + 40, top + 76), value, font=font(52, True), fill=accent)
    if hint:
        pen.text((left + 40, top + 146), hint, font=font(24), fill=GREY)


# --------------------------------------------------------------------------- 真实数据


def report() -> dict:
    if not REPORT_JSON.exists():
        raise SystemExit(f"缺少真实会话产物：{REPORT_JSON}（先跑 ningsi demo --root var/video）")
    return json.loads(REPORT_JSON.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- 静态面板


def panel_intro() -> Path:
    out = PANELS / "intro.png"
    image = Image.new("RGB", (W, H), INK)
    pen = ImageDraw.Draw(image)
    pen.rectangle((0, 0, W, 14), fill=TEAL)
    for index, (x0, y0, x1, y1) in enumerate([(1380, 120, 1900, 640), (1560, 520, 1980, 940)], start=1):
        pen.ellipse((x0, y0, x1, y1), fill="#12345A")
    pen.text((110, 300), "凝思（Ningsi）", font=font(96, True), fill=WHITE)
    pen.line((112, 430, 430, 430), fill=TEAL, width=6)
    pen.text((110, 470), "AI+便携脑电设备的专注力强化训练系统", font=font(40), fill="#C9D6E6")
    pen.text((110, 540), "每一段专注，都看得见", font=font(40, True), fill=TEAL)
    pen.rounded_rectangle((110, 630, 900, 700), radius=35, fill="#123B2F")
    pen.text((150, 650), "真实采集 · 特征可解释 · 评估可联动 · 训练有反馈", font=font(28), fill=TEAL)
    pen.text((110, 800), f"赛题编号 A09 ｜ 命题企业 杭州金扬智能科技有限公司 ｜ {TEAM}", font=font(28), fill="#9FB3CC")
    pen.text((110, 848), "项目演示视频 ｜ 2026 年 9 月", font=font(28), fill="#7E93AD")
    image.save(out)
    return out


def panel_scales() -> Path:
    data = report()
    sas, sds = data["scales"]["SAS"], data["scales"]["SDS"]
    image = base_canvas("量表联合评估：SAS / SDS", "Zung 规则计分，反向题自动反转，粗分 ×1.25 得标准分；作为可横向比较的第二类证据")
    card(image, (110, 280, 940, 560), "焦虑自评量表 SAS",
         f"粗分 {sas['raw_score']} → 标准分 {sas['standard_score']}", BLUE,
         f"分级：{sas['level']}｜作答 {sas['answered']}/20｜反向题 {len(sas['reverse_items'])} 题")
    card(image, (980, 280, 1810, 560), "抑郁自评量表 SDS",
         f"粗分 {sds['raw_score']} → 标准分 {sds['standard_score']}", AMBER,
         f"分级：{sds['level']}｜作答 {sds['answered']}/20｜反向题 {len(sds['reverse_items'])} 题")
    card(image, (110, 620, 1810, 880), "落盘与复算",
         "scales/sub-p01_ses-01_run-001_scales.jsonl", TEAL,
         "逐题作答、反向标记、有效分与标准分全部留档；与脑电指标、行为任务一起进入联合评估，任一条结论都可沿链路回算")
    image.save(PANELS / "scales.png")
    return PANELS / "scales.png"


def _behavior_value(data: dict, key: str, digits: int = 3):
    value = data.get(key)
    return "—" if value is None else (round(float(value), digits) if isinstance(value, (int, float)) else value)


def panel_behavior() -> Path:
    data = report()
    behavior = data.get("behavior", {})
    sart = behavior.get("sart", behavior)
    pvt = behavior.get("pvt", {})
    image = base_canvas("行为任务：SART 180 试次 + PVT-B 3 分钟", "第三类独立证据：任务表现本身；序列按被试、会话与 Run 确定性轮换，可复现")
    rows = [
        ("SART 正确率", f"{float(sart.get('accuracy', 0)):.1%}" if "accuracy" in sart else "—", BLUE, "Go 试次按对比例"),
        ("SART 虚报率", f"{float(sart.get('commission_rate', 0)):.1%}" if "commission_rate" in sart else "—", AMBER, "No-Go 误按比例（20 个 No-Go）"),
        ("SART 平均反应时", f"{_behavior_value(sart, 'mean_rt')} s", TEAL, "Go 试次反应时均值"),
        ("反应时变异", f"{_behavior_value(sart, 'rt_cv')}", BLUE, "试次级波动，越大越不稳定"),
        ("PVT-B 中位反应时", f"{_behavior_value(pvt, 'median_rt')} s", AMBER, "连续 3 分钟警觉任务"),
        ("PVT-B 慢反应率", f"{float(pvt.get('lapse_rate', 0)):.1%}" if "lapse_rate" in pvt else "—", TEAL, "反应时超过阈值比例"),
    ]
    top = 290
    for index, (label, value, accent, hint) in enumerate(rows):
        column, line = index % 2, index // 2
        x = 110 + column * 870
        y = top + line * 250
        card(image, (x, y, x + 830, y + 210), label, value, accent, hint)
    image.save(PANELS / "behavior.png")
    return PANELS / "behavior.png"


def panel_alerts() -> Path:
    data = report()
    indicators = data["indicators"]
    summary = indicators["summary"]
    quality = data["quality"]
    alerts = data.get("alerts", {})
    active = alerts.get("active") or alerts.get("triggered") or []
    if isinstance(active, dict):
        active = list(active.keys())
    image = base_canvas("状态指标、质量门控与预警", "每 2 秒推进一个 4 秒窗；质量不合格窗不参与指标与预警计时（等于把时钟暂停）")
    card(image, (110, 280, 640, 560), "专注度", f"{summary['focus']['mean']:.2f}", BLUE,
         f"n={summary['focus']['n']}｜0.5 表示与个体基线持平")
    card(image, (690, 280, 1220, 560), "放松度", f"{summary['relax']['mean']:.2f}", TEAL,
         f"n={summary['relax']['n']}｜α/β 相对基线")
    card(image, (1270, 280, 1810, 560), "认知负荷", f"{summary['load']['mean']:.2f}", AMBER,
         f"n={summary['load']['n']}｜θ/α 相对基线")
    card(image, (110, 620, 940, 880), "本次运行触发的预警",
         "低专注（low_focus）" if active else "无", "#D55E00",
         "阈值：专注 <0.35 持续 20 秒触发，≥0.45 持续 10 秒解除；高负荷 ≥0.75 持续 30 秒触发")
    card(image, (980, 620, 1810, 880), "可用窗比例",
         f"{quality['valid_ratio']:.0%}（{quality['windows_usable']}/{quality['windows_total']}）", TEAL,
         "单窗判定：幅度 >150 µV、恒定通道、通道跨度、30–45 Hz 肌电代理、0.5–4 Hz 眼电代理")
    image.save(PANELS / "alerts.png")
    return PANELS / "alerts.png"


def panel_trend() -> Path:
    image = base_canvas("历史趋势与可比性", "按周 / 月聚合；设备、采样率或通道数变化时明确标注不可比并排除该记录")
    figure = ROOT / "deliverables" / "figures" / "fig_trend.png"
    if figure.exists():
        chart = Image.open(figure).convert("RGB")
        chart.thumbnail((1680, 620))
        image.paste(chart, (120, 300))
    pen = ImageDraw.Draw(image)
    pen.text((120, 940), "首次会话只有一个数据点：趋势随复评次数累积，这也是系统不在首次会话就给趋势结论的原因。",
             font=font(26), fill=GREY)
    image.save(PANELS / "trend.png")
    return PANELS / "trend.png"


# --------------------------------------------------------------------------- 终端动画


ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# PowerShell 把子进程的 stderr 升级成异常时会在日志里插入一段报错堆栈，演示画面里必须滤掉
PS_NOISE = re.compile(
    r"^(powershell\s*:|At line:|FullyQualifiedErrorId|CategoryInfo|\s*\+|注意|.*NativeCommandError|"
    r"-{5,}|={5,}|\.{5,}\s*$)")


def read_log(path: Path) -> str:
    """按 BOM / 编码特征读取日志：PowerShell 重定向默认写 UTF-16-LE，直接按 UTF-8 读会整屏乱码。"""
    raw = path.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    if raw[:3] == b"\xef\xbb\xbf":
        return raw.decode("utf-8-sig")
    for encoding in ("utf-8", "gbk"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def terminal_lines() -> list[str]:
    lines: list[str] = []
    if VERIFY_LOG.exists():
        raw = ANSI.sub("", read_log(VERIFY_LOG))
        keep = [item.rstrip() for item in raw.splitlines()
                if item.strip() and not PS_NOISE.match(item)]
        lines.extend(keep)
    if not lines:
        raise SystemExit(f"缺少终端日志：{VERIFY_LOG}")
    return lines


def panel_terminal_frames() -> int:
    if TERMINAL.exists():
        for stale in TERMINAL.glob("term_*.png"):
            stale.unlink()
    else:
        TERMINAL.mkdir(parents=True, exist_ok=True)
    lines = terminal_lines()
    header = [
        "PS> cd ningsi",
        "PS ningsi> $env:PYTHONPATH = \"src\"",
        "PS ningsi> .\\scripts\\verify.ps1",
        "",
    ]
    payload = header + lines
    visible = 30
    wrap_at = 138                     # 超长行按字符宽度折行，不截断内容
    wrapped: list[str] = []
    for line in payload:
        if len(line) <= wrap_at:
            wrapped.append(line)
        else:
            head = line
            while len(head) > wrap_at:
                cut = head.rfind(" ", 0, wrap_at)
                cut = cut if cut > wrap_at * 0.6 else wrap_at
                wrapped.append(head[:cut])
                head = "    " + head[cut:].lstrip()
            wrapped.append(head)
    payload = wrapped
    total = max(visible + 8, len(payload) + 4)
    frame_font = mono(23)
    line_height = 30
    for frame in range(total):
        shown = payload[: min(len(payload), max(1, frame * max(1, len(payload) // total) + visible))]
        image = Image.new("RGB", (W, H), "#0C1117")
        pen = ImageDraw.Draw(image)
        pen.rectangle((0, 0, W, 64), fill="#161B22")
        pen.text((26, 20), "凝思 · 一键复现（真实运行输出）", font=font(26, True), fill="#E6EDF3")
        pen.text((W - 360, 24), "PowerShell · scripts/verify.ps1", font=mono(22), fill="#7D8590")
        y = 96
        for line in shown[-visible:]:
            color = "#7EE787" if line.startswith("PS ") else ("#FFA657" if "预警" in line else "#C9D1D9")
            # Consolas 无中文字形：含中文的行改用微软雅黑，避免出现方框
            line_font = font(22) if any(ord(ch) > 127 for ch in line) else frame_font
            pen.text((26, y), line[:wrap_at], font=line_font, fill=color)
            y += line_height
        pen.text((26, H - 46), "51 个自动化用例 + 端到端演示：scripts/verify.ps1", font=font(22), fill="#7D8590")
        image.save(TERMINAL / f"term_{frame + 1:05d}.png")
    print(f"  终端帧：{total} 张 → {TERMINAL}")
    return total


# --------------------------------------------------------------------------- 界面录屏中间文件


def build_ui_full() -> Path:
    frames = sorted(UI_FRAMES.glob("ui_*.jpg"))
    if not frames:
        raise SystemExit(f"缺少界面帧：{UI_FRAMES}（先运行 tools/capture_ui.py）")
    # 按抓帧日志换算实际帧率，保证 mp4 时间轴与截图时间表（wall clock 秒）一致
    effective = 8.0
    log = UI_FRAMES / "capture_log.jsonl"
    if log.exists():
        entries = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
        if entries:
            effective = len(entries) / max(0.1, entries[-1]["t"])
    out = BUILD / "ui_full.mp4"
    run([
        "ffmpeg", "-y", "-framerate", f"{effective:.6f}", "-i", str(UI_FRAMES / "ui_%05d.jpg"),
        "-vf", f"fps={FPS},scale={W}:{H}", "-c:v", "libx264", "-crf", "18",
        "-preset", "medium", "-pix_fmt", "yuv420p", str(out),
    ])
    print(f"  界面中间片：{len(frames)} 帧 / 有效帧率 {effective:.2f} fps / {probe_duration(out):.1f}s → {out}")
    return out


# --------------------------------------------------------------------------- 分镜

SCENES: list[dict] = [
    {"id": "s01", "kind": "image", "src": "intro.png",
     "text": "我们是 A09 胆double天队。凝思，是一套基于便携脑电设备的专注力强化训练与心理状态评估系统。"},
    {"id": "s02", "kind": "slide", "src": "slide_02.png",
     "text": "传统评估靠问卷与访谈，主观而且离散。便携脑电已经普及，但把信号变成可信指标，把指标与量表行为放进同一套证据，把指标变成能练的反馈，这三件事仍然缺。"},
    {"id": "s03", "kind": "slide", "src": "slide_05.png",
     "text": "系统分四层：用户层、交互层、智能层、支撑层，底部是数据安全与合规横条。架构图上每一个模块，都能在代码里找到对应文件。"},
    {"id": "s04", "kind": "ui", "start": 3.0, "end": 22.0,
     "text": "接上设备先做质检：逐窗检查幅度、恒定通道、通道跨度、肌电与眼电代理。不合格的窗会被剔除并记下原因，不进入指标，也不参与预警计时。"},
    {"id": "s05", "kind": "image", "src": "scales.png",
     "text": "量表按 Zung 规则计分，反向题自动反转，粗分乘以一点二五得到标准分。本次 SAS 标准分五十六，SDS 标准分六十三，作为可横向比较的第二类证据。"},
    {"id": "s06", "kind": "ui", "start": 24.0, "end": 36.0,
     "text": "静息基线是个体化的参照：睁眼与闭眼各两分钟，两段各自独立质检与建基线，任务态指标以睁眼基线为准。合格窗少于五个时，系统会明确拒绝出结论，而不是给一条不可比的曲线。"},
    {"id": "s07", "kind": "image", "src": "behavior.png",
     "text": "行为任务提供第三类独立证据：SART 记录正确率、虚报率、漏报率与反应时变异；PVT-B 给出中位反应时与慢反应率。序列按被试、会话与 Run 确定性轮换，可复现。"},
    {"id": "s08", "kind": "ui", "start": 38.0, "end": 58.0,
     "text": "实时显示去直流波形与功率谱；每两秒推进一个四秒窗。热力图按窗着色，低质量窗用灰色的缺失块标出，绝不把伪迹画成状态变化。"},
    {"id": "s09", "kind": "ui", "start": 62.0, "end": 82.0,
     "text": "预警按连续越界时长触发：低专注低于零点三五持续二十秒，高负荷高于零点七五持续三十秒。质量不合格窗不参与计时，等于把时钟暂停，避免动作伪迹造成误报。"},
    {"id": "s10", "kind": "ui", "start": 114.0, "end": 134.0,
     "text": "报告里每条结论都能指回所用的指标与口径版本。当脑电与量表或行为方向不一致时，系统写的是建议复测，而不是硬合并成一个分数。"},
    {"id": "s11", "kind": "ui", "start": 85.0, "end": 110.0,
     "text": "训练初始目标取自当次基线中位数对应的评分。每段记录平均专注度、达标时间占比与波动幅度，达标占比高就上调目标并延长保持时长，表现波动则先降低要求。训练前后各采一次静息基线，用同一口径对比。"},
    {"id": "s12", "kind": "ui", "start": 137.0, "end": 151.0},
    {"id": "s13", "kind": "terminal", "src": "terminal",
     "text": "一条命令即可跑通质检、量表、基线、行为任务、指标、报告、训练与模型训练；五十一个自动化用例全部通过，模型按被试划分评估，结果写入 models 目录下的分类器文件。"},
    {"id": "s14", "kind": "slide", "src": "slide_15.png",
     "text": "凝思不提供医疗诊断，也不得用于处罚或自动上岗决策。真实设备的多品牌适配与效度验证，是我们下一步的重点。可信度，随实测数据积累而提升。"},
]

SCENES[11]["text"] = "趋势只在设备、采样率与通道数一致时绘制。配置变化的记录会被排除，并明确标注不可比。"

PAD = 3.0          # 每镜旁白后留白，让画面有呼吸空间
MIN_IMAGE = 8.0    # 静态画面最短时长
MIN_UI = 14.0      # 界面镜头最短时长


def narration_path(scene: dict) -> Path:
    return VOICE / f"{scene['id']}.wav"


def scene_duration(scene: dict) -> float:
    voice = narration_path(scene)
    if not voice.exists():
        raise SystemExit(f"缺少旁白：{voice}（先运行 tools/build_demo_voice.ps1）")
    spoken = probe_duration(voice)
    floor = MIN_UI if scene["kind"] == "ui" else MIN_IMAGE
    return round(max(spoken + PAD, floor), 3)


def encode_scene(scene: dict, duration: float, ui_full: Path) -> Path:
    out = SCENE_DIR / f"{scene['id']}.mp4"
    voice = narration_path(scene)
    common = ["-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
              "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", "-r", str(FPS), "-t", f"{duration}"]
    if scene["kind"] == "ui":
        span = scene["end"] - scene["start"]
        factor = max(0.5, min(2.4, duration / span))
        run(["ffmpeg", "-y", "-ss", f"{scene['start']}", "-t", f"{span}", "-i", str(ui_full), "-i", str(voice),
             "-filter_complex",
             f"[0:v]setpts={factor:.4f}*PTS,fps={FPS},scale={W}:{H}:force_original_aspect_ratio=decrease,"
             f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=white[v];[1:a]apad[a]",
             "-map", "[v]", "-map", "[a]"] + common + [str(out)])
    elif scene["kind"] == "terminal":
        run(["ffmpeg", "-y", "-framerate", "6", "-i", str(TERMINAL / "term_%05d.png"), "-i", str(voice),
             "-filter_complex",
             f"[0:v]scale={W}:{H},fps={FPS}[v];[1:a]apad[a]",
             "-map", "[v]", "-map", "[a]"] + common + [str(out)])
    else:
        source = (PANELS / scene["src"]) if scene["kind"] == "image" else (SLIDES / scene["src"])
        if not source.exists():
            raise SystemExit(f"缺少画面素材：{source}")
        run(["ffmpeg", "-y", "-loop", "1", "-i", str(source), "-i", str(voice),
             "-filter_complex",
             f"[0:v]scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=white,"
             f"fps={FPS}[v];[1:a]apad[a]",
             "-map", "[v]", "-map", "[a]"] + common + [str(out)])
    return out


def srt_timestamp(seconds: float) -> str:
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{secs:06.3f}".replace(".", ",")


def wrap_subtitle(text: str, max_lines: int = 3) -> str:
    """把一句旁白切成最多三行、按长度均分的字幕文本（优先在标点处断行，保持语序）。"""
    limit = max(20, -(-len(text) // max(1, max_lines)))
    lines: list[str] = []
    current = ""
    for char in text:
        current += char
        if (len(current) >= limit and char in "。；，、！？：") or len(current) >= limit + 6:
            lines.append(current)
            current = ""
    if current:
        lines.append(current)
    if len(lines) > max_lines:                # 行数超限时把末尾几行并进最后一行（保持顺序）
        lines = lines[:max_lines - 1] + ["".join(lines[max_lines - 1:])]
    return "\n".join(lines)


def finalize() -> int:
    """烧入中文字幕并压到 150MB 以内。"""
    joined = BUILD / "joined.mp4"
    if not joined.exists():
        raise SystemExit(f"缺少拼接结果：{joined}（先执行 assemble）")
    # 依据各镜实际时长重建字幕（场景已按同一规则编码，重跑 burn 不会再改画面）
    cursor, entries = 0.0, []
    for index, scene in enumerate(SCENES, 1):
        duration = scene_duration(scene)
        entries.append(f"{index}\n{srt_timestamp(cursor)} --> {srt_timestamp(cursor + duration - 0.2)}\n"
                       f"{wrap_subtitle(scene['text'])}\n")
        cursor += duration
    OUT_SRT.write_text("\n".join(entries), encoding="utf-8")
    subs = BUILD / "subs.srt"
    subs.write_text(OUT_SRT.read_text(encoding="utf-8"), encoding="utf-8")
    print("  开始烧字幕并按 150MB 上限压制 ...")
    run(["ffmpeg", "-y", "-i", str(joined), "-vf",
         f"subtitles={subs.name}:force_style='FontName=Microsoft YaHei,FontSize=12,"
         "PrimaryColour=&H00FFFFFF,OutlineColour=&H66000000,BorderStyle=3,Outline=1,Shadow=0,MarginV=26'",
         "-c:v", "libx264", "-profile:v", "high", "-preset", "slow", "-crf", "16",
         "-maxrate", "6000k", "-bufsize", "12000k", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2",
         "-movflags", "+faststart", str(OUT_VIDEO)], cwd=BUILD)
    size_mb = OUT_VIDEO.stat().st_size / 1024 / 1024
    print(f"  成品：{OUT_VIDEO}｜{probe_duration(OUT_VIDEO)/60:.2f} 分钟｜{size_mb:.1f} MB")
    print(f"  字幕：{OUT_SRT}")
    return 0


def assemble() -> int:
    for folder in (PANELS, SCENE_DIR):
        folder.mkdir(parents=True, exist_ok=True)
    ui_full = BUILD / "ui_full.mp4"
    if not ui_full.exists():
        ui_full = build_ui_full()

    total, cursor = 0.0, 0.0
    srt_lines: list[str] = []
    list_file = SCENE_DIR / "concat.txt"
    parts: list[str] = []
    for index, scene in enumerate(SCENES, 1):
        duration = scene_duration(scene)
        part = encode_scene(scene, duration, ui_full)
        parts.append(part.name)
        srt_lines.append(f"{index}\n{srt_timestamp(cursor)} --> {srt_timestamp(cursor + duration - 0.2)}\n"
                         f"{wrap_subtitle(scene['text'])}\n")
        cursor += duration
        total += duration
        print(f"  {scene['id']}  {duration:5.1f}s  （累计 {cursor/60:.2f} 分钟）")
    list_file.write_text("\n".join(f"file '{name}'" for name in parts), encoding="utf-8")

    joined = BUILD / "joined.mp4"
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file.name, "-c", "copy", str(joined)], cwd=SCENE_DIR)

    OUT_SRT.write_text("\n".join(srt_lines), encoding="utf-8")
    print(f"  分镜拼接完成：{total/60:.2f} 分钟 → {joined.name}")
    return finalize()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", nargs="?", default="assemble")
    parser.add_argument("--stage", dest="stage_opt", choices=["panels", "terminal", "ui", "assemble", "burn", "all"])
    args = parser.parse_args()
    stage = args.stage_opt or args.stage

    if stage in ("panels",):
        PANELS.mkdir(parents=True, exist_ok=True)
        for builder in (panel_intro, panel_scales, panel_behavior, panel_alerts, panel_trend):
            print("  面板：", builder().name)
    elif stage == "terminal":
        panel_terminal_frames()
    elif stage == "ui":
        build_ui_full()
    elif stage == "assemble":
        assemble()
    elif stage == "burn":
        finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
