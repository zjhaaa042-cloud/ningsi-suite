"""录制凝思桌面界面，输出演示视频用的界面帧序列（真机录屏，非合成）。

用法（在 ningsi 仓库根目录）：
    $env:PYTHONPATH="src"
    python tools/capture_ui.py --root var/video --out var/video_frames/ui --fps 8 --duration 66

产物：
    <out>/ui_00001.jpg ... 以及 <out>/capture_log.jsonl（每帧的时间戳与当前页签）

说明：
  - 窗口按 1100x620 摆放，按屏幕 DPI 缩放换算成物理像素后抓屏，成品可 1:1 落到 1920x1080；
  - 页签按时间表自动切换（状态指标 → 实时波形与频谱 → 状态热力图 → 神经反馈训练 → 评估报告 → 历史趋势）；
  - 训练页签会真实点“开始训练段/结束训练段”，报告页签直接加载上一次真实会话生成的报告文件。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import ImageGrab


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="var/video", help="会话数据根目录（用于趋势与报告）")
    parser.add_argument("--out", default="var/video_frames/ui")
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--duration", type=float, default=66.0)
    parser.add_argument("--participant", default="p01")
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(Path("src").resolve()))
    from ningsi.app.ui import StudioWindow

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("ui_*.jpg"):
        stale.unlink()

    app = StudioWindow(root_dir=args.root, participant=args.participant)
    window = app.window
    window.geometry("1100x620+20+20")
    window.attributes("-topmost", True)
    window.lift()
    window.update()

    screen = ImageGrab.grab()
    scale_x = screen.size[0] / max(1, window.winfo_screenwidth())
    scale_y = screen.size[1] / max(1, window.winfo_screenheight())
    print(f"屏幕：Tk {window.winfo_screenwidth()}x{window.winfo_screenheight()} → 抓屏 {screen.size[0]}x{screen.size[1]}（缩放 {scale_x:.2f}）")

    # 报告页签：直接加载上一次真实会话产出的报告，避免录制中途阻塞
    report_path = Path(args.root) / "reports" / "sub-p01_ses-01_run-001_report.md"
    report_text = report_path.read_text(encoding="utf-8") if report_path.exists() else "（未找到报告文件，请先运行 ningsi demo）"

    notebook = app.notebook
    frames = list(notebook.tabs())

    # 时间表：(秒, 动作名, 回调)
    def load_report():
        app.report_text.delete("1.0", "end")
        app.report_text.insert("1.0", report_text)

    def scroll_report():
        if not hasattr(app, "_scroll_step"):
            app._scroll_step = 0.0
        app._scroll_step = min(1.0, app._scroll_step + 0.12)
        app.report_text.yview_moveto(app._scroll_step)

    schedule = [
        (1.0, "tab:状态指标", lambda: notebook.select(frames[0])),
        (36.0, "tab:实时波形与频谱", lambda: notebook.select(frames[1])),
        (60.0, "tab:状态热力图", lambda: notebook.select(frames[2])),
        (84.0, "tab:神经反馈训练", lambda: notebook.select(frames[3])),
        (86.0, "action:开始训练段", app._start_segment),
        (96.0, "action:结束训练段", app._finish_segment),
        (98.0, "action:开始第二段", app._start_segment),
        (108.0, "action:结束第二段", app._finish_segment),
        (112.0, "tab:评估报告", lambda: (notebook.select(frames[4]), load_report())),
        (120.0, "action:滚动报告", scroll_report),
        (126.0, "action:滚动报告", scroll_report),
        (131.0, "action:滚动报告", scroll_report),
        (136.0, "tab:历史趋势", lambda: (notebook.select(frames[5]), app._refresh_trend())),
        (152.0, "action:结束", lambda: None),
    ]
    schedule.sort(key=lambda item: item[0])

    log = out / "capture_log.jsonl"
    entries = []
    index = 0
    start = time.perf_counter()
    interval = 1.0 / args.fps
    pending = list(schedule)
    current_action = "start"

    while True:
        now = time.perf_counter() - start
        if now > args.duration:
            break
        while pending and pending[0][0] <= now:
            _, action, callback = pending.pop(0)
            try:
                callback()
            except Exception as exc:  # 录制不应因为单个动作失败而中断
                print(f"  [warn] {action} 失败：{exc}")
            current_action = action
            window.update()
            print(f"  t={now:5.1f}s  {action}")
        window.update()

        left = window.winfo_rootx() * scale_x
        top = window.winfo_rooty() * scale_y
        right = (window.winfo_rootx() + window.winfo_width()) * scale_x
        bottom = (window.winfo_rooty() + window.winfo_height()) * scale_y
        image = ImageGrab.grab(bbox=(int(left), int(top), int(right), int(bottom)))
        index += 1
        name = out / f"ui_{index:05d}.jpg"
        image.save(name, quality=88)
        entries.append({"frame": index, "t": round(now, 3), "tab": current_action, "size": image.size})

        sleep_for = interval - ((time.perf_counter() - start) - now)
        if sleep_for > 0:
            time.sleep(sleep_for)

    log.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in entries), encoding="utf-8")
    print(f"完成：{index} 帧 → {out}；时间表 {log}")

    window.destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
