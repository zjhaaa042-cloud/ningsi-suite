"""桌面界面：指标面板、状态热力图、神经反馈训练、评估报告与趋势。

界面绘制只依赖标准库 Tkinter，便于随包分发；数据来自与命令行相同的流水线，
无设备时使用仿真源联调。
"""

from __future__ import annotations

import json
from pathlib import Path

from ningsi import config
from ningsi.app.pipeline import SessionConfig, SessionRunner
from ningsi.acquisition.simulate import SyntheticEEG
from ningsi.monitoring import history as history_module
from ningsi.monitoring.heatmap import band_of, render_text
from ningsi.signal.baseline import build_baseline
from ningsi.signal.indicators import compute_indicators
from ningsi.signal.window import analyze_window
from ningsi.training.neurofeedback import NeurofeedbackSession, initial_target

STATE_SEQUENCE = ("rest",) * 4 + ("focused",) * 10 + ("drowsy",) * 8 + ("loaded",) * 6


class StudioWindow:
    """主窗口：一个 Notebook，六个页签对应六类界面。"""

    def __init__(self, root_dir: str = "var/session", participant: str = "p01") -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root_dir = Path(root_dir)
        self.participant = participant
        self.sim = SyntheticEEG(seed=3)
        self.index = 0
        self.series = []
        self.training = None
        self.baseline = None

        self.window = tk.Tk()
        self.window.title(f"凝思 · 专注力强化训练与心理状态评估（{config.VERSION}）")
        self.window.geometry("980x680")
        self.notebook = ttk.Notebook(self.window)
        self.notebook.pack(fill="both", expand=True)
        self._build_panel_tab()
        self._build_wave_tab()
        self._build_heatmap_tab()
        self._build_training_tab()
        self._build_report_tab()
        self._build_trend_tab()
        self._prepare_baseline()
        self.window.after(600, self._tick)

    # --- 页签 1：状态指标面板 ---
    def _build_panel_tab(self) -> None:
        frame = self.ttk.Frame(self.notebook)
        self.notebook.add(frame, text="状态指标")
        self.bars = {}
        for index, (key, label) in enumerate((("focus", "专注度"), ("relax", "放松度"), ("load", "认知负荷"))):
            self.tk.Label(frame, text=label, font=("Microsoft YaHei", 12)).grid(row=index, column=0, sticky="w", padx=12, pady=10)
            bar = self.ttk.Progressbar(frame, length=420, maximum=100)
            bar.grid(row=index, column=1, padx=12)
            value = self.tk.Label(frame, text="--", font=("Consolas", 12))
            value.grid(row=index, column=2, padx=12)
            self.bars[key] = (bar, value)
        self.quality_label = self.tk.Label(frame, text="等待数据…", font=("Microsoft YaHei", 11), justify="left")
        self.quality_label.grid(row=3, column=0, columnspan=3, sticky="w", padx=12, pady=12)
        self.alert_label = self.tk.Label(frame, text="预警：无", font=("Microsoft YaHei", 11), fg="#b23c17")
        self.alert_label.grid(row=4, column=0, columnspan=3, sticky="w", padx=12)
        self.tk.Label(frame, text="数据来源：仿真源（无设备联调）；接真实设备时改为 LSL 源，界面不变。",
                      font=("Microsoft YaHei", 9), fg="#666").grid(row=5, column=0, columnspan=3, sticky="w", padx=12, pady=8)

    # --- 页签 2：实时波形与频谱 ---
    def _build_wave_tab(self) -> None:
        frame = self.ttk.Frame(self.notebook)
        self.notebook.add(frame, text="实时波形与频谱")
        self.wave_canvas = self.tk.Canvas(frame, width=940, height=250, bg="white", highlightthickness=0)
        self.wave_canvas.pack(fill="x", padx=8, pady=(8, 2))
        self.spectrum_canvas = self.tk.Canvas(frame, width=940, height=230, bg="white", highlightthickness=0)
        self.spectrum_canvas.pack(fill="x", padx=8, pady=(2, 8))
        self.wave_label = self.tk.Label(frame, text="去直流波形（最近 4 秒）｜频谱（Welch 4 s/2 s/50%）",
                                        font=("Microsoft YaHei", 9), fg="#666")
        self.wave_label.pack(anchor="w", padx=8)

    def _draw_wave_and_spectrum(self, raw, srate: float) -> None:
        import numpy as np
        from ningsi.signal.spectrum import welch_psd

        self.wave_canvas.delete("all")
        channel = np.asarray(raw, dtype=float)[0] - float(np.mean(np.asarray(raw, dtype=float)[0]))
        if channel.size > 1:
            step = max(1, channel.size // 900)
            shown = channel[::step]
            scale = max(1e-6, float(np.max(np.abs(shown))))
            width, height = 920, 210
            coordinates = []
            for index, value in enumerate(shown):
                x = 10 + width * index / max(1, len(shown) - 1)
                y = height / 2 - (height / 2 - 12) * value / scale
                coordinates.extend([x, y])
            self.wave_canvas.create_line(*coordinates, fill="#2b6cb0", width=1)
            self.wave_canvas.create_text(12, 12, anchor="nw", text=f"幅度 ±{scale:.1f} µV", font=("Consolas", 9))

        self.spectrum_canvas.delete("all")
        spectrum = welch_psd(channel, srate)
        peak = max(1e-9, float(np.max(spectrum.psd)))
        bar_width = max(1.0, 920 / max(1, spectrum.freqs.size))
        for index, (freq, power) in enumerate(zip(spectrum.freqs, spectrum.psd)):
            x = 10 + index * bar_width
            height = 190 * float(power) / peak
            color = "#E69F00" if 4 <= freq < 8 else ("#56B4E9" if 8 <= freq < 13 else ("#1B3B6F" if 13 <= freq < 30 else "#6E9E8A"))
            self.spectrum_canvas.create_rectangle(x, 210 - height, x + bar_width - 1, 210, fill=color, outline="")
        self.spectrum_canvas.create_line(10, 210, 930, 210, fill="#999")
        for freq in (4, 8, 13, 30):
            x = 10 + 920 * freq / 45.0
            self.spectrum_canvas.create_text(x, 220, text=f"{freq}Hz", font=("Consolas", 8))
        self.spectrum_canvas.create_text(12, 14, anchor="nw",
                                         text="功率谱密度（µV²/Hz）｜黄=θ 蓝=α 深蓝=β 绿=γ",
                                         font=("Microsoft YaHei", 9), fill="#444")

    # --- 页签 3：状态热力图 ---
    def _build_heatmap_tab(self) -> None:
        frame = self.ttk.Frame(self.notebook)
        self.notebook.add(frame, text="状态热力图")
        self.canvas = self.tk.Canvas(frame, width=940, height=520, bg="white", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=8, pady=8)
        legend = "   ".join(f"{item['range']} {item['label']}" for item in
                            [{"range": f"{low:.2f}–{high:.2f}", "label": label} for low, high, label, _ in config.HEATMAP_BANDS])
        self.tk.Label(frame, text="档位：" + legend + "   — 低质量缺失", font=("Microsoft YaHei", 9)).pack(anchor="w", padx=8)

    # --- 页签 3：神经反馈训练 ---
    def _build_training_tab(self) -> None:
        frame = self.ttk.Frame(self.notebook)
        self.notebook.add(frame, text="神经反馈训练")
        self.training_info = self.tk.Label(frame, text="等待基线…", font=("Microsoft YaHei", 11), justify="left")
        self.training_info.pack(anchor="w", padx=12, pady=10)
        self.feedback_canvas = self.tk.Canvas(frame, width=420, height=240, bg="#f4f7fb", highlightthickness=0)
        self.feedback_canvas.pack(anchor="w", padx=12)
        row = self.ttk.Frame(frame)
        row.pack(anchor="w", padx=12, pady=8)
        self.ttk.Button(row, text="开始训练段", command=self._start_segment).pack(side="left", padx=4)
        self.ttk.Button(row, text="结束训练段", command=self._finish_segment).pack(side="left", padx=4)
        self.segment_label = self.tk.Label(frame, text="分段记录：无", font=("Consolas", 10), justify="left")
        self.segment_label.pack(anchor="w", padx=12, pady=6)

    # --- 页签 4：评估报告 ---
    def _build_report_tab(self) -> None:
        frame = self.ttk.Frame(self.notebook)
        self.notebook.add(frame, text="评估报告")
        self.ttk.Button(frame, text="运行一次完整会话并生成报告", command=self._run_session).pack(anchor="w", padx=12, pady=8)
        self.report_text = self.tk.Text(frame, wrap="word", font=("Microsoft YaHei", 10))
        self.report_text.pack(fill="both", expand=True, padx=12, pady=8)

    # --- 页签 5：历史趋势 ---
    def _build_trend_tab(self) -> None:
        frame = self.ttk.Frame(self.notebook)
        self.notebook.add(frame, text="历史趋势")
        self.ttk.Button(frame, text="刷新周趋势", command=self._refresh_trend).pack(anchor="w", padx=12, pady=8)
        self.trend_canvas = self.tk.Canvas(frame, width=900, height=420, bg="white", highlightthickness=0)
        self.trend_canvas.pack(fill="both", expand=True, padx=12, pady=8)

    # --- 数据流 ---
    def _prepare_baseline(self) -> None:
        windows = [analyze_window(self.sim.window("rest"), self.sim.srate, t_end=index * config.STEP_SEC) for index in range(8)]
        self.baseline = build_baseline(windows, device="sim-bsense", srate=self.sim.srate)
        target, rationale = initial_target(self.baseline)
        self.training = NeurofeedbackSession(self.participant, "01", "001", mode="quick", target=target, rationale=rationale)
        self.training_info.config(text=f"初始目标：{target:.2f}（{rationale}）")

    def _tick(self) -> None:
        state = STATE_SEQUENCE[self.index % len(STATE_SEQUENCE)]
        raw = self.sim.window(state)
        window = analyze_window(raw, self.sim.srate, t_end=self.index * config.STEP_SEC)
        result = compute_indicators(window, self.baseline) if window.usable else None
        if result is not None:
            for key, (bar, label) in self.bars.items():
                score = result.score(key) or 0.0
                bar["value"] = score * 100
                label.config(text=f"{score:.2f}")
            self.quality_label.config(
                text=f"窗 {self.index + 1}｜状态源：{state}｜质量：{'可用' if window.usable else '不可用（' + ','.join(window.quality.reasons) + '）'}"
                     f"｜可用窗比例 {sum(1 for _, value in self.series if value is not None) / max(1, len(self.series) or 1):.0%}"
            )
            self.series.append((window.t_end, result.score("focus")))
            if self.training and getattr(self, "segment_open", False):
                self.segment_samples.append((window.t_end, result.score("focus")))
            self._draw_heatmap()
            self._draw_wave_and_spectrum(raw, self.sim.srate)
            self._draw_feedback(result.score("focus") or 0.0, self.training.target if self.training else 0.5)
        else:
            self.series.append((window.t_end, None))
            self.quality_label.config(text=f"窗 {self.index + 1}｜质量不合格：{','.join(window.quality.reasons)}（不计入指标与计时）")
        self.index += 1
        self.window.after(200, self._tick)

    def _draw_heatmap(self) -> None:
        self.canvas.delete("all")
        cell = 22
        columns = 36
        for index, (_, score) in enumerate(self.series[-360:]):
            band = band_of(score)
            x = 12 + (index % columns) * cell
            y = 12 + (index // columns) * cell
            self.canvas.create_rectangle(x, y, x + cell - 2, y + cell - 2, fill=band["color"], outline="")
        pass

    def _draw_feedback(self, score: float, target: float) -> None:
        self.feedback_canvas.delete("all")
        size = 60 + 140 * max(0.0, min(1.0, score))
        cx, cy = 210, 120
        color = "#2b6cb0" if score >= target else "#9aa5b1"
        self.feedback_canvas.create_oval(cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2, fill=color, outline="")
        self.feedback_canvas.create_text(cx, cy, text=f"{score:.2f}", fill="white", font=("Consolas", 14))
        self.feedback_canvas.create_text(cx, 220, text=f"目标 {target:.2f}", font=("Microsoft YaHei", 10))

    def _start_segment(self) -> None:
        self.segment_open = True
        self.segment_samples = []
        self.segment_label.config(text="分段记录：采集中…")

    def _finish_segment(self) -> None:
        if not getattr(self, "segment_open", False):
            return
        self.segment_open = False
        segment = self.training.add_segment(self.segment_samples, 120.0)
        stats = segment.stats()
        self.segment_label.config(
            text=(f"分段记录：第 {segment.index} 段 均值 {stats['mean']:.2f}／达标 {stats['on_target_ratio']:.0%}／"
                  f"波动 {stats['volatility']:.2f} → 下一段目标 {self.training.target:.2f}")
        )

    def _run_session(self) -> None:
        artifacts = SessionRunner(SessionConfig(root=str(self.root_dir), participant=self.participant), eeg=SyntheticEEG(seed=3)).run()
        report = Path(artifacts.paths["report_md"])
        self.report_text.delete("1.0", "end")
        self.report_text.insert("1.0", report.read_text(encoding="utf-8") if report.exists() else "报告生成失败")
        self._refresh_trend()

    def _refresh_trend(self) -> None:
        records = history_module.load_records(self.root_dir)
        usable, _ = history_module.comparable(records, "sim-bsense", self.sim.srate, self.sim.channels)
        points = history_module.aggregate(usable, "focus", "week")
        self.trend_canvas.delete("all")
        if not points:
            self.trend_canvas.create_text(20, 20, anchor="w", text="暂无可比记录（先运行一次完整会话）")
            return
        width, height, pad = 860, 340, 50
        self.trend_canvas.create_line(pad, height - pad, width, height - pad, fill="#999")
        self.trend_canvas.create_line(pad, pad, pad, height - pad, fill="#999")
        step = (width - 2 * pad) / max(1, len(points) - 1)
        coords = []
        for index, point in enumerate(points):
            x = pad + index * step
            y = height - pad - (height - 2 * pad) * max(0.0, min(1.0, point["mean"]))
            coords.extend([x, y])
            self.trend_canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="#2b6cb0")
            self.trend_canvas.create_text(x, height - pad + 16, text=point["period"], font=("Consolas", 9))
        if len(coords) >= 4:
            self.trend_canvas.create_line(*coords, fill="#2b6cb0", width=2)
        self.trend_canvas.create_text(pad, 24, anchor="w", text="专注度周趋势（同一设备配置内可比）", font=("Microsoft YaHei", 11))


def launch(root: str = "var/session", participant: str = "p01") -> None:
    StudioWindow(root_dir=root, participant=participant).window.mainloop()
