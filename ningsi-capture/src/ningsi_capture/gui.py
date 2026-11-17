"""Windows desktop operator console and a separate participant task window."""

from __future__ import annotations

from contextlib import closing
from datetime import datetime
import json
import math
import os
from pathlib import Path
import queue
import re
import sqlite3
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import numpy as np

from . import __version__
from .acquisition import LslRecorder, SimRecorder, discover, simulation_source
from . import biomulti
from ningsi.acquisition.eeg import UNIT_FACTORS, unit_conflict
from ningsi.acquisition.timing import BIOMULTI_LSL_TIMING, BIOMULTI_REC_TIMING, STRICT_TIMING, PACKET_TIMING
from ningsi.models.capture_protocols import FULL_PROTOCOL, RESEARCH_PROTOCOLS, SHORT_PROTOCOL
from .clock import Clock
from .export import build_dataset, export_session, recent_window_quality
from .live_quality import QualityMonitor, describe_reasons
from .protocol import PROBE_CHOICES, ProtocolConfig, TaskRunner
from .readiness import cohort_use, montage_hint
from .storage import SessionStore, open_readonly, read_metadata

COHORT_LABELS = {"预试": "pilot", "开发组": "development", "独立测试组": "test", "未分组": "unassigned"}
PROTOCOL_CHOICES = {RESEARCH_PROTOCOLS[FULL_PROTOCOL]: ProtocolConfig,
                    RESEARCH_PROTOCOLS[SHORT_PROTOCOL]: ProtocolConfig.short, "快速模拟联调": ProtocolConfig.demo}
REC_CHOICE = "BioMulti 原生 REC 双录"
AUTO_REC = "BioMulti 自动（可选 REC）"
PACKET_CLOCK = "BioMulti 分包时钟估计"
AUTO_UNIT = "自动识别（BioMulti 推定 µV）"
TIMING_CHOICES = {"严格源时间戳": STRICT_TIMING, "分包时标兼容（限 50 毫秒）": PACKET_TIMING,
                  REC_CHOICE: BIOMULTI_REC_TIMING, AUTO_REC: "auto", PACKET_CLOCK: BIOMULTI_LSL_TIMING}
ABORT_REASONS = {"participant_window_closed": "受试者窗口已关闭", "task_focus_lost": "受试者窗口失去焦点",
                 "eeg_stream_stale": "脑电超过 3 秒未更新", "recording_stopped": "操作员停止采集",
                 "application_callback_error": "程序运行异常"}


def default_data_dir():
    base = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[3]
    return base / "var" / "capture"


def minutes_seconds(seconds):
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes:02d}:{seconds:02d}"


class ExportUnitDialog(simpledialog.Dialog):
    def __init__(self, parent, source):
        self.source = source
        super().__init__(parent, title="双录导出的 EEG 单位")

    def body(self, master):
        ttk.Label(master, text="如已核实本设备的 EEG 输出单位，可在本次导出补录。\n"
                  "尚未确认时可直接导出原始数据和任务作答。").grid(row=0, columnspan=2, sticky="w", pady=6)
        self.unit = tk.StringVar(master, "未知（仅原始及行为）")
        self.factor = tk.StringVar(master)
        self.evidence = tk.StringVar(master)
        for row, label in enumerate(("源信号单位", "ADC 倍率（µV/计数）", "确认依据（手册页或校准记录）"), start=1):
            ttk.Label(master, text=label).grid(row=row, column=0, sticky="w", pady=4)
        ttk.Combobox(master, textvariable=self.unit, state="readonly",
                     values=(self.unit.get(), *UNIT_FACTORS, "ADC")).grid(row=1, column=1, sticky="ew")
        ttk.Entry(master, textvariable=self.factor).grid(row=2, column=1, sticky="ew")
        ttk.Entry(master, textvariable=self.evidence, width=30).grid(row=3, column=1, sticky="ew")

    def validate(self):
        self.result = {}
        if self.unit.get() in (*UNIT_FACTORS, "ADC"):
            try:
                selection = {"unit": self.unit.get(), "evidence": self.evidence.get(),
                             "factor": float(self.factor.get()) if self.factor.get().strip() else None}
                biomulti.confirmed_calibration(self.source, **selection)
            except ValueError as exc:
                messagebox.showwarning("请核对单位", str(exc), parent=self)
                self.result = None
                return False
            self.result = selection
        return True

    def buttonbox(self):
        box = ttk.Frame(self)
        box.pack(pady=8)
        ttk.Button(box, text="继续导出", command=self.ok).pack(side="left", padx=4)
        ttk.Button(box, text="取消", command=self.cancel).pack(side="left", padx=4)
        self.bind("<Escape>", self.cancel)


class CaptureApp:
    def __init__(self, root, data_dir=None, *, lsl_markers=False):
        self.root = root
        self.data_dir = Path(data_dir or default_data_dir()).resolve()
        self.clock = Clock()
        self.messages = queue.Queue()
        self.store = self.recorder = self.runner = self.task_window = None
        self.last_session = None
        self.streams = []
        self.stream_key = None
        self.pressed = set()
        self.busy = False
        self.stopping = False
        self.focus_check = None
        self.started_perf = None
        self.last_preview = -math.inf
        self.quality_monitor = None
        self.callback_error = None
        self.task_close_pending = False
        self.initial_lsl_markers = bool(lsl_markers)
        self._build()
        root.report_callback_exception = self.on_callback_error
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.after(50, self.poll)

    def _build(self):
        root = self.root
        root.title(f"凝思实验采集器  Ningsi Capture {__version__}")
        initial_height = min(780, root.winfo_screenheight() - 90)
        root.geometry(f"1200x{initial_height}")
        root.minsize(1050, 690)
        root.configure(bg="#eff3f8")
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TFrame", background="#eff3f8")
        style.configure("TLabel", background="#eff3f8", foreground="#233047", font=("Microsoft YaHei UI", 10))
        style.configure("TButton", font=("Microsoft YaHei UI", 10), padding=(12, 7))
        style.configure("TEntry", padding=5, font=("Microsoft YaHei UI", 10))
        style.configure("TCombobox", padding=4, font=("Microsoft YaHei UI", 10))
        style.configure("TLabelframe", background="#eff3f8", padding=12)
        style.configure("TLabelframe.Label", background="#eff3f8", font=("Microsoft YaHei UI", 11, "bold"))
        header = tk.Frame(root, bg="#14283e", padx=20, pady=8)
        header.pack(fill="x")
        tk.Label(header, text="凝思实验采集器", fg="white", bg="#14283e", font=("Microsoft YaHei UI", 18, "bold")).pack(anchor="w")
        body = ttk.Frame(root, padding=14)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=0, minsize=390)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        left = ttk.Frame(body)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 18))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)
        right = ttk.Frame(body)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        tabs = ttk.Notebook(left)
        tabs.grid(row=0, column=0, sticky="nsew", pady=(0, 6))
        identity = ttk.Frame(tabs, padding=12)
        tabs.add(identity, text="参与者与协议")
        self.participant = tk.StringVar(value="P001")
        self.visit = tk.StringVar(value="01")
        self.cohort = tk.StringVar(value="预试")
        self.source_mode = tk.StringVar(value="LSL 真机")
        self.protocol_mode = tk.StringVar(value=RESEARCH_PROTOCOLS[FULL_PROTOCOL])
        self.controls = []
        self._field(identity, "匿名编号", self.participant, 0)
        self._field(identity, "到访编号", self.visit, 1)
        self._field(identity, "预先分组", self.cohort, 2, list(COHORT_LABELS))
        self._field(identity, "信号来源", self.source_mode, 3, ["LSL 真机", "模拟演示"])
        self._field(identity, "任务协议", self.protocol_mode, 4, list(PROTOCOL_CHOICES))
        source = ttk.Frame(tabs, padding=12)
        tabs.add(source, text="设备与单位")
        self.discover_button = ttk.Button(source, text="发现 LSL 脑电流", command=self.find_streams)
        self.discover_button.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        self.selected_stream = tk.StringVar()
        self.stream_combo = ttk.Combobox(source, textvariable=self.selected_stream, state="readonly", width=31)
        self.stream_combo.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        self.stream_combo.bind("<<ComboboxSelected>>", self.on_select_stream)
        self.controls.append(self.stream_combo)
        self.units = tk.StringVar(value="未知  仅保存原始")
        self.factor = tk.StringVar(value="")
        self.labels = tk.StringVar(value="")
        self.reference = tk.StringVar(value="unknown")
        self.timing_mode = tk.StringVar(value="严格源时间戳")
        self._field(source, "源信号单位", self.units, 2, [AUTO_UNIT, "uV", "mV", "V", "nV", "ADC 计数", "未知  仅保存原始"])
        self._field(source, "计数转 µV 倍率", self.factor, 3)
        self._field(source, "电极标签", self.labels, 4)
        self._field(source, "参考电极", self.reference, 5)
        self._field(source, "时间处理", self.timing_mode, 6, list(TIMING_CHOICES))
        ttk.Label(source, text="BioMulti 可直接按包估计采样时间。先点厂商 REC 可在导出时用设备时钟复核。", wraplength=330).grid(row=7, column=0, columnspan=2, sticky="w", pady=(7, 0))
        sync = ttk.Frame(tabs, padding=12)
        tabs.add(sync, text="外部同步")
        self.lsl_markers = tk.BooleanVar(root, value=self.initial_lsl_markers)
        self.marker_checkbox = ttk.Checkbutton(sync, text="向 LabRecorder 发布任务标记（可选）",
                                               variable=self.lsl_markers)
        self.marker_checkbox.grid(row=0, column=0, sticky="w")
        self.controls.append(self.marker_checkbox)
        self.marker_status = tk.StringVar(root, value="默认关闭；不含答案，需另用 LabRecorder 选择标记流并录制。")
        ttk.Label(sync, textvariable=self.marker_status, wraplength=330).grid(
            row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Label(sync, text="先开始连续采集，等 LabRecorder 发现 EEG 与任务标记流并开始录制，再打开受试者任务。\n\n"
                  "标记采用软件事件时刻；物理显示与设备延迟仍需独立测量。",
                  wraplength=330).grid(row=2, column=0, sticky="w", pady=(12, 0))
        path_frame = ttk.Frame(left)
        path_frame.grid(row=1, column=0, sticky="ew")
        self.path_label = ttk.Label(path_frame, text=f"保存到：{self.data_dir}", wraplength=370)
        self.path_label.pack(anchor="w")
        self.choose_path_button = ttk.Button(path_frame, text="选择保存位置", command=self.choose_path)
        self.choose_path_button.pack(anchor="w", pady=3)
        self.start_button = ttk.Button(left, text="1  开始连续采集", command=self.start_recording)
        self.start_button.grid(row=2, column=0, sticky="ew", pady=(5, 3))
        self.task_button = ttk.Button(left, text="2  打开受试者任务窗口", command=self.start_tasks, state="disabled")
        self.task_button.grid(row=3, column=0, sticky="ew", pady=3)
        self.stop_button = ttk.Button(left, text="3  停止并保存本次采集", command=self.stop_recording, state="disabled")
        self.stop_button.grid(row=4, column=0, sticky="ew", pady=3)
        preview = ttk.LabelFrame(right, text="采集状态与波形预览")
        preview.grid(row=0, column=0, sticky="nsew")
        preview.columnconfigure(0, weight=1)
        preview.rowconfigure(3, weight=1)
        self.status = tk.StringVar(value="就绪。先填写匿名编号与设备信息，再开始连续采集。")
        ttk.Label(preview, textvariable=self.status, wraplength=620).grid(row=0, column=0, sticky="w", pady=(0, 4))
        self.counter = tk.StringVar(value="尚未采集")
        wave_header = ttk.Frame(preview)
        wave_header.grid(row=1, column=0, sticky="ew")
        ttk.Label(wave_header, textvariable=self.counter, font=("Microsoft YaHei UI", 13, "bold")).pack(side="left")
        self.wave_mode = tk.StringVar(value="处理后 · µV")
        ttk.Combobox(wave_header, state="readonly", width=14, textvariable=self.wave_mode,
                     values=("处理后 · µV", "仅滤波 · µV", "原始 · 源数值")).pack(side="right")
        self.quality_text = tk.StringVar(value="实时脑电质控待采集")
        quality_label = ttk.Label(preview, textvariable=self.quality_text, wraplength=600, justify="left")
        quality_label.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        preview.bind("<Configure>", lambda e: quality_label.configure(wraplength=max(160, e.width - 30)))
        self.canvas = tk.Canvas(preview, bg="#11283c", height=125, highlightthickness=0)
        self.canvas.grid(row=3, column=0, sticky="nsew", pady=5)
        self.phase_text = tk.StringVar(value="任务尚未开始")
        self.participant_time = tk.StringVar(root)
        ttk.Label(preview, textvariable=self.phase_text, wraplength=600).grid(row=4, column=0, sticky="w", pady=(4, 0))
        exports = ttk.LabelFrame(right, text="训练资格与数据导出")
        exports.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self.eligibility_text = tk.StringVar()
        eligibility_label = ttk.Label(exports, textvariable=self.eligibility_text, wraplength=600, justify="left")
        eligibility_label.pack(anchor="w", fill="x", pady=(0, 5))
        exports.bind("<Configure>", lambda e: eligibility_label.configure(wraplength=max(160, e.width - 30)))
        for variable in (self.cohort, self.source_mode, self.selected_stream, self.labels, self.reference):
            variable.trace_add("write", self.setup_eligibility)
        self.setup_eligibility()
        buttons = ttk.Frame(exports)
        buttons.pack(fill="x")
        self.export_button = ttk.Button(buttons, text="导出本次采集", command=self.export_current, state="disabled")
        self.export_button.pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="导出已有记录", command=self.export_existing).pack(side="left", padx=6)
        ttk.Button(buttons, text="合并训练数据", command=self.dataset_dialog).pack(side="left", padx=6)
        self.log = tk.Text(right, height=3, bg="white", fg="#33485e", font=("Microsoft YaHei UI", 9),
                           relief="flat", padx=10, pady=8, state="disabled")
        self.log.grid(row=2, column=0, sticky="ew", pady=(8, 0))

    def setup_eligibility(self, *_):
        if self.store and not self.store.closed:
            return
        real = self.source_mode.get() == "LSL 真机"
        index = self.stream_combo.current()
        origin = "real" if real else "simulation"
        hint = ""
        if real and 0 <= index < len(self.streams):
            origin = self.streams[index][1]["data_origin"]
            if origin == "real":
                hint = montage_hint(self.labels.get().split(","), self.reference.get())
        role = cohort_use(COHORT_LABELS.get(self.cohort.get(), "unassigned"), origin)
        self.eligibility_text.set("\n".join(filter(None, (role, hint))))

    def _field(self, parent, label, variable, row, values=None):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4, padx=(0, 12))
        if values:
            widget = ttk.Combobox(parent, textvariable=variable, values=values, state="readonly", width=25)
        else:
            widget = ttk.Entry(parent, textvariable=variable, width=27)
        widget.grid(row=row, column=1, sticky="ew", pady=4)
        parent.columnconfigure(1, weight=1)
        self.controls.append(widget)
        return widget

    def note(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", datetime.now().strftime("%H:%M:%S") + "  " + str(text) + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def background(self, operation, finished):
        if self.busy:
            messagebox.showinfo("正在处理", "请等待当前操作完成。", parent=self.root)
            return
        self.busy = True
        def work():
            try:
                self.messages.put((finished, operation(), None))
            except Exception as exc:
                self.messages.put((finished, None, f"{type(exc).__name__}: {exc}"))
        threading.Thread(target=work, daemon=True, name="capture-operation").start()

    def find_streams(self):
        if self.store and not self.store.closed:
            return
        self.status.set("正在发现设备……")
        def done(result, error):
            if error:
                self.streams = []
                self.stream_combo["values"] = ()
                self.selected_stream.set("")
                self.status.set("未能发现设备，请核对 LSL 发布程序。")
                messagebox.showerror("设备发现失败", error, parent=self.root)
                return
            self.streams = result
            self.stream_combo["values"] = [
                f"{'【模拟】' if meta['data_origin'] != 'real' else ''}{meta['name']}  "
                f"{meta['channels']}ch / {meta['srate']:g}Hz  {meta['hostname']}  {meta['source_id']} [{i+1}]"
                for i, (_, meta) in enumerate(result)]
            self.selected_stream.set("")
            if len(result) == 1:
                self.stream_combo.current(0)
                self.on_select_stream()
            self.status.set(f"发现 {len(result)} 个脑电流。请按设备名称和标识核对选择。")
        self.background(discover, done)

    def on_select_stream(self, _event=None):
        index = self.stream_combo.current()
        if 0 <= index < len(self.streams):
            source = self.streams[index][1]
            key = tuple(source.get(field) for field in ("uid", "source_id", "hostname", "name", "channels", "srate"))
            if self.stream_key != key:
                self.labels.set(", ".join(source["channel_labels"]))
                unit = biomulti.automatic_unit(source)
                self.units.set(AUTO_UNIT if unit.get("unit_basis") == "biomulti_software_inference"
                               else unit.get("source_unit", "未知  仅保存原始"))
                self.factor.set("")
                self.reference.set("unknown")
                self.timing_mode.set(AUTO_REC if biomulti.supported(source) else
                                     "分包时标兼容（限 50 毫秒）" if re.match(r"BioMulti\s*Lite\s+EEG", source["name"], re.I)
                                     else "严格源时间戳")
                self.stream_key = key

    def choose_path(self):
        selected = filedialog.askdirectory(parent=self.root, title="选择采集数据的保存目录",
                                           initialdir=str(self.data_dir if self.data_dir.is_dir() else self.data_dir.parent))
        if selected:
            self.data_dir = Path(selected).resolve()
            self.path_label.configure(text=f"保存到：{self.data_dir}")

    def start_recording(self):
        if self.busy or self.store and not self.store.closed:
            return
        try:
            simulated = self.source_mode.get() == "模拟演示"
            if simulated:
                source, info = simulation_source(), None
            else:
                index = self.stream_combo.current()
                if not 0 <= index < len(self.streams):
                    raise ValueError("请先发现并选择真实脑电流。")
                info, source = self.streams[index]
                source = dict(source)
                unit = self.units.get()
                factor = UNIT_FACTORS.get(unit)
                if unit == "ADC 计数":
                    factor = float(self.factor.get())
                source.update(source_unit=unit, scale_to_uv=factor, reference=self.reference.get().strip() or "unknown",
                              timestamp_policy=TIMING_CHOICES[self.timing_mode.get()])
                if unit == AUTO_UNIT:
                    selection = biomulti.automatic_unit(source)
                    if not selection:
                        raise ValueError("此流没有可核实的自动单位配置，请选择设备声明的单位或仅保存原始。")
                    source.update(selection, unit_selection="automatic")
                else:
                    source["unit_selection"] = "raw_only" if factor is None else "selected"
                conflict = unit_conflict(source)
                if conflict:
                    raise ValueError(conflict)
                labels = [label.strip() for label in self.labels.get().split(",")]
                if len(labels) != source["channels"] or any(not label for label in labels):
                    raise ValueError("电极标签数量必须等于实际通道数，使用英文逗号分隔。")
                source["channel_labels"] = labels
                if source["timestamp_policy"] in ("auto", BIOMULTI_REC_TIMING):
                    if not biomulti.supported(source):
                        raise ValueError("原生 REC 双录需要本机 BioMulti Lite 的 2 通道 250 Hz 真机脑电。")
                    try:
                        source["native_rec"] = biomulti.active_record(biomulti.default_database())
                    except (OSError, ValueError, sqlite3.Error):
                        if source["timestamp_policy"] != "auto":
                            raise
                        source["timestamp_policy"] = BIOMULTI_LSL_TIMING
                    else:
                        source["timestamp_policy"] = BIOMULTI_REC_TIMING
                        source["live_timestamp_policy"] = BIOMULTI_LSL_TIMING
                if source["timestamp_policy"] == BIOMULTI_LSL_TIMING and not biomulti.supported(source):
                    raise ValueError("BioMulti 分包时钟仅适用于双通道 250 Hz 的 BioMulti Lite EEG。")
            if not simulated and self.protocol_mode.get() == "快速模拟联调":
                raise ValueError("快速联调仅限模拟演示，真人采集请选择短版或完整研究协议。")
            self.config = PROTOCOL_CHOICES[self.protocol_mode.get()]()
            self.store = SessionStore(self.data_dir, participant=self.participant.get().strip(),
                                      visit=self.visit.get().strip(), cohort=COHORT_LABELS[self.cohort.get()],
                                      source=source, protocol=self.config.as_dict(), clock=self.clock,
                                      lsl_markers=self.lsl_markers.get())
            self.callback_error = None
            self.recorder = SimRecorder(self.store) if simulated else LslRecorder(self.store, info)
            self.runner = None
            pilot = source["kind"] == "lsl" and self.store.metadata["cohort"] == "pilot"
            self.task_button.configure(text="2  开始预试任务（不计入训练）" if pilot else "2  打开受试者任务窗口")
            self.stop_button.configure(text="3  停止并保存本次采集")
            if pilot:
                self.phase_text.set("收到 4 秒脑电后可开始预试。任务按键和探针作答会随原始脑电一起保存。")
            elif source.get("native_rec"):
                self.phase_text.set("双录已绑定原生 REC。请在两边同时录制期间进行任务，保存 REC 后导出核对。")
            else:
                self.phase_text.set("任务尚未开始")
            self.started_perf = self.clock.now().perf
            self.last_session = self.store.path
            self.recorder.start()
            self.quality_monitor = QualityMonitor(self.last_session)
            self.quality_monitor.start()
            self.quality_text.set("等待首个四秒窗，随后显示原始工频占比与近期合格窗数。")
            self.start_button.configure(state="disabled")
            self.stop_button.configure(state="normal")
            self.export_button.configure(state="disabled")
            self.discover_button.configure(state="disabled")
            self.choose_path_button.configure(state="disabled")
            for control in self.controls:
                control.configure(state="disabled")
            self.status.set(("模拟演示：" if simulated or source.get("data_origin") != "real" else "LSL 连续采集：") + "正在接收并写入原始样本。")
            self.note(f"{self.protocol_mode.get()}　保存到 {self.store.path}")
            if self.lsl_markers.get():
                marker = self.store.markers.snapshot()
                if marker["enabled"]:
                    self.note(f"可选任务标记流：{marker['stream_name']}。请另用 LabRecorder 选择 EEG 与此流；发布不代表已录入 XDF。")
                else:
                    self.note("任务标记未启用：当前不是 LSL 本地时钟。原始采集和任务事件继续保存。")
            if source.get("native_rec"):
                self.note(f"已绑定 BioMulti 原生 REC #{source['native_rec']['record_id']}。")
            elif source.get("timestamp_policy") == BIOMULTI_LSL_TIMING:
                self.note("使用 BioMulti 分包时钟估计。原始时标和样本按接收顺序保存。")
            if source.get("unit_basis") == "biomulti_software_inference":
                self.note("EEG 单位按本机 BioMulti 软件推定为 µV，倍率 1。依据随记录保存。")
        except Exception as exc:
            messagebox.showerror("无法开始采集", str(exc), parent=self.root)

    def start_tasks(self):
        if not self.store or self.store.closed or self.runner is not None or self.stopping:
            return
        snapshot = self.store.snapshot()
        last = snapshot["last_received_perf"]
        if (snapshot["samples"] < 4 * self.store.metadata["source"]["srate"]
                or last is None or self.clock.now().perf - last >= 3):
            messagebox.showinfo("等待信号", "请确认信号持续更新并已收到至少4秒的样本。", parent=self.root)
            return
        if self.store.metadata["source"]["kind"] == "lsl":
            source = self.store.metadata["source"]
            pilot = self.store.metadata["cohort"] == "pilot"
            paired = source.get("timestamp_policy") == BIOMULTI_REC_TIMING
            if not pilot and not paired and source.get("scale_to_uv") is None:
                messagebox.showwarning("请先确认 EEG 单位", "当前仅保存原始数值，无法建立有效基线。\n\n"
                                       "请先停止并保存，在“设备与单位”按设备说明填写源单位，然后新建采集。"
                                       "不要填写光源驱动电压。", parent=self.root)
                return
            reasons = recent_window_quality(self.store.path)
            preflight = (self.quality_monitor.latest.latest_preflight(self.clock.now(), last)
                         if self.quality_monitor else None)
            check = {"reasons": list(reasons), "cohort": self.store.metadata["cohort"]}
            if preflight is not None:
                check["signal_preflight"] = preflight.as_dict()
            blocking = [r for r in reasons if not (paired and r in ("native_clock_pending", "unit_unconfirmed"))]
            if not pilot and paired:
                try:
                    if biomulti.active_record(biomulti.default_database()) != source["native_rec"]:
                        raise ValueError("原生 REC 已切换，请停止保存并重新开始双录。")
                except (OSError, ValueError, sqlite3.Error) as exc:
                    messagebox.showwarning("原生 REC 未就绪", str(exc), parent=self.root)
                    return
            if blocking and not pilot:
                detail = describe_reasons(blocking)
                self.store.event("task_signal_precheck", {**check, "continued": False})
                message = (f"最近 4 秒脑电未通过导出质控：{detail}。\n\n"
                           "原始数据仍在保存。请先处理提示的问题，再点击“打开受试者任务窗口”复检。\n"
                           "更改设备单位或时间处理方式，需要先停止保存，再新建采集。")
                if preflight is not None:
                    message += "\n\n" + preflight.text()
                messagebox.showwarning("脑电预检未通过", message, parent=self.root)
                return
            last = self.store.last_received
            if self.store.closed or self.stopping or last is None or self.clock.now().perf - last >= 3:
                self.status.set("脑电接收已中断，请确认设备仍在发布。")
                return
            if pilot and reasons:
                self.note("开始预试，保存任务与作答。脑电质控未通过：" + describe_reasons(reasons) + "。")
            elif paired:
                self.note("开始原生双录任务。设备对时和脑电质控将在导出时按匹配样本复核。")
            self.store.event("task_signal_precheck", {**check, "continued": True})
        self.task_window = tk.Toplevel(self.root)
        win = self.task_window
        win.title("凝思实验任务  受试者窗口")
        win.geometry("1000x740")
        win.minsize(850, 650)
        win.configure(bg="#112436")
        win.protocol("WM_DELETE_WINDOW", self.close_tasks)
        self.task_title = tk.Label(win, bg="#112436", fg="#b7c7d8", font=("Microsoft YaHei UI", 17))
        self.task_title.pack(pady=(15, 2))
        tk.Label(win, textvariable=self.participant_time, bg="#112436", fg="#b7c7d8",
                 font=("Microsoft YaHei UI", 11)).pack(pady=(0, 6))
        self.task_text = tk.Label(win, bg="#112436", fg="white", font=("Microsoft YaHei UI", 31),
                                  wraplength=920, justify="center")
        self.task_text.pack(expand=True, fill="both", padx=30)
        self.choice_frame = tk.Frame(win, bg="#112436")
        self.task_footer = tk.Label(win, bg="#112436", fg="#c3d2df", font=("Microsoft YaHei UI", 13), wraplength=930)
        self.task_footer.pack(pady=(12, 15))
        tk.Label(win, bg="#112436", fg="#7892a8", text="F11 全屏／退出全屏    关闭窗口可中止实验", font=("Microsoft YaHei UI", 9)).pack(pady=(0, 10))
        win.bind("<F11>", lambda _e: win.attributes("-fullscreen", not win.attributes("-fullscreen")))
        win.bind("<KeyPress>", self.on_key)
        win.bind("<KeyRelease>", self.on_key_release)
        win.bind("<FocusOut>", self.on_focus_out)
        win.focus_force()
        self.pressed.clear()
        self.runner = TaskRunner(self.store, self.render_task, self.config, self.clock)
        self.task_button.configure(state="disabled")
        self.runner.start()
        self.update_task_status()
        self.note("实验任务已开始；受试者窗口不显示专注分数或奖励。")

    def render_task(self, view):
        if self.task_window is None or not self.task_window.winfo_exists():
            return self.clock.now()
        self.task_title.configure(text=view["title"])
        self.task_text.configure(text=view["text"], font=("Microsoft YaHei UI", 20 if view["choices"] else 31))
        self.task_footer.configure(text=view["footer"])
        for child in self.choice_frame.winfo_children():
            child.destroy()
        if view["choices"]:
            self.choice_frame.pack(before=self.task_footer, fill="x", padx=55, pady=5)
            for index, text in enumerate(PROBE_CHOICES, 1):
                tk.Label(self.choice_frame, text=f"{index}   {text}", bg="#112436", fg="white",
                         anchor="w", font=("Microsoft YaHei UI", 13)).pack(fill="x", pady=3)
        else:
            self.choice_frame.pack_forget()
        self.task_window.update_idletasks()
        return self.clock.now()  # Software paint estimate, not physical display onset.

    def on_key(self, event):
        stamp = self.clock.now()
        key = event.keysym.lower()
        if key.startswith("kp_") and key[3:] in "1234567" and len(key) == 4:
            key = key[3:]
        if key in self.pressed:
            return
        self.pressed.add(key)
        if self.runner and key in ("space", "1", "2", "3", "4", "5", "6", "7"):
            self.runner.key(key, stamp, getattr(event, "time", None))
            self.update_task_status()

    def on_key_release(self, event):
        key = event.keysym.lower()
        if key.startswith("kp_") and key[3:] in "1234567" and len(key) == 4:
            key = key[3:]
        self.pressed.discard(key)
        if self.runner and self.runner.running:
            self.store.event("key_up", {"key": event.keysym.lower(), "phase": self.runner.phase.get("name")})

    def on_focus_out(self, _event):
        if self.task_close_pending:
            return
        if self.focus_check:
            self.root.after_cancel(self.focus_check)
        def check():
            self.focus_check = None
            if self.task_close_pending or not self.runner or not self.runner.running or self.task_window is None:
                return
            focus = self.root.focus_get()
            if focus is None or focus.winfo_toplevel() != self.task_window:
                self.store.event("task_focus_lost", {"phase": self.runner.phase.get("name")})
                self.runner.abort("task_focus_lost")
                self.update_task_status()
                self.note("任务窗口失去焦点，实验已中止；脑电仍在连续保存。")
        self.focus_check = self.root.after(120, check)

    def close_tasks(self, force=False):
        if self.runner and self.runner.running:
            if not force:
                self.task_close_pending = True
                try:
                    confirmed = messagebox.askyesno("中止本次任务？", "关闭后本次任务将中止，已采集数据保留。\n"
                                                    "重新实验需要停止保存并新建采集。是否中止？",
                                                    parent=self.task_window, default="no")
                finally:
                    self.task_close_pending = False
                if not confirmed:
                    self.task_window.focus_force()
                    return
            self.runner.abort("participant_window_closed")
        if self.task_window is not None:
            self.task_window.destroy()
            self.task_window = None
        self.update_task_status()

    def update_task_status(self):
        runner = self.runner
        if not runner:
            return
        participant = ""
        if runner.finished:
            message = ("预试任务已完成。停止并保存后可导出原始脑电和作答，本次预试不计入训练。"
                       if self.store.metadata["cohort"] == "pilot" else "任务已完成。请停止并保存，再导出本次采集。")
        elif runner.running:
            now = self.clock.now().perf
            elapsed = minutes_seconds(int(max(0., now - runner.phase_started.perf)))
            remaining = runner.phase_remaining(now)
            if remaining is None:
                end = "不限时，作答后继续"
            else:
                low, high = (minutes_seconds(math.ceil(value)) for value in remaining)
                label = "预计剩余" if runner.phase["kind"] == "sart" else "剩余"
                end = f"{label} {low}" if low == high else f"{label} {low}–{high}"
            stage = f"阶段 {runner.phase_index + 1}/{len(runner.plan)}"
            timing = f"已用 {elapsed} · {end}"
            participant = f"{stage} · {timing}"
            message = f"{stage}  {runner.phase['title']} · {timing}"
        else:
            reason = ABORT_REASONS.get(runner.abort_reason, runner.abort_reason or "任务未完成")
            message = f"任务已中止：{reason}。请停止保存，再新建采集。"
            self.task_button.configure(text="2  任务已中止，请先停止保存", state="disabled")
            self.stop_button.configure(text="3  保存中止记录，准备重新采集")
        if self.phase_text.get() != message:
            self.phase_text.set(message)
        if self.participant_time.get() != participant:
            self.participant_time.set(participant)

    def stop_recording(self):
        if self.stopping or not self.store or self.store.closed:
            return
        self.stopping = True
        if self.quality_monitor:
            self.quality_monitor.stop()
        self.quality_text.set("实时脑电质控已停止")
        self.task_button.configure(state="disabled")
        self.stop_button.configure(state="disabled")
        if self.runner and self.runner.running:
            self.runner.abort("recording_stopped")
        self.recorder.stop()
        self.status.set("正在停止接收并提交磁盘文件……")
        def finish():
            if self.recorder.is_alive() or (self.quality_monitor and self.quality_monitor.is_alive()):
                self.root.after(100, finish)
                return
            try:
                failure = self.recorder.error or self.callback_error
                state = "failed" if failure else "completed" if self.runner and self.runner.finished else "aborted"
                self.store.close(state, failure or ("" if state == "completed" else "protocol_not_completed"))
                if failure:
                    self.status.set(f"采集已停止：{failure} 已写入的原始记录可以导出。")
                elif state == "completed":
                    self.status.set("任务与采集已完成。请导出本次采集并核对质量。")
                else:
                    self.status.set("采集已保存，任务未完成。原始记录可导出，本次不进入训练。")
                self.eligibility_text.set("任务完成 · 请导出核定训练与测试行数。" if state == "completed"
                                          else "本场任务未完成 · 训练与测试均为 0 行。")
                self.note(f"采集已结束：{self.store.sample_count} 个样本，{self.store.event_count} 个事件。")
                self.export_button.configure(state="normal")
            except Exception as exc:
                self.status.set("保存收尾失败；保留原始数据库，可用恢复导出。")
                messagebox.showerror("保存失败", str(exc), parent=self.root)
            self.close_tasks(force=True)
            self.stopping = False
            self.start_button.configure(state="normal")
            self.discover_button.configure(state="normal")
            self.choose_path_button.configure(state="normal")
            for control in self.controls:
                control.configure(state="readonly" if isinstance(control, ttk.Combobox) else "normal")
        self.root.after(100, finish)

    def export_current(self):
        if self.last_session:
            self.export_path(self.last_session)

    def export_existing(self):
        selected = filedialog.askdirectory(parent=self.root, title="选择含 recording.sqlite3 的采集目录",
                                           initialdir=str(self.data_dir))
        if selected:
            self.export_path(Path(selected))

    def export_path(self, session):
        if self.store and not self.store.closed or self.busy:
            messagebox.showinfo("请先停止采集", "停止连续采集后再导出。", parent=self.root)
            return
        parent = filedialog.askdirectory(parent=self.root, title="选择导出父目录（自动新建独立导出文件夹）",
                                        initialdir=str(self.data_dir))
        if not parent:
            return
        try:
            with closing(open_readonly(session)) as connection:
                metadata = read_metadata(connection)
        except (OSError, ValueError, sqlite3.Error, TypeError) as exc:
            messagebox.showerror("无法读取记录", str(exc), parent=self.root)
            return
        source = metadata["source"]
        calibration = None
        if (source.get("timestamp_policy") == BIOMULTI_REC_TIMING and source.get("scale_to_uv") is None
                and metadata["cohort"] in ("development", "test")
                and not ("unit_selection" not in source and biomulti.local_publisher(source)
                         and biomulti.automatic_unit(source))):
            calibration = ExportUnitDialog(self.root, source).result
            if calibration is None:
                return
        output = Path(parent) / (Path(session).name + "_export_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.status.set("正在导出数据；大记录的特征计算可能需要一些时间。")
        def done(result, error):
            if error:
                self.status.set("导出失败，原始记录保留。")
                messagebox.showerror("导出失败", error + "\n异常退出的记录可使用命令行 export --recover。", parent=self.root)
                return
            self.status.set(f"导出完成：{result['raw_samples']} 样本，{result['probes_total']} 探针。")
            counts = result["training_readiness"]
            baseline = "通过" if result["baseline_valid"] else "未通过"
            eligibility = (f"{cohort_use(counts['cohort'], result['source_origin'])} · "
                           f"导出训练 {result['train_eligible']} 行，测试 {result['test_eligible']} 行"
                           f"（专注 {counts['eligible_on_task']}，内部走神 {counts['eligible_internal_mind_wandering']}）\n"
                           f"基线{baseline}（{counts['baseline_usable']}/{counts['baseline_windows']} 窗） · "
                           f"正式提示 {counts['formal_probes']} · 自报专注 {counts['on_task']}，"
                           f"内部走神 {counts['internal_mind_wandering']} · 脑电与二分类配对 {counts['paired']}")
            self.eligibility_text.set(eligibility)
            self.note(f"导出到：{output}；真人开发行 {result['train_eligible']}，测试行 {result['test_eligible']}。")
            comparison = result["probe_comparison"]
            align = result.get("native_alignment")
            alignment_text = ""
            if align:
                alignment_text = (f"\n原生时钟已匹配 {align['matched_packets']} 包，合格脑电窗 "
                                  f"{result['windows_usable']}/{result['windows_total']}。"
                                  if align["status"] == "matched" else f"\n原生时标未核对：{align['reason']}")
            messagebox.showinfo("导出完成", f"文件位置：\n{output}\n\n{eligibility}"
                                f"{alignment_text}\n\n250 Hz 双模型同时评分 {comparison['scored']} 个探针，"
                                f"其中 {comparison['comparable']} 个有二分类回答。"
                                "\n两组分数、任务时间、质控及作答见 probe_comparison.csv。", parent=self.root)
        if not source.get("preprocessing_policy"):
            self.note("旧记录将用新版预处理复算，结果写入新导出目录。")
        self.background(lambda: export_session(session, output, calibration=calibration, reprocess=True), done)

    def dataset_dialog(self):
        if self.busy or self.store and not self.store.closed:
            return
        parent = filedialog.askdirectory(parent=self.root, title="选择存放多份导出记录的父目录",
                                        initialdir=str(self.data_dir))
        if not parent:
            return
        paths = sorted({path.parent for path in Path(parent).rglob("export.json")})
        if not paths:
            messagebox.showinfo("未找到导出记录", "请先导出采集记录，再选择其父目录。", parent=self.root)
            return
        output = Path(parent) / ("dataset_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        def done(result, error):
            if error:
                messagebox.showerror("未生成数据集", error, parent=self.root)
            else:
                self.note(f"数据集：{output}")
                messagebox.showinfo("数据集已生成", f"开发行：{result['development_rows']}\n测试行：{result['test_rows']}\n排除行：{result['excluded_rows']}\n\n{output}", parent=self.root)
        self.background(lambda: build_dataset(paths, output), done)

    def draw_wave(self, values, *, processed=False, mask=None):
        canvas = self.canvas
        canvas.delete("all")
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if not len(values) or width < 10:
            message = "等待已对时的完整处理窗（2 秒历史 + 4 秒信号）" if processed else "等待连续样本"
            canvas.create_text(width/2, height/2, text=message, fill="#88a4bc", font=("Microsoft YaHei UI", 11))
            return
        data = np.asarray(values)
        channels = min(4, data.shape[1])
        if mask is not None:
            edges = np.diff(np.r_[False, mask, False].astype(np.int8))
            for begin, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                canvas.create_rectangle(begin*width/len(mask), 0, end*width/len(mask), height,
                                        outline="", fill="#503f45")
        for index in range(channels):
            signal = data[:, index]
            finite = signal[np.isfinite(signal)]
            center = (index + .5) * height / channels
            canvas.create_line(0, center, width, center, fill="#244057")
            label = self.store.metadata["source"]["channel_labels"][index] if self.store else f"Ch{index+1}"
            if not finite.size:
                reason = "本窗全部屏蔽" if processed else "无有限样本"
                canvas.create_text(10, center-10, text=f"{label}  {reason}", anchor="w", fill="#a9c2d6")
                continue
            base = float(np.median(finite))
            scale = max(float(np.max(np.abs(finite-base))), 1e-12)
            edges = np.diff(np.r_[False, np.isfinite(signal), False].astype(np.int8))
            for begin, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                points = np.unique(np.r_[np.arange(begin, end, max(1, len(signal)//600)), end-1])
                if len(points) < 2:
                    continue
                coords = np.column_stack((points / max(1, len(signal)-1) * width,
                                          center - (signal[points]-base)/scale*height/channels*.32)).ravel()
                canvas.create_line(*coords, fill=("#62d5d0", "#82baff", "#c7b0ed", "#f0cf85")[index], width=1.3)
            unit = "µV" if processed else "源数值"
            canvas.create_text(10, center-height/channels*.37, text=f"{label}  ±{scale:.1f} {unit}",
                               anchor="w", fill="#a9c2d6")

    def on_callback_error(self, error_type, error, _traceback):
        self.callback_error = f"{error_type.__name__}: {error}"
        self.note("程序异常，实验将停止并保留数据：" + self.callback_error)
        if self.store and not self.store.closed:
            try:
                self.store.event("application_callback_error", {"error": self.callback_error})
                if self.runner and self.runner.running:
                    self.runner.abort("application_callback_error")
            except Exception:
                if self.runner:
                    self.runner.running = False
            self.stop_recording()
        self.status.set("程序异常，已停止本次实验。请查看日志；已写入的数据保留。")

    def poll(self):
        try:
            self._poll_once()
        except Exception:
            self.on_callback_error(*sys.exc_info())
        finally:
            try:
                self.root.after(20, self.poll)
            except tk.TclError:
                pass

    def _poll_once(self):
        try:
            while True:
                callback, result, error = self.messages.get_nowait()
                self.busy = False
                callback(result, error)
        except queue.Empty:
            pass
        if self.store and not self.store.closed:
            snapshot = self.store.snapshot()
            markers = snapshot["lsl_markers"]
            if markers["requested"]:
                if markers["state"] in ("failed", "unavailable"):
                    self.marker_status.set("任务标记未发布/已中断：" + markers["error_code"] + "；原始采集继续。")
                else:
                    self.marker_status.set(f"标记已发送 {markers['published']}，排队 {markers['pending']}，丢弃 {markers['dropped']}；不代表已保存 XDF。")
            elapsed = self.clock.now().perf - self.started_perf
            fresh = snapshot["last_received_perf"] is not None and self.clock.now().perf - snapshot["last_received_perf"] < 3
            self.counter.set(f"{snapshot['samples']:,} 个样本    {snapshot['events']:,} 个事件    {int(elapsed)//60:02d}:{int(elapsed)%60:02d}")
            if not self.runner:
                ready = fresh and snapshot["samples"] >= 4 * self.store.metadata["source"]["srate"] and not self.stopping
                self.task_button.configure(state="normal" if ready else "disabled")
            if self.recorder.error and not self.stopping:
                self.note("采集异常：" + self.recorder.error)
                self.stop_recording()
            if self.runner and self.runner.running:
                if not fresh:
                    self.runner.abort("eeg_stream_stale")
                    self.note("脑电超过3秒未更新，实验已中止；采集文件保留。")
                else:
                    self.runner.tick()
            self.update_task_status()
            if self.clock.now().perf - self.last_preview >= .25:
                if self.quality_monitor and not self.stopping:
                    latest = self.quality_monitor.latest
                    self.quality_text.set(latest.operator_text(
                        self.clock.now(), snapshot["last_received_perf"], self.store.metadata["source"]["data_origin"] != "real"))
                    source = self.store.metadata["source"]
                    role = cohort_use(self.store.metadata["cohort"], source["data_origin"])
                    details = latest.probe_summary
                    if not details and source["data_origin"] == "real":
                        details = montage_hint(source["channel_labels"], source.get("reference"))
                    self.eligibility_text.set("\n".join(filter(None, (role, details))))
                mode = self.wave_mode.get()
                if mode != "原始 · 源数值":
                    latest = self.quality_monitor.latest if self.quality_monitor and not self.stopping else None
                    current = latest and latest.latest_preflight(self.clock.now(), snapshot["last_received_perf"])
                    wave, mask = (), None
                    if current and latest.cleaned_samples:
                        wave = latest.cleaned_samples
                        if mode == "仅滤波 · µV":
                            mask = ~np.isfinite(np.asarray(wave)[:, 0])
                            wave = latest.filtered_samples
                    self.draw_wave(wave, processed=True, mask=mask)
                else:
                    self.draw_wave(snapshot["latest"])
                self.last_preview = self.clock.now().perf

    def on_close(self):
        if self.busy:
            messagebox.showinfo("正在导出", "当前操作完成后可关闭程序。", parent=self.root)
            return
        if self.store and not self.store.closed:
            self.stop_recording()
            def wait():
                if self.stopping:
                    self.root.after(100, wait)
                else:
                    self.root.destroy()
            self.root.after(100, wait)
        else:
            self.root.destroy()


def run(data_dir=None, *, lsl_markers=False):
    root = tk.Tk()
    CaptureApp(root, data_dir, lsl_markers=lsl_markers)
    root.mainloop()
