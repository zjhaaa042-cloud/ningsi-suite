"""Read-only, operator-facing quality checks on committed capture data."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import threading

import numpy as np

from . import biomulti
from .export import analysis_windows
from .readiness import Preflight, RECENT_WINDOWS
from .storage import open_readonly, read_metadata
from ningsi import config
from ningsi.acquisition.timing import BIOMULTI_LSL_TIMING, BIOMULTI_REC_TIMING, BIOMULTI_CONFIRM_SAMPLES
from ningsi.models.research_features import aggregate_probe
from ningsi.signal.baseline import build_baseline


QUALITY_TEXT = {
    "unit_unconfirmed": "源单位未确认", "sample_rate_below_100hz": "采样率低于 100 Hz",
    "signal_too_short": "样本不足 4 秒", "clock_unsynchronized": "LSL 时钟尚未同步",
    "nonmonotonic_timestamps": "时间戳重复或倒退", "timestamp_gap_over_100ms": "时间戳间断超过 100 毫秒",
    "sample_rate_mismatch_over_5pct": "实测采样率偏差超过 5%", "non_finite_signal": "信号含非有限值",
    "flat_channel": "通道近似平线", "amplitude": "幅度过大", "channel_span": "通道间幅度差过大",
    "eog": "低频伪迹偏高", "emg": "高频伪迹偏高", "no_data": "缺少样本",
    "no_usable_window": "无合格窗", "insufficient_baseline_duration": "合格时长不足",
    "native_clock_pending": "设备时标待原生 REC 核对",
    "packet_clock_unresolved": "分包时钟未能覆盖完整窗口",
    "preprocessing_warmup": "等待 2 秒滤波历史和 4 秒脑电",
    "sample_gap_in_filter_history": "滤波历史内样本不连续",
    "ocular_transient": "疑似眼动或眨眼片段", "muscle_burst": "疑似短时肌电",
    "motion_transient": "波形突变", "contact_dropout": "通道出现连续相同读数",
    "insufficient_baseline_windows": "合格窗数不足", "insufficient_baseline_valid_ratio": "合格窗比例不足",
    "inconsistent_baseline_data": "基线窗数据不一致",
}


def describe_reasons(reasons):
    return "、".join(QUALITY_TEXT.get(reason, reason) for reason in reasons)


def describe_window(verdict, source):
    if verdict.ok:
        return "通过"
    details = []
    fields = {"flat_channel": ("std_uv", "标准差", config.QUALITY["flat_std_uv"], True),
              "amplitude": ("peak_uv", "峰值", config.QUALITY["amp_max_uv"], False),
              "emg": ("emg_rel", "高频占比", config.QUALITY["emg_rel_max"], False),
              "eog": ("eog_rel", "低频占比", config.QUALITY["eog_rel_max"], False)}
    labels = source["channel_labels"]
    for reason in verdict.reasons:
        if reason not in fields:
            continue
        field, title, threshold, below = fields[reason]
        for channel in verdict.metrics.get("channels", []):
            value = channel[field]
            if (value < threshold if below else value > threshold):
                index = channel["channel"]
                name = labels[index] if index < len(labels) else f"Ch{index + 1}"
                reading = f"{value:.1%}" if field.endswith("_rel") else f"{value:g} µV"
                details.append(f"{name} {title} {reading}")
    if "channel_span" in verdict.reasons:
        details.append(f"通道间最大差 {verdict.metrics['span_uv']:g} µV")
    detail = "，".join(details[:3])
    if len(details) > 3:
        detail += f" 等 {len(details)} 项"
    reading = "未通过 · " + describe_reasons(verdict.reasons) + (f"（{detail}）" if detail else "")
    actions = []
    if "unit_unconfirmed" in verdict.reasons:
        actions.append("按设备说明确认 EEG 单位")
    clock_errors = {"nonmonotonic_timestamps", "timestamp_gap_over_100ms", "sample_rate_mismatch_over_5pct"}
    if (clock_errors.intersection(verdict.reasons) and biomulti.supported(source)
            and source.get("timestamp_policy") != BIOMULTI_LSL_TIMING):
        actions.append("下次先点 BioMulti REC，再开始凝思双录")
    if actions:
        reading += "\n" + "。".join(actions) + "。"
    return reading


@dataclass(frozen=True)
class QualitySnapshot:
    reading: str = ""
    window_end: float | None = None
    received_perf: float | None = None
    summary: str = "基线待采集"
    confirmation_sec: float = 0.
    preflight: Preflight | None = None
    probe_summary: str = ""
    cleaned_samples: tuple = ()
    filtered_samples: tuple = ()

    def latest_preflight(self, now, last_received):
        if (last_received is not None and now.perf - last_received < 3
                and self.received_perf is not None and now.perf - self.received_perf < 3 + self.confirmation_sec):
            return self.preflight
        return None

    def operator_text(self, now, last_received, simulation):
        origin = "模拟源" if simulation else "设备"
        if last_received is None:
            status = "等待脑电"
        elif now.perf - last_received >= 3:
            status = "信号超过 3 秒未更新"
        elif not self.reading:
            status = "等待完整 4 秒窗"
        elif self.received_perf is not None and now.perf - self.received_perf >= 3 + self.confirmation_sec:
            status = "等待最新窗口质控"
        elif self.window_end is not None and abs(now.lsl - self.window_end) >= 3 + self.confirmation_sec:
            status = f"窗口时间与当前时钟相差超过 {3 + self.confirmation_sec:g} 秒" + (" · " + self.reading if self.reading.startswith("未通过") else "")
        else:
            status = self.reading
        preflight = self.latest_preflight(now, last_received)
        lines = [f"{origin} · 4 秒窗质控  {status}"]
        if preflight is not None:
            lines.append(preflight.text())
        lines.append(self.summary)
        return "\n".join(lines)


@dataclass(frozen=True)
class ProbeWindow:
    usable: bool


class CaptureQuality:
    """Consume each aligned window and relevant protocol event once per session."""

    def __init__(self):
        self.next_index = 0
        self.last_event = 0
        self.baseline_start = self.baseline_end = None
        self.calibration = []
        self.probe_windows = []
        self.formal_blocks = {}
        self.probes = {}
        self.reading = ""
        self.end = self.received_perf = None
        self.recent = deque(maxlen=RECENT_WINDOWS)
        self.recent_spectral = deque(maxlen=RECENT_WINDOWS)
        self.preflight = None
        self.cleaned_samples = ()
        self.filtered_samples = ()

    def _read_events(self, connection):
        for row in connection.execute(
                "SELECT event_id,kind,time_lsl,payload FROM events WHERE event_id>? AND "
                "kind IN ('phase_start','phase_end','probe_onset','probe_response','probe_cancelled') "
                "ORDER BY event_id", (self.last_event,)):
            self.last_event = row["event_id"]
            payload = json.loads(row["payload"])
            phase, kind, when = payload.get("phase"), row["kind"], row["time_lsl"]
            if phase == "baseline_calibration":
                if kind == "phase_start":
                    self.baseline_start = when
                elif kind == "phase_end":
                    self.baseline_end = when
            if payload.get("practice") or not payload.get("block"):
                continue
            if kind == "phase_start" and payload.get("spec", {}).get("kind") == "sart":
                self.formal_blocks[phase] = (payload["block"], when)
            elif phase in self.formal_blocks:
                if kind == "probe_onset":
                    self.probes[payload["probe_id"]] = {"phase": phase, "onset": when, "choice": None}
                elif kind in ("probe_response", "probe_cancelled") and payload["probe_id"] in self.probes:
                    self.probes[payload["probe_id"]]["choice"] = payload.get("choice") if kind == "probe_response" and not payload.get("timeout") else None

    def update(self, connection, metadata):
        source = metadata["source"]
        paired = source.get("timestamp_policy") == BIOMULTI_REC_TIMING
        if paired and source.get("live_timestamp_policy") == BIOMULTI_LSL_TIMING:
            source = {**source, "timestamp_policy": BIOMULTI_LSL_TIMING}
            metadata = {**metadata, "source": source}
        elif paired:
            status = "双录中，设备时标待原生 REC 核对"
            if source.get("scale_to_uv") is None:
                status += "，源单位待确认"
            return QualitySnapshot(status, summary="保存两份记录后导出，可查看脑电窗和基线质控。")
        self._read_events(connection)
        rate = source["srate"]
        lag = BIOMULTI_CONFIRM_SAMPLES if source.get("timestamp_policy") == BIOMULTI_LSL_TIMING else 0
        last_index = connection.execute("SELECT max(sample_index) FROM samples").fetchone()[0]
        catching_up = False
        if last_index is not None:
            size, hop = round(4 * rate), max(1, round(2 * rate))
            newest_end = size - 1 + (last_index + 1 - size - lag) // hop * hop
            if last_index + 1 >= size and newest_end >= self.next_index:
                # Bound catch-up work so stopping capture can release its DB reader promptly.
                first_end = size - 1 if self.next_index == 0 else self.next_index - 1 + hop
                end = min(newest_end, first_end + hop * 4)
                windows = analysis_windows(connection, metadata, first_index=self.next_index, stop_index=end + 1,
                                           keep_traces=True, include_spectral=True)
                for row in windows:
                    window = row["window"]
                    self.recent.append(window.usable)
                    spectral = row.get("descriptive_spectrum") or {}
                    self.recent_spectral.append(spectral.get("status") in ("full", "partial"))
                    if (self.baseline_start is not None and row["start"] is not None and row["end"] is not None
                            and row["start"] >= self.baseline_start and (self.baseline_end is None or row["end"] <= self.baseline_end)):
                        self.calibration.append((row["end"], window))
                    self.probe_windows.append({"start": row["start"], "end": row["end"], "features": row["features"],
                                               "window": ProbeWindow(window.usable)})
                if windows:
                    row = windows[-1]
                    spectrum = row.get("descriptive_spectrum")
                    self.preflight = Preflight(tuple(source["channel_labels"][:2]), row["raw_50hz_ratios"],
                                               rate, sum(self.recent), len(self.recent), row["sample_end_index"],
                                               row.get("filtered_50hz_ratios", ()), row.get("raw_60hz_ratios", ()),
                                               row.get("filtered_60hz_ratios", ()),
                                               spectral_passed=sum(self.recent_spectral) if spectrum is not None else None,
                                               descriptive_spectrum=spectrum)
                    wave = row.get("filtered_uv")
                    self.cleaned_samples = ()
                    self.filtered_samples = ()
                    if wave is not None:
                        clean = np.where(row["artifact_mask"], np.nan, wave[:4]).T
                        self.cleaned_samples = tuple(clean)
                        self.filtered_samples = tuple(wave[:4].T)
                    self.reading = describe_window(row["window"].quality, source)
                    audit = row["timestamp_adjustment"]
                    if audit["policy"] == BIOMULTI_LSL_TIMING:
                        if "clock_unsynchronized" in row["window"].quality.reasons:
                            self.reading = self.reading.replace("LSL 时钟尚未同步", "分包时钟未能覆盖完整窗口")
                        self.reading += " · 分包时钟估计"
                        if source.get("unit_basis") == "biomulti_software_inference":
                            self.reading += " · µV（软件推定）"
                        if paired:
                            self.reading += " · 导出时复核 REC"
                    elif audit["applied"]:
                        self.reading += f" · 分包时标估计，最大修正 {audit['max_shift_sec'] * 1000:.1f} 毫秒"
                    self.end = row["end"]
                    self.received_perf = connection.execute("SELECT received_perf FROM samples WHERE sample_index=?",
                                                            (row["sample_end_index"],)).fetchone()[0]
                    self.next_index = row["sample_end_index"] + 1
                catching_up = self.next_index <= newest_end

        calibration = [window for end, window in self.calibration if self.baseline_end is None or end <= self.baseline_end]
        baseline = build_baseline(calibration, source["name"], rate)
        baseline_done = self.baseline_end is not None and self.end is not None and self.end >= self.baseline_end
        valid = baseline_done and baseline.valid
        if self.baseline_start is None:
            summary = "任务开始后记录睁眼基线"
        else:
            state = ("采集中" if self.baseline_end is None else "等待末窗" if not baseline_done
                     else "通过" if valid else "未通过")
            ratio = baseline.n_windows / len(calibration) if calibration else 0.
            summary = (f"睁眼基线 {state} · 合格 {baseline.n_windows}/{len(calibration)} 窗"
                       f"（{ratio:.1%}，要求 ≥{config.QUALITY['valid_ratio_min']:.0%}），"
                       f"覆盖 {baseline.quality.get('usable_seconds', 0):g}/{config.BASELINE_PROTOCOL['min_usable_sec']:g} 秒")
            if baseline_done and not valid:
                summary += "（" + describe_reasons(baseline.reasons) + "）"
                if metadata.get("cohort") in ("development", "test"):
                    summary += " · " + ("保存 REC 后复核" if paired else "正式特征不可用，建议调整后新建采集")
        if catching_up:
            return QualitySnapshot(self.reading, self.end, self.received_perf, "正在追赶已写入的脑电窗口，基线与区块统计待更新")
        probe_summary = ""
        if self.formal_blocks:
            phase = next(reversed(self.formal_blocks))
            block = self.formal_blocks[phase][0]
            current = eeg_ok = block_paired = on_task = wandering = 0
            paired_on_task = paired_wandering = 0
            for probe in self.probes.values():
                choice = probe["choice"]
                on_task += choice == 1
                wandering += choice in (3, 5)
                # Match export's window coverage and phase boundary without using the answer to judge EEG.
                available = valid and not aggregate_probe(
                    self.probe_windows, probe["onset"], baseline_valid=valid,
                    phase_start=self.formal_blocks[probe["phase"]][1])["reasons"]
                pair = available and choice in (1, 3, 5)
                paired_on_task += available and choice == 1
                paired_wandering += available and choice in (3, 5)
                if probe["phase"] == phase:
                    current += 1
                    eeg_ok += available
                    block_paired += pair
            summary += f"\nSART 区块 {block} · 已提示 {current}，脑电可用 {eeg_ok}，同时有二分类回答 {block_paired}"
            probe_summary = (f"本场正式提示 {len(self.probes)} 个 · 自报专注 {on_task}，内部走神 {wandering}"
                             f" · 脑电配对专注 {paired_on_task}，内部走神 {paired_wandering}（导出核定）")
        return QualitySnapshot(self.reading, self.end, self.received_perf, summary, lag / rate,
                               self.preflight, probe_summary, self.cleaned_samples, self.filtered_samples)


class QualityMonitor(threading.Thread):
    def __init__(self, path):
        super().__init__(name="capture-live-quality", daemon=True)
        self.path = path
        self.done = threading.Event()
        self.latest = QualitySnapshot()

    def run(self):
        state = CaptureQuality()
        while not self.done.is_set():
            try:
                connection = open_readonly(self.path)
                try:
                    updated = state.update(connection, read_metadata(connection))
                finally:
                    connection.close()
            except Exception as exc:
                updated = QualitySnapshot(f"质控暂不可用（{type(exc).__name__}）", summary="等待重新计算")
            self.latest = updated
            self.done.wait(.5)

    def stop(self):
        self.done.set()
