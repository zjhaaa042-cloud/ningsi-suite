"""数据源选择：**只用真实信号**，没有信号就如实显示"无信号"。

产品规则（2026-10-07 明确）：
- 没有脑机信号时**不再自动使用仿真源**——不静默、不降级，界面直接显示"无信号"；
- 仿真源只在**显式选择**时使用（`device="sim-bsense"`），且必须在界面与报告里标注；
- 没开会话也能看真实信号：`/api/devices/preview` 直接读实时流（见 `PreviewStream`）。

真实链路的代码路径始终存在且优先；"当前是仿真"这个事实必须如实出现在会话的 source 字段、
SSE 事件与报告里，不做静默替换。
"""

from __future__ import annotations

import time
import math
import re
import os
import json
from dataclasses import dataclass

import numpy as np

from ningsi import config
from ningsi.acquisition.simulate import SyntheticEEG
from ningsi.acquisition.timing import (
    BIOMULTI_CONFIRM_SAMPLES, BIOMULTI_LSL_RECOVERY_TIMING, BIOMULTI_LSL_TIMING, STRICT_TIMING,
    analysis_timestamps,
)
from ningsi.signal.live_artifacts import (
    SPEC as ARTIFACT_SPEC, ACCEL_LIMITS, accel_unit, channel_layout, check_motion_reference,
    process_live_artifacts, voltage_scales, with_artifact_evidence,
)
from ningsi.signal.preprocess import preprocess as run_preprocess

SIM_SOURCE = "sim-bsense"

# 真机信号调理口径（只作用于真实设备；仿真源按原样出数，一切仿真数值不变）。
# 文档 8.2 的零相位处理链（0.5 Hz 去漂移 → 50/60 Hz 陷波 → 45 Hz 低通）此前只被它自己的
# 单元测试引用，实时链路把**未调理**的原始值直接送进质检与频谱，于是设备的直流偏置
# （实测均值 -2.3e5 µV，见 `_analysis/lsl-hardware/units.json`）让每一窗都判为
# amplitude / channel_span 越界，可用窗比例恒为 0（`lsl-hardware/ab2.json` 的 A 组）。
CONDITION_SPEC = "acq-condition-v1"
DC_REMOVAL = "per_channel_dc_removal"

#: 内置仿真 outlet 的 source_id（`acquisition/sim_outlet.py`）。它走的是真实 LSL 传输，
#: 但数据是算出来的，界面与报告必须写清楚"这不是真实设备"，否则演示时会误导。
SIM_OUTLET_SOURCE_ID = "ningsi-sim-outlet-v1"


def _is_sim_outlet(source_id, name="") -> bool:
    text = f"{source_id or ''} {name or ''}".lower()
    return bool(re.search(r"(^|[^a-z])(sim|simulation|synthetic|demo)([^a-z]|$)", text))


def _known_biomulti_packet_source(descriptor) -> bool:
    return (descriptor is not None and descriptor.nominal_srate == 250 and descriptor.channel_count == 2
            and re.fullmatch(r"BioMulti Lite EEG-[0-9a-f]{6}", descriptor.name, flags=re.I) is not None
            and not _is_sim_outlet(descriptor.source_id, descriptor.name))


def _assessment_clock_reasons(stamps, rate, expected):
    """Check the exact analysis history with Capture's clock tolerances."""
    if not np.isfinite(stamps).all():
        return ["clock_unsynchronized"]
    reasons = []
    diffs = np.diff(stamps)
    if np.any(diffs <= 0):
        reasons.append("nonmonotonic_timestamps")
    if np.any(diffs > .1):
        reasons.append("timestamp_gap_over_100ms")
    elif np.any(diffs > 1.75 / rate):
        reasons.append("sample_gap_in_filter_history")
    if 1 < expected <= len(stamps):
        span = stamps[-1] - stamps[-expected]
        if abs(span * rate / (expected - 1) - 1) > .05:
            reasons.append("sample_rate_mismatch_over_5pct")
    return reasons


def _assessment_signal_context(snapshot, rate, wanted, expected, *, recover_packets):
    """Resolve time for exactly the EEG data passed to the Studio filter."""
    stamps = np.asarray(snapshot["timestamps"], dtype=float)
    samples = snapshot["samples"]
    latest = snapshot["total_samples"]
    raw = stamps[-wanted:]
    reasons = _assessment_clock_reasons(raw, rate, expected)
    audit = {"policy": STRICT_TIMING, "applied": False, "max_shift_sec": 0., "reasons": reasons}
    if not reasons or not recover_packets:
        return samples[:, -wanted:], raw, latest, audit

    # The publisher timestamps the last sample of its ten-sample packet. The
    # existing bounded reconstruction needs the following one-second epoch;
    # report the earlier cursor rather than credit the confirmation samples.
    epoch = round(rate)
    end = (latest - BIOMULTI_CONFIRM_SAMPLES) // epoch * epoch
    offset = latest - len(stamps)
    stop = end - offset
    if stop < expected:
        return samples[:, -wanted:], raw, latest, audit
    first = max(0, stop - wanted)
    estimate, _ = analysis_timestamps(stamps, rate, BIOMULTI_LSL_RECOVERY_TIMING, first_index=offset)
    timed = estimate[first:stop]
    problems = _assessment_clock_reasons(timed, rate, expected)
    paired = np.isfinite(timed) & np.isfinite(stamps[first:stop])
    max_shift = float(np.max(np.abs(timed[paired] - stamps[first:stop][paired]))) if paired.any() else 0.
    return samples[:, first:stop], timed, end, {"policy": BIOMULTI_LSL_RECOVERY_TIMING, "applied": not problems,
                                               "max_shift_sec": max_shift, "reasons": problems}


def _lsl_note(descriptor: dict, name: str, srate: float) -> str:
    """数据源说明文案：只写"这是什么"，不写调理链。

    调理口径原来逐行重复在每一条数据源上（界面里同一句话出现 6 遍）——现在统一放在
    `CONDITION_NOTE` 里，由界面作为表格脚注显示一次，报告里也有「采集调理」行。
    """
    source_id = descriptor.get("source_id")
    if source_id == SIM_OUTLET_SOURCE_ID:
        head = f"内置仿真 LSL 流（非真实设备，source_id={SIM_OUTLET_SOURCE_ID}）"
    elif _is_sim_outlet(source_id, name):
        head = "模拟 LSL 流（非真实设备）"
    else:
        head = "真实 LSL 流"
    kind = str(descriptor.get("kind") or descriptor.get("stream_type") or "").lower()
    # 预览是"只接流不等样本"启动的，此时描述符还没解析出来：如实说"尚未收到样本"
    if not descriptor.get("channel_count"):
        return f"{head}：{name}（尚未收到样本）"
    return (f"{head}（{kind or '未知类型'}）：{name}"
            f"（{descriptor.get('channel_count')} 通道，{srate:.0f} Hz，"
            f"标签 {descriptor.get('channel_labels')}）")


def condition_eeg_samples(raw, expected: int, srate: float, *, labels=(), units=(),
                          research_scale=1.0, operator_scale=None, observed_srate=None,
                          personal_blink=None, fragment_masks=False):
    """Apply Studio's EEG filtering and artifact gate to a contiguous sample block."""
    pick, _ = channel_layout(labels, len(raw))
    scales = voltage_scales(units, len(raw), research_scale, pick, operator_scale)
    values = raw * np.asarray(scales)[:, None]
    blink_evidence = {}
    blink_mask = None
    if len(pick) == 2:
        from ningsi.signal.blinks import process_blinks
        selected_labels = [labels[i] if i < len(labels) else f"Ch{i+1}" for i in pick]
        try:
            blink = (personal_blink(values[pick]) if personal_blink is not None else
                     process_blinks(values[pick], srate, channel_labels=selected_labels))
            values[pick] = blink.cleaned_uv
            blink_evidence = blink.as_dict(start_index=max(0, values.shape[1] - expected))
            blink_mask = blink.mask
        except ValueError as exc:
            blink_evidence = {"state": "invalid", "reasons": ["blink_calibration_mismatch"],
                              "detail": str(exc), "corrected_events": 0}
    values -= values.mean(axis=1, keepdims=True)
    filtered = []
    log = None
    for channel in values:
        wave, log = run_preprocess(channel, srate)
        filtered.append(wave)
    wave, evidence = process_live_artifacts(np.vstack(filtered), raw, srate, expected, labels, units,
                                            research_scale, operator_scale, blink_evidence=blink_evidence,
                                            fragment_masks=fragment_masks, blink_mask=blink_mask)
    conditioning = {
        "spec": CONDITION_SPEC, "dc_removal": DC_REMOVAL,
        "chain": log.as_dict() if log is not None else {},
        "srate": round(float(srate), 3),
        "observed_srate": None if observed_srate is None else round(float(observed_srate), 3),
        "context_samples": int(values.shape[1]), "window_samples": int(wave.shape[1]),
        "expected_samples": int(expected), "scale_to_uv": research_scale,
        "channel_scale_to_uv": [scales[i] for i in evidence["eeg_source_indices"]],
        "artifact_spec": ARTIFACT_SPEC, "blink_control": blink_evidence,
    }
    return wave, evidence, conditioning


#: 真机信号调理口径的一句话说明（界面表格脚注 / 报告「采集调理」行共用）。
CONDITION_NOTE = (f"真实设备信号按 {CONDITION_SPEC} 调理：逐通道去直流 → 0.5 Hz 去漂移 → "
                  f"50/60 Hz 陷波 → 45 Hz 低通。按 {ARTIFACT_SPEC} 检查瞬态与肌电；"
                  "有已知单位的同步眼电参考和足够历史时可校正眼电。")


@dataclass
class SourceInfo:
    key: str
    kind: str            # lsl | sim | replay
    srate: float
    channels: int
    device: str
    note: str
    engine: object

    @property
    def channel_labels(self):
        return list(getattr(self.engine, "channel_labels", ()))

    @property
    def is_real(self) -> bool:
        if self.kind != "lsl":
            return False
        descriptor = self.engine.descriptor_dict() if hasattr(self.engine, "descriptor_dict") else {}
        return not _is_sim_outlet(descriptor.get("source_id"), self.device)

    def as_dict(self) -> dict:
        return {"key": self.key, "kind": self.kind, "srate": self.srate,
                "channels": self.channels, "device": self.device, "note": self.note,
                "real": self.is_real}


def sim_source_info() -> dict:
    """仿真源的描述（只在调用方**显式**选择仿真时使用，不再作为缺省）。"""
    return SourceInfo(SIM_SOURCE, "sim", 250.0, 1, SIM_SOURCE,
                      "仿真脑电源（仅用于演示与自动测试，必须在界面与报告中标注）", None).as_dict()


#: LSL 扫描结果的短时缓存：`/api/health` 会被前端高频轮询，每次真扫 0.2~1 秒会把接口拖慢。
#: 3 秒 TTL 对"插上设备后界面很快出现信号"足够灵敏，又不至于让 health 变慢。
_PROBE_CACHE: dict = {"at": 0.0, "sources": None, "probe": 0.0}


def cached_sources(probe_seconds: float = 0.4, ttl: float = 3.0) -> list[dict]:
    """带 TTL 的 `list_available()`：给健康检查/顶栏这类高频调用用。"""
    now = time.monotonic()
    cached = _PROBE_CACHE.get("sources")
    if cached is not None and (now - float(_PROBE_CACHE.get("at") or 0.0)) < ttl:
        return cached
    sources = list_available(probe_seconds)
    _PROBE_CACHE.update({"at": now, "sources": sources, "probe": probe_seconds})
    return sources


#: 已知的**非脑电**流类型。一台 BioMultiLite 会同时推 6~7 条流
#: （EEG / Metric / FNIRS / HeartRate / General Metric / Motion / Marker），
#: 但本产品的质检、频谱、专注/放松/负荷指标与 SART/PVT 判定**全部基于脑电通道**：
#: 拿 FNIRS（近红外）或 Metric 跑一遍，照样会算出一堆看着很像的数字，却没有意义。
#: 所以建会话时对"已知非脑电"的流直接 409（除非显式 allow_non_eeg），
#: 类型为空/未知的流不拦（可能只是采集端没写 type 字段）。
NON_EEG_STREAM_KINDS = frozenset({
    "fnirs", "metric", "motion", "heart_rate", "general_metric", "marker",
})

#: 中文名（错误信息与界面共用，避免各写一份对不上）
STREAM_KIND_LABELS = {
    "eeg": "脑电（EEG）",
    "fnirs": "近红外（FNIRS）",
    "metric": "设备指标（Metric）",
    "motion": "运动（Motion）",
    "heart_rate": "心率（HeartRate）",
    "general_metric": "通用指标（General Metric）",
    "marker": "事件标记（Marker）",
    "sim": "内置仿真",
}


def stream_kind_label(kind: object) -> str:
    key = str(kind or "").strip().lower()
    return STREAM_KIND_LABELS.get(key, key or "未知类型")


def is_eeg_stream(row: dict) -> bool:
    """这条流是不是脑电。类型为空/未知时返回 False（调用方需自行决定是否放行）。"""
    return str(row.get("stream_kind") or "").strip().lower() == "eeg"


def rejected_kind(row: dict) -> str | None:
    """返回被拒绝的原因类型（已知非脑电），否则 None。"""
    kind = str(row.get("stream_kind") or "").strip().lower()
    return kind if kind in NON_EEG_STREAM_KINDS else None


def detected_source(sources: list[dict] | None = None, probe_seconds: float = 0.4) -> dict | None:
    """自动选择已发现并标记为脑电的真实 LSL 流。只有辅助流或模拟源时返回 None。"""
    rows = sources if sources is not None else cached_sources(probe_seconds)
    return next((row for row in rows if row.get("kind") == "lsl" and is_eeg_stream(row)
                 and not row.get("simulated") and row.get("real") is not False
                 and str(row.get("key", "")).startswith("lsl:")), None)


def detected_source_fields(sources: list[dict] | None = None, probe_seconds: float = 0.4) -> dict:
    """给 `/api/health`、`/api/overview` 的 `source`/`source_kind`/`source_note` 三件套。

    检测不到实时源时三者都是 None —— 顶栏要能如实写成「数据来源：无信号」，
    而不是回落到"新会话将使用仿真源"（那正是用户要求去掉的行为）。
    """
    row = detected_source(sources, probe_seconds)
    if row is None:
        return {"source": None, "source_kind": None, "source_note": None, "has_real_source": False}
    return {"source": row.get("key"), "source_kind": row.get("kind"),
            "source_note": row.get("note"), "has_real_source": not row.get("simulated")}


def default_device(sources: list[dict] | None = None, probe_seconds: float = 0.4) -> str | None:
    """新会话的缺省数据源键；没有检测到实时源时返回 None（调用方必须报错，不能改用仿真）。"""
    row = detected_source(sources, probe_seconds)
    return row.get("key") if row else None


def list_available(probe_seconds: float = 2.0) -> list[dict]:
    """列出可用数据源：**真实 LSL 流在前**，仿真源放最后并标记 `explicit_only`。

    顺序很关键：前端用"第一个 real 的源"作为缺省选择，仿真源不能排在前面，
    否则缺省又变成了仿真。
    """
    note = "仿真脑电源（仅用于演示与自动测试，必须在界面与报告中标注）"
    sim_entry = {**SourceInfo(SIM_SOURCE, "sim", 250.0, 1, SIM_SOURCE, note, None).as_dict(),
                 "explicit_only": True}
    sources: list[dict] = []

    try:
        from ningsi_studio.acquisition import lsl as lsl_module
    except Exception as error:                       # noqa: BLE001 - 导入失败也要能列出仿真源
        sim_entry["hardware_note"] = f"LSL 模块不可用：{error}"
        return [sim_entry]

    try:
        found = lsl_module.probe_streams(timeout=max(0.2, min(5.0, probe_seconds)))
    except lsl_module.LslUnavailable as error:
        sim_entry["hardware_note"] = str(error)
        return [sim_entry]
    except Exception as error:                       # noqa: BLE001 - 扫描失败不应让接口失败
        sim_entry["hardware_note"] = f"LSL 流扫描失败：{error}"
        return [sim_entry]

    for item in found:
        if not item.get("supported") or not item.get("kind"):
            sources.append({
                **item,
                "key": f"unsupported:{item.get('name', '')}",
                "device": item.get("name", ""),
                "note": item.get("label", "不支持的流类型"),
                "real": False,
            })
            continue
        name = item["name"]
        simulated = _is_sim_outlet(item.get("source_id"), name)
        from ningsi_studio.domain import bsense_device
        try:
            device_configuration = bsense_device.configuration(item)
        except (ValueError, TypeError, OverflowError):
            # Preserve conflicting live metadata for preflight to explain; it
            # must not receive a successful documented-parameter badge.
            device_configuration = None
        sources.append({
            "key": f"lsl:{name}",
            "kind": "lsl",
            "srate": float(item.get("nominal_srate") or 0.0) or 250.0,
            "channels": int(item.get("channel_count") or 0) or 1,
            "device": name,
            "note": _lsl_note(item, name, float(item.get("nominal_srate") or 0.0) or 250.0),
            "real": not simulated,
            "simulated": simulated,
            "explicit_only": simulated,
            "stream_type": item.get("stream_type"),
            # 归一后的流类型（eeg / metric / heart_rate / motion …）：缺省选源要按它排序，
            # 否则同一台设备的多条流里可能先拿到 Metric（还没有样本）而不是 EEG。
            "stream_kind": item.get("kind"),
            # 给界面直接用：中文类型名 + 是不是脑电 + 本产品能不能用它跑检测
            # （一台 BioMultiLite 推 6~7 条流，只有 EEG 那条能用，用户会问"选哪个"）
            "stream_kind_label": stream_kind_label(item.get("kind")),
            "is_eeg": str(item.get("kind") or "").strip().lower() == "eeg",
            "usable": str(item.get("kind") or "").strip().lower() not in NON_EEG_STREAM_KINDS,
            "channel_labels": item.get("channel_labels", []),
            "declared_units": item.get("declared_units", []),
            "nominal_srate": item.get("nominal_srate"),
            "source_id": item.get("source_id", ""),
            "device_configuration": device_configuration,
        })

    # 脑电流排到最前：前端"缺省预填第一个 real 源"与 detected_source() 都据此取值
    sources.sort(key=lambda row: 0 if str(row.get("stream_kind") or "").lower() == "eeg" else 1)

    if not sources:
        sim_entry["hardware_note"] = (
            "未发现 LSL 流（当前没有脑机信号）。请先启动采集端（如 BioMultiLite / BSense-R）；"
            "仅做演示/自测时，可显式选择仿真源 sim-bsense，"
            "或先用内置仿真流验证链路：python -m ningsi_studio simulate-outlet"
        )
    sources.append(sim_entry)
    return sources


class ManagedLslSource:
    """常驻缓冲的 LSL 数据源：`window()` 从缓冲切片，不会因为设备停顿而卡住采集线程。

    - `start()` 立刻返回是否就绪；未就绪时 `window()` 返回空窗（0 列），
      上层质量门控会把它判为不可用窗并记录原因，而不是让会话挂死；
    - 设备插拔由 LiveStreamManager 自动重连，`status()`/`errors()` 暴露细节给界面。
    - **真机调理**：`window()` 返回的每一窗都先做"逐通道去直流 → 文档 8.2 四级链"，
      见 `_condition()`；调理记录随 `status()["conditioning"]` 与报告一起留痕。
    """

    #: 调理上下文长度（秒）。去漂移高通是 0.5 Hz，只滤 4 秒窗会让零相位镜像填充的
    #: 边缘效应吃掉大半个窗；因此多取一段历史一起滤、再截取最后一窗。上限同时也是
    #: 显示通道的跨度上限（纯 Python 双二阶的成本与时长线性相关）。
    condition_seconds = 6.0

    def __init__(self, stream_name: str, *, device: str, srate: float = 250.0,
                 channels: int = 1, buffer_seconds: float = 60.0,
                 ready_timeout: float = 15.0, fill_timeout: float = 6.0,
                 stale_timeout: float = 3.0, research_timing: bool = False) -> None:
        from ningsi_studio.acquisition.lsl import LiveStreamManager

        self.device = device
        self.stream_name = stream_name
        self._research_timing = bool(research_timing)
        kwargs = {"stale_timeout": stale_timeout}
        if research_timing:
            # Clock correction is allowed; timestamp smoothing/monotonization must
            # not conceal gaps and backwards samples from the research QC gate.
            def inlet_factory(info, seconds):
                from pylsl import StreamInlet, proc_clocksync
                return StreamInlet(info, max_buflen=max(1, math.ceil(seconds)), recover=False,
                                   processing_flags=proc_clocksync)
            kwargs["inlet_factory"] = inlet_factory
        self._manager = LiveStreamManager(buffer_seconds=buffer_seconds, wanted_kinds=("eeg",),
                                          wanted_name=stream_name, **kwargs)
        self._motion_manager = None
        pairing = os.environ.get("NINGSI_MOTION_REFERENCE")
        if pairing:
            profile = json.loads(pairing)
            if not isinstance(profile, dict) or not profile.get("eeg"):
                raise ValueError("NINGSI_MOTION_REFERENCE 需要 EEG 流名、Motion 流名、三轴索引与单位。")
            if profile["eeg"] == stream_name:
                axes, unit = profile.get("axes"), accel_unit(profile.get("unit"))
                if (not isinstance(axes, list) or len(axes) != 3 or any(type(i) is not int or i < 0 for i in axes)
                        or len(set(axes)) != 3 or unit not in ACCEL_LIMITS or not isinstance(profile.get("motion"), str)
                        or not profile["motion"].strip() or profile["motion"] == stream_name):
                    raise ValueError("动作参考须指定不同的流名、三个不同的非负轴索引和 g/mg/m/s² 单位。")
                self._motion_axes = axes
                self._motion_unit = unit
                self._motion_manager = LiveStreamManager(buffer_seconds=buffer_seconds, wanted_kinds=("motion",),
                                                         wanted_name=profile["motion"], **kwargs)
        self._ready = False
        self._ready_timeout = float(ready_timeout)
        self.fill_timeout = float(fill_timeout)
        self.stale_timeout = float(stale_timeout)
        self._buffer_seconds = float(buffer_seconds)
        self._srate = float(srate)
        self._channels = int(channels)
        self._last_condition: dict = {}
        self.research_scale_to_uv = 1.0
        self.research_profile = None
        self.operator_eeg_scale_to_uv = None
        self.blink_binding = None
        self._blink_buffer_id = None
        self._blink_connection_generation = None


    # ------------------------------------------------------------------ 生命周期
    def start(self, *, wait: bool = True) -> bool:
        try:
            self._manager.start()
            if self._motion_manager:
                self._motion_manager.start()
        except Exception as error:                   # noqa: BLE001
            raise RuntimeError(f"启动 LSL 采集失败：{error}") from error
        if wait:
            self._ready = self._manager.wait_for("eeg", timeout=self._ready_timeout, min_samples=1)
        else:
            self._ready = False
        if self._ready:
            descriptor = self._manager.descriptor("eeg")
            if descriptor is not None:
                self._srate = descriptor.nominal_srate or self._srate
                self._channels = descriptor.channel_count or self._channels
        return self._ready

    def stop(self) -> None:
        self._manager.stop()
        if self._motion_manager:
            self._motion_manager.stop()

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def srate(self) -> float:
        descriptor = self._manager.descriptor("eeg")
        return float(descriptor.nominal_srate) if descriptor and descriptor.nominal_srate else self._srate

    @property
    def channels(self) -> int:
        descriptor = self._manager.descriptor("eeg")
        return len(channel_layout(descriptor.channel_labels, descriptor.channel_count)[0]) if descriptor else self._channels

    @property
    def channel_labels(self):
        descriptor = self._manager.descriptor("eeg")
        if descriptor is None:
            return []
        labels = descriptor.channel_labels
        return [labels[i] if i < len(labels) else f"Ch{i+1}"
                for i in channel_layout(labels, descriptor.channel_count)[0]]

    def descriptor_dict(self) -> dict:
        descriptor = self._manager.descriptor("eeg")
        return descriptor.as_dict() if descriptor is not None else {}

    def conditioning(self) -> dict:
        """最近一窗实际用到的调理口径（供报告与排查留痕）。"""
        report = dict(self._last_condition)
        if self.research_profile and self.research_profile.get("preprocessing_policy"):
            from ningsi.signal.conditioning import description
            report["research_analysis"] = description(self.srate)
        return report

    def status(self) -> dict:
        payload = self._manager.status()
        payload["conditioning"] = self.conditioning()
        return payload

    def errors(self) -> dict:
        return self._manager.errors()

    def assessment_timing_anchor(self) -> dict | None:
        """Return the acquisition cursor for the beginning of an assessment phase."""
        buffer = self._manager.buffer("eeg")
        return None if buffer is None else buffer.assessment_timing_anchor()

    def model_snapshot(self) -> dict:
        buffer = self._manager.buffer("eeg")
        if buffer is None:
            return {"research_timing": self._research_timing, "age": None}
        snapshot = buffer.model_snapshot(round(20 * self.srate))
        snapshot["research_timing"] = self._research_timing
        return snapshot

    # ------------------------------------------------------------------ 调理
    def _condition(self, data, expected: int, observed_srate: float | None = None, *, return_artifacts=False,
                   fragment_masks=False):
        """真机调理：逐通道去直流 → 文档 8.2 零相位处理链 → 截取最后一窗。

        为什么必须先去直流：设备把带几百 mV 电极偏置的原始值直接推出来，而 0.5 Hz
        高通的零相位实现（镜像填充 + 正反各滤一次）在这么大的阶跃上会产生远大于信号的
        启动瞬态——实测"只滤波"反而把 peak 从 2.5e5 压到 3.9e4 µV、eog 相对功率升到 1.0
        （`_analysis/lsl-hardware/ab2.json` 的 B 组），比不滤更糟。先扣均值再滤波，
        同一段数据的 peak 落到 44~53 µV、可用（同文件的 E/F 组）。
        """
        raw = np.atleast_2d(np.asarray(data, dtype=float))
        if raw.shape[1] < 2:  # A single new sample cannot be zero-phase filtered.
            empty = np.empty((self.channels, 0))
            return (empty, {"reasons": ["preprocessing_warmup"]}) if return_artifacts else empty
        descriptor = self._manager.descriptor("eeg")
        labels = descriptor.channel_labels if descriptor else ()
        units = descriptor.declared_units if descriptor else ()
        if descriptor is not None:
            from ningsi_studio.domain.eeg_units import resolve_calibration
            documented = resolve_calibration(None, descriptor.as_dict())
            if documented:
                if self.operator_eeg_scale_to_uv not in (None, documented["scale_to_uv"]):
                    raise ValueError("当前 EEG 换算倍率与 BSense-R 手册配置不一致")
                self.operator_eeg_scale_to_uv = documented["scale_to_uv"]
        y, evidence, self._last_condition = condition_eeg_samples(
            raw, expected, self.srate, labels=labels, units=units,
            research_scale=self.research_scale_to_uv, operator_scale=self.operator_eeg_scale_to_uv,
            observed_srate=observed_srate,
            fragment_masks=fragment_masks,
            personal_blink=self._apply_personal_blink if self.blink_binding else None,
        )
        return (y, evidence) if return_artifacts else y

    def _apply_personal_blink(self, values):
        from ningsi.signal.blinks import process_blinks
        from .blink_control import verify_binding

        calibration, _, _, labels, montage = verify_binding(self.blink_binding, self)
        buffer = self._manager.buffer("eeg")
        if buffer is None:
            raise ValueError("设备已断开，请重新连接并校准")
        buffer_id, generation = buffer.buffer_id, buffer.connection_generation
        if self._blink_buffer_id is not None and self._blink_buffer_id != buffer_id:
            raise ValueError("设备已重连，请重新做个人眨眼校准")
        if self._blink_connection_generation is not None and self._blink_connection_generation != generation:
            raise ValueError("设备已重连，请重新做个人眨眼校准")
        self._blink_buffer_id = buffer_id
        self._blink_connection_generation = generation
        return process_blinks(values, self.srate, calibration, channel_labels=labels, montage_id=montage)

    def analysis_window(self, state="rest", *, assessment_timing=False, **kwargs):
        """取同一批样本及其伪迹处理记录，避免并发显示帧覆盖评分依据。"""
        if assessment_timing:
            kwargs.update(wait=False, assessment_timing=True)
        return self.window(state, return_artifacts=True, **kwargs)

    # ------------------------------------------------------------------ 取数
    def window(self, state: str = "rest", seconds: float | None = None, artifact: bool = False,
               wait: bool = True, return_artifacts: bool = False, *, assessment_timing: bool = False,
               fragment_masks: bool = False):
        """返回 (通道, 采样点)。

        三条真实设备特有的规则：
        - **有限等待**：刚连上时缓冲还没攒满一窗，最多等 `fill_timeout` 秒
          （显示通道用 `wait=False` 关掉，避免 10 FPS 的推帧被拖住）；
        - **新鲜度检查**：如果最近一次收到样本已经过去超过 `stale_timeout` 秒（设备停了、
          线掉了、采集端崩了），就返回空窗，让上层把这一窗判为不可用。
          否则会一直拿缓冲里的旧数据当"新数据"算指标，指标看着正常其实是冻结的。
        - **调理**：返回值一律经过 `_condition()`，质检与频谱不会再看到未去直流的原始值；
        - **按点数取窗**：窗口长度用样点数（`seconds × 声明采样率`）兑现，而不是用 LSL 时间戳
          切片——本设备的时间戳会漂，同一"4 秒窗"按时间戳能拿到 2.7~6.8 秒的数据
          （见 `_analysis/lsl-hardware/window_geometry.json`），窗长失真会连带 Welch 分辨率失真。
        """
        duration = float(seconds or config.WINDOW_SEC)
        options = {"return_artifacts": True} if return_artifacts else {}
        if fragment_masks:
            options["fragment_masks"] = True
        expected = int(round(duration * self.srate))
        descriptor = self._manager.descriptor("eeg")
        has_refs = bool(descriptor and len(self.channel_labels) < descriptor.channel_count)
        context = max(duration, min(max(self.condition_seconds, duration + 2 if has_refs else duration),
                                    self._buffer_seconds))
        wanted = int(round(context * self.srate))
        deadline = time.monotonic() + (self.fill_timeout if wait else 0.0)
        while True:
            stats = self._manager.status().get("streams", {}).get("eeg", {})
            age = stats.get("seconds_since_last")
            fresh = age is None or age <= self.stale_timeout
            buffer = self._manager.buffer("eeg")
            profile = self.research_profile
            aligned_research = (return_artifacts and profile and profile.get("preprocessing_policy")
                                and duration == config.WINDOW_SEC)
            eeg_times = ()
            timing_audit = None
            if assessment_timing and buffer is not None:
                recover_packets = self._research_timing and _known_biomulti_packet_source(buffer.descriptor)
                count = round(20 * self.srate) if aligned_research or recover_packets else wanted
                snapshot = buffer.model_snapshot(count)
                generation = snapshot["connection_generation"]
                fresh = snapshot["age"] is not None and snapshot["age"] <= self.stale_timeout
                if aligned_research:
                    data = snapshot["samples"][:, -wanted:]
                else:
                    data, timestamps, end, timing_audit = _assessment_signal_context(
                        snapshot, self.srate, wanted, expected, recover_packets=recover_packets)
                    eeg_times = timestamps[-expected:]
            else:
                generation = buffer.connection_generation if buffer is not None else None
                if return_artifacts and self._motion_manager and buffer is not None:
                    eeg = buffer.model_snapshot(wanted)
                    eeg_times = eeg["timestamps"][-expected:]
                    data = eeg["samples"] if eeg_times else np.zeros((self.channels, 0))
                else:
                    data = (buffer.samples_count(wanted) if buffer is not None
                            else np.zeros((self.channels, 0), dtype=float))
            # 就绪判据只看"最后一窗"那部分，与调理前完全一致（多取的历史不参与判据）。
            window_part = data[:, -expected:] if (expected > 0 and data.shape[1] > expected) else data
            if (data.size and fresh and window_part.shape[1] >= expected) or time.monotonic() >= deadline:
                if not fresh or not data.size:
                    empty = np.zeros((self.channels, 0), dtype=float)
                    return (empty, {}) if return_artifacts else empty
                if aligned_research:
                    # The active human model and its baseline must see the same
                    # recording-aligned, timestamp-checked data as its training exports.
                    from .human_inference import snapshot_window
                    snap = {**snapshot, "research_timing": self._research_timing} if assessment_timing else self.model_snapshot()
                    if not snap.get("research_timing") or snap.get("age") is None or snap["age"] > self.stale_timeout:
                        return np.zeros((self.channels, 0)), {"reasons": ["signal_stale"]}
                    lag = BIOMULTI_CONFIRM_SAMPLES if profile.get("timestamp_policy") == BIOMULTI_LSL_TIMING else 0
                    hop = round(config.STEP_SEC * self.srate)
                    end = expected + (snap["total_samples"] - lag - expected) // hop * hop
                    row = snapshot_window(snap, profile, end, traces=True)
                    if row is None:
                        return np.zeros((self.channels, 0)), {"reasons": ["preprocessing_warmup"]}
                    eeg_times = row["analysis_timestamps"]
                    evidence = {"spec": profile["preprocessing_policy"],
                                "sample_end_index": end-1, "reasons": list(row["window"].quality.reasons)}
                    if fragment_masks and row.get("artifact_mask") is not None:
                        evidence["fragment_artifact_mask"] = row["artifact_mask"]
                    wave = row["filtered_uv"]
                    # The frozen human-model pipeline stays training-aligned.
                    # Additional raw dual-channel detection may only veto a contaminated window.
                    if len(self.channel_labels) == 2:
                        from ningsi.signal.blinks import process_blinks
                        snap_raw = np.asarray(snap["samples"], dtype=float)
                        desc = snap["descriptor"]
                        picks, _ = channel_layout(desc.get("channel_labels", ()), len(snap_raw))
                        scales = voltage_scales(desc.get("declared_units", ()), len(snap_raw),
                                                self.research_scale_to_uv, picks, self.operator_eeg_scale_to_uv)
                        # Use the same end index as the filtered model window.
                        # The buffer may already hold nearly two seconds more.
                        stop = end - (snap["total_samples"] - snap_raw.shape[1])
                        blink = process_blinks(snap_raw[picks, stop-expected:stop] * np.asarray(scales)[picks, None],
                                               self.srate, channel_labels=self.channel_labels)
                        evidence["blink_control"] = {**blink.as_dict(), "scope": "veto_only_frozen_model"}
                        evidence["reasons"] = list(dict.fromkeys(evidence["reasons"] + list(blink.reasons)))
                    result = (wave if wave is not None else np.zeros((self.channels, 0)), evidence)
                else:
                    result = self._condition(data, expected, stats.get("observed_srate"), **options)
                if assessment_timing and return_artifacts:
                    _, record = result
                    latest = snapshot["total_samples"]
                    record["sample_timing"] = {"buffer_id": snapshot["buffer_id"], "connection_generation": generation,
                                               "sample_end": end,
                                               "latest_sample_end": latest, "received_at": snapshot["received_at"]}
                    if timing_audit is not None:
                        record["timestamp_adjustment"] = timing_audit
                        record["reasons"] = list(dict.fromkeys((*record["reasons"], *timing_audit["reasons"])))
                if return_artifacts and self._motion_manager:
                    _, record = result
                    motion_buffer = self._motion_manager.buffer("motion")
                    snap = {}
                    if motion_buffer is not None:
                        hz = motion_buffer.descriptor.nominal_srate or 100.
                        meta = motion_buffer.descriptor
                        eeg_simulated = _is_sim_outlet(buffer.descriptor.source_id, buffer.descriptor.name)
                        if eeg_simulated == _is_sim_outlet(meta.source_id, meta.name):
                            snap = motion_buffer.model_snapshot(round((duration + 3) * hz))
                    motion = check_motion_reference(snap, eeg_times, self._motion_axes, self._motion_unit)
                    motion["stream"] = self._motion_manager.wanted_name
                    record["motion_reference"] = motion
                    if motion["reason"] and motion["reason"] not in record["reasons"]:
                        record["reasons"].append(motion["reason"])
                if self._manager.buffer("eeg") is not buffer or buffer.connection_generation != generation:
                    empty = np.zeros((self.channels, 0))
                    return (empty, {"reasons": ["signal_stale"]}) if return_artifacts else empty
                return result
            time.sleep(0.1)


def build_source(device: str = SIM_SOURCE, *, channels: int = 1, srate: float = 250.0,
                 seed: int = 7, ready_timeout: float = 15.0, wait_ready: bool = True,
                 research_timing: bool = False) -> SourceInfo:
    """按设备名构造数据源；真实设备不可用时抛错，由调用方决定怎么处理（**不再降级为仿真**）。

    `wait_ready=False`：只把流接上、不等样本（预览用）。这样"设备在推但还没推数据"
    会以 `live=false / no_signal=true` 呈现给界面，而不是直接报错——预览本身就是用来
    诊断"到底有没有信号"的，不能在没信号时拒绝启动。
    """
    if device and device.startswith("lsl:"):
        name = device.split(":", 1)[1]
        from ningsi_studio.domain import bsense_device
        if bsense_device.is_bsense_eeg({"kind": "eeg", "name": name}):
            srate, channels = bsense_device.SRATE, bsense_device.CHANNELS
        source = ManagedLslSource(name, device=name, srate=srate, channels=channels,
                                  ready_timeout=ready_timeout, **({"research_timing": True} if research_timing else {}))
        try:
            ready = source.start(wait=wait_ready)
        except Exception:
            source.stop()
            raise
        if wait_ready and not ready:
            status = source.status()
            source.stop()
            detail = status.get("errors") or "缓冲里还没有样本"
            raise RuntimeError(
                f"已连接 LSL 流 {name} 但 {ready_timeout:.0f} 秒内没有收到样本（{detail}）；"
                f"请确认该名称的流正在推送（GET /api/devices 可列出当前可见的流）")
        descriptor = source.descriptor_dict()
        return SourceInfo(
            key=device, kind="lsl", srate=source.srate, channels=source.channels,
            device=name,
            note=_lsl_note(descriptor, name, source.srate),
            engine=source,
        )

    engine = SyntheticEEG(srate=srate, channels=channels, seed=seed)
    return SourceInfo(
        key=SIM_SOURCE, kind="sim", srate=float(srate), channels=int(channels),
        device=SIM_SOURCE, note="仿真脑电源", engine=engine,
    )


class PreviewStream:
    """**无会话**的设备实时预览：直接读某个数据源，只做调理与质检，不落库、不出指标。

    为什么需要：会话是"按 11 个阶段跑一遍并落库"的重对象；只想确认"现在有没有信号、
    波形长什么样、这一窗能不能用"不该建会话、也不该在库里留下一条记录。
    设备状态页与实时监测页（未选会话时）都用它。

    - 未显式指定源时，自动用**当前检测到的实时源**；一个都没有就抛错（界面显示"无信号"）
      —— 与"不静默使用仿真"同一条规则：`sim-bsense` 必须被显式传入。
    - 线程安全：路由是并发的，start/stop/window 都加锁。
    """

    def __init__(self, *, window_seconds: float = 6.0, buffer_seconds: float = 60.0,
                 ready_timeout: float = 8.0) -> None:
        import threading

        self.window_seconds = float(window_seconds)
        self.buffer_seconds = float(buffer_seconds)
        self.ready_timeout = float(ready_timeout)
        self._lock = threading.RLock()
        self._source = None
        self._key: str | None = None
        self._started_at: float | None = None
        self._error: str | None = None

    # ------------------------------------------------------------------ 生命周期
    def start(self, device: str | None = None, *, probe_seconds: float = 0.8) -> dict:
        """启动预览（幂等）。`device` 为空时自动选当前检测到的实时源；没有源抛 RuntimeError。"""
        with self._lock:
            if self._source is not None and (device == self._key or device is None and self._source.is_real):
                return self.status()
            resolved = device or default_device(probe_seconds=probe_seconds)
            if not resolved:
                raise RuntimeError(
                    "未检测到脑电信号（LSL）：请先启动采集端（如 BioMultiLite / BSense-R），"
                    "或在“设备 / 数据源”里显式选择仿真源 sim-bsense（仅演示）")
            if self._source is not None:
                self.stop()
            try:
                # 预览只"接上流"，不等样本：流在但没数据时由 status() 报 live=false / no_signal=true，
                # 界面显示"无信号"，而不是整个预览起不来（那样就没法用它诊断了）。
                source = build_source(resolved, ready_timeout=self.ready_timeout, wait_ready=False)
            except Exception as exc:                     # noqa: BLE001 - 原样抛给界面显示
                self._error = str(exc)
                raise
            self._source = source
            self._key = resolved
            self._started_at = time.time()
            self._error = None
            return self.status()

    def stop(self) -> None:
        with self._lock:
            engine = getattr(self._source, "engine", None)
            if engine is not None and hasattr(engine, "stop"):
                try:
                    engine.stop()
                except Exception as exc:
                    raise RuntimeError(f"停止预览失败：{exc}") from exc
            self._source = None
            self._key = None
            self._started_at = None
            self._error = None

    @property
    def active(self) -> bool:
        return self._source is not None

    # ------------------------------------------------------------------ 状态/取数
    def status(self) -> dict:
        with self._lock:
            source = self._source
            key = self._key
            started_at = self._started_at
            error = self._error
        if source is None:
            return {"active": False, "source": None, "kind": None, "device": None,
                    "no_signal": True, "note": error or "未开始预览：当前没有脑机信号",
                    "started_at": None}
        engine = source.engine
        stats: dict = {}
        has_status = hasattr(engine, "status")
        if has_status:
            try:
                stats = engine.status() or {}
            except Exception:                            # noqa: BLE001
                stats = {}
        stream = (stats.get("streams") or {}).get("eeg") or {}
        descriptor = (stats.get("descriptors") or {}).get("eeg") or {}
        age = stream.get("seconds_since_last")
        if not has_status:
            # 仿真引擎没有流状态：它是即时生成的，直接视为有信号
            live, no_signal = True, False
        else:
            # 真机判据：**先要有过样本**（total/buffered > 0），再看新鲜度。
            # 只判"age is None ⇒ 新鲜"会把"流接上了但一个样本都没收到"误报成有信号。
            total = stream.get("total_samples")
            buffered = stream.get("buffered_samples")
            received = any(isinstance(value, (int, float)) and value > 0
                           for value in (total, buffered))
            stale = float(getattr(engine, "stale_timeout", 3.0) or 3.0)
            live = bool(received) and (age is None or float(age) <= stale)
            no_signal = not live
        conditioning = {}
        if hasattr(engine, "conditioning"):
            try:
                conditioning = engine.conditioning() or {}
            except Exception:                            # noqa: BLE001
                conditioning = {}
        srate = round(float(getattr(engine, "srate", source.srate) or 0.0), 3)
        return {
            "active": True,
            "source": key,
            "kind": source.kind,
            "simulated": source.kind == "sim" or _is_sim_outlet(descriptor.get("source_id"), source.device),
            "device": source.device,
            "srate": srate,
            "channels": int(getattr(engine, "channels", source.channels) or 1),
            "channel_labels": list(getattr(source, "channel_labels", []) or []),
            "ready": bool(live),
            # live=False 就是界面上的"无信号"：流在，但最近没有样本（设备没开推 / 线掉了）
            "live": bool(live),
            "seconds_since_last": None if age is None else round(float(age), 2),
            "buffered_samples": stream.get("buffered_samples"),
            "conditioning": conditioning,
            "errors": stats.get("errors") or {},
            "note": _lsl_note(descriptor, source.device, srate) if descriptor else source.note,
            "started_at": started_at,
            "no_signal": bool(no_signal),
        }

    def window(self) -> dict:
        """取一窗调理后的真实波形 + 质检判定（预览用，不落库）。"""
        with self._lock:
            source = self._source
            if source is None:
                return {**self.status(), "samples": [], "seconds": 0.0}
        engine = source.engine
        window_sec = min(self.window_seconds, float(getattr(engine, "condition_seconds", 0.0) or self.window_seconds))
        artifact_record = {}
        read_error = None
        try:
            # 真机（ManagedLslSource）支持 wait=False 关掉"有限等待"，预览才能按刷新率出帧；
            # 仿真源（SyntheticEEG）即时生成，接口不带该参数（与会话 /signal 通道同一处理）。
            if hasattr(engine, "analysis_window"):
                data, artifact_record = engine.analysis_window("rest", seconds=window_sec, wait=False)
            elif hasattr(engine, "condition_seconds"):
                data = engine.window("rest", seconds=window_sec, wait=False)
            else:
                data = engine.window("rest", seconds=window_sec)
        except Exception as exc:                         # noqa: BLE001 - 预览不阻塞界面
            read_error = f"读取失败：{exc}"
        # A slow frame can finish after another client switched or stopped the
        # preview. Never label that frame or its quality verdict as the new source.
        with self._lock:
            status = self.status()
            if self._source is not source:
                return {**status, "samples": [], "seconds": 0.0}
        if read_error:
            return {**status, "samples": [], "seconds": 0.0, "window_error": read_error}
        arr = np.atleast_2d(np.asarray(data, dtype=float))
        if not arr.size or status["no_signal"]:
            return {**status, "samples": [], "seconds": 0.0}
        from ningsi.signal.artifacts import check_channels

        # 刚接上流时缓冲里可能只有一两百点，而 Welch 分段需要 segment_sec × srate 点
        # （250 Hz 时 500 点）——直接算会抛"样本数不足"并把预览接口打成 500。
        # 预览是"看一眼现场信号"的工具，**绝不能因为样本不够而失败**：如实报"样本不足"。
        srate = float(status["srate"])
        required = int(round(float(config.WELCH.get("segment_sec", 2.0)) * srate))
        if arr.shape[1] < required:
            quality = {"ok": None, "reasons": ["insufficient_samples"],
                       "metrics": {"samples": int(arr.shape[1]), "required": required}}
            note = (f"样本还不够算质检：本窗 {arr.shape[1]} 点，Welch 分段需要 {required} 点"
                    f"（约 {required / max(srate, 1e-6):.1f} 秒），等一两秒再看")
        else:
            try:
                quality = with_artifact_evidence(check_channels(arr, srate), artifact_record).as_dict()
                note = None
            except Exception as exc:                     # noqa: BLE001 - 预览不因质检失败而报错
                quality = {"ok": None, "reasons": ["quality_error"], "metrics": {"error": str(exc)}}
                note = f"质检计算失败：{exc}"
        payload_arr = arr
        if arr.shape[1] > 2400:                          # 只传显示需要的点数（避免响应过大）
            step = int(np.ceil(arr.shape[1] / 2400))
            payload_arr = arr[:, ::step]
        return {
            **status,
            "seconds": round(float(arr.shape[1]) / srate, 3),
            "samples": [[round(float(value), 3) for value in channel] for channel in payload_arr],
            "peak_uv": round(float(np.max(np.abs(arr))), 2),
            "std_uv": round(float(np.mean(np.std(arr, axis=1))), 2),
            "quality": quality,
            "quality_note": note,
        }


#: 进程内单例：设备状态页与"未选会话"的实时监测页共用同一条预览流，避免重复打开设备。
_PREVIEW = PreviewStream()


def preview_stream() -> PreviewStream:
    return _PREVIEW
