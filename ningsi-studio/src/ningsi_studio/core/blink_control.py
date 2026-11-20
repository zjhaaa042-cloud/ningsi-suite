"""Explicit, short-lived personal blink calibration; never trains on ordinary EEG."""

import hashlib
import json
import math
import threading
import time
import uuid

import numpy as np

from ningsi.acquisition.eeg import UNIT_FACTORS, _unit
from ningsi.signal.blinks import MIN_EVENTS, BlinkCalibration, calibrate_screened_blinks, screen_blink_observation
from ningsi.signal.live_artifacts import channel_layout, voltage_scales
from ningsi_studio.domain import eeg_units


def context(engine, confirmation=None):
    descriptor = engine.descriptor_dict()
    if not descriptor.get("channel_count"):
        raise ValueError("已连接，正在等待设备信息和至少2.5秒数据，请稍后再次点击开始校准")
    labels = descriptor.get("channel_labels") or []
    count = descriptor.get("channel_count", len(labels))
    pick, _ = channel_layout(labels, count)
    if len(pick) != 2 or any(i >= len(labels) or not labels[i] for i in pick):
        raise ValueError("眨眼校准需要两路有明确标签的原始 EEG")
    rate = float(engine.srate)
    if not math.isfinite(rate) or not 100 <= rate <= 2000:
        raise ValueError("眨眼校准需要 100–2000 Hz 的原始 EEG")
    units = descriptor.get("declared_units") or []
    if confirmation:
        eeg_units.verify_descriptor(confirmation, descriptor)
    if any((i >= len(units) or _unit(units[i]) not in UNIT_FACTORS) and not confirmation for i in pick):
        raise ValueError("请先确认 EEG 源单位与标定倍率，再做眨眼校准")
    operator = confirmation["scale_to_uv"] if confirmation else engine.operator_eeg_scale_to_uv
    scales = voltage_scales(units, count, engine.research_scale_to_uv, pick, operator)
    identity = {key: descriptor.get(key) for key in ("name", "source_id", "hostname", "stream_type", "uid")}
    identity.update(labels=[labels[i] for i in pick], indices=pick, scales=[scales[i] for i in pick],
                    rate=rate)
    montage = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return pick, scales, identity["labels"], montage


def verify_binding(binding, engine):
    pick, scales, labels, montage = context(engine, binding.get("unit_confirmation"))
    if montage != binding["montage_id"]:
        raise ValueError("眨眼校准的设备、导联、采样率或单位已变化，请重新校准")
    descriptor = engine.descriptor_dict()
    active_scales = voltage_scales(descriptor.get("declared_units", ()), descriptor["channel_count"],
        engine.research_scale_to_uv, pick, engine.operator_eeg_scale_to_uv)
    if any(active_scales[i] != scales[i] for i in pick):
        raise ValueError("本次 EEG 换算倍率与眨眼校准不同，请重新校准")
    calibration = BlinkCalibration.from_dict(binding["profile"])
    return calibration, pick, scales, labels, montage


class BlinkCalibrationRetry(ValueError):
    """The current observation failed but earlier accepted observations still apply."""

    def __init__(self, message, count):
        super().__init__(message)
        self.detail = {"retryable": True, "accepted_count": count}


class BlinkCalibrationService:
    """Keep screened mark observations in memory for one 120-second collection."""
    def __init__(self):
        self._lock = threading.RLock()
        self._drafts = {}

    @staticmethod
    def _recent(snap, engine, pick, scales):
        rate = float(engine.srate)
        raw = np.asarray(snap.get("samples"), dtype=float)
        stamps = np.asarray(snap.get("timestamps"), dtype=float)
        size = round(2.5 * rate)
        if (snap.get("age") is None or snap["age"] > 1 or raw.ndim != 2 or stamps.ndim != 1
                or len(stamps) < size or raw.shape[1] != len(stamps)):
            raise ValueError("请等待至少 2.5 秒新鲜的双通道脑电，再记录。")
        timing = stamps[-size:]
        differences = np.diff(timing)
        if (not np.isfinite(timing).all() or np.any(differences <= 0) or np.any(differences > 1.75 / rate)
                or abs((timing[-1] - timing[0]) * rate / (size - 1) - 1) > .05):
            raise ValueError("本次记录的时间戳间断或不稳定，请等 EEG 推流稳定后再眨眼。")
        wave = raw[np.array(pick), -size:] * np.asarray(scales)[pick, None]
        if not np.isfinite(wave).all():
            raise ValueError("本次记录含无效波形，请等信号稳定后重新眨眼。")
        return wave, timing

    def _mark(self, row, snap, engine, pick, scales):
        rate = float(engine.srate)
        total = snap.get("total_samples", 0)
        if row["profile"] is not None:
            raise ValueError("本次校准已完成。")
        if total - row["start_index"] > round(120 * rate):
            raise ValueError("此次 120 秒的标记时段已结束。已有足够标记时可结束校准。")
        wave, stamps = self._recent(snap, engine, pick, scales)
        size = wave.shape[1]
        if row["sample_end"] is not None and total - size < row["sample_end"]:
            raise ValueError("记录间隔太短，请间隔至少 3 秒再自然短眨一次。")
        width = max(1, round(.06 * rate))
        slow = np.stack([np.convolve(ch - np.median(ch), np.ones(width) / width, mode="same") for ch in wave])
        # The click selects recent data; the candidate peak comes from the measured wave.
        first, last = size - round(1.6 * rate), size - round(.35 * rate)
        strength = np.min(np.abs(slow[:, first:last]), axis=0)
        peak = first + int(np.argmax(strength))
        if strength[peak - first] < 35:
            raise ValueError("未找到清楚的双通道短眨眼。请自然短眨一次，等约 0.8 秒再点记录。")
        observation = screen_blink_observation(wave, rate, peak)
        if total - size + observation[0] < row["start_index"]:
            raise ValueError("请在开始本次校准后自然短眨，再点记录。")
        peak_time = float(stamps[observation[0]])
        if row["last_peak_time"] is not None and peak_time - row["last_peak_time"] < 1.5:
            raise ValueError("同一次眨眼不能重复标记，请间隔至少 3 秒再自然短眨一次。")
        row["last_peak_time"] = peak_time
        row["sample_end"] = total
        row["chunks"].append(observation)
        row["count"] += 1

    def action(self, payload, preview):
        with self._lock, preview._lock:
            now = time.monotonic()
            self._drafts = {key: row for key, row in self._drafts.items() if now-row["created"] < 1800}
            action = payload.get("action")
            if not isinstance(action, str) or action not in ("start", "mark", "finish", "status", "clear"):
                raise ValueError("未知眨眼校准操作")
            participant = payload.get("participant")
            device = payload.get("device")
            if not isinstance(participant, str) or not participant.strip() or not isinstance(device, str) or not device.startswith("lsl:"):
                raise ValueError("先选择参与者和真实 EEG 数据流")
            if action == "start":
                if len(self._drafts) >= 32:
                    raise ValueError("眨眼校准草稿过多，请稍后重试")
                preview.start(device)
                engine = preview._source.engine
                confirmation = eeg_units.resolve_calibration(payload.get("eeg_calibration"), engine.descriptor_dict())
                pick, scales, labels, montage = context(engine, confirmation)
                token = uuid.uuid4().hex
                snap = engine.model_snapshot()
                self._recent(snap, engine, pick, scales)
                if confirmation:
                    engine.operator_eeg_scale_to_uv = confirmation["scale_to_uv"]
                self._drafts[token] = {"created": now, "participant": participant, "device": device,
                    "unit_confirmation": confirmation, "montage_id": montage, "labels": labels,
                    "chunks": [], "count": 0, "last_peak_time": None, "profile": None,
                    "sample_end": None, "start_index": snap["total_samples"],
                    "buffer_id": snap.get("buffer_id"), "connection_generation": snap.get("connection_generation")}
            else:
                token = payload.get("token")
                if not isinstance(token, str):
                    raise ValueError("请使用本次个人眨眼校准 token")
                row = self._drafts.get(token)
                if not row or row["participant"] != participant or row["device"] != device:
                    raise ValueError("本人的眨眼校准已过期或设备已改变，请重新开始")
                if action == "clear":
                    del self._drafts[token]
                    return {"state": "cleared", "count": 0}
                if not preview._source or preview._key != device:
                    raise ValueError("预览源已经改变，请重新开始眨眼校准")
                engine = preview._source.engine
                pick, scales, labels, montage = context(engine, row["unit_confirmation"])
                snap = engine.model_snapshot()
                if (montage != row["montage_id"] or snap.get("buffer_id") != row["buffer_id"]
                        or snap.get("connection_generation") != row["connection_generation"]):
                    raise ValueError("数据源已重连或配置已改变，请重新开始眨眼校准")
                if action == "mark":
                    try:
                        self._mark(row, snap, engine, pick, scales)
                    except ValueError as error:
                        raise BlinkCalibrationRetry(str(error), row["count"]) from error
                elif action == "finish":
                    if row["count"] < MIN_EVENTS:
                        raise BlinkCalibrationRetry(f"至少需要 {MIN_EVENTS} 次通过筛查的短眨眼。", row["count"])
                    if row["profile"] is None:
                        try:
                            profile = calibrate_screened_blinks(row["chunks"], engine.srate, channel_labels=labels,
                                                                montage_id=montage)
                        except ValueError as error:
                            raise BlinkCalibrationRetry(str(error), row["count"]) from error
                        row["profile"] = profile.as_dict()
                        row["chunks"] = []
                elif action != "status":
                    raise ValueError("未知眨眼校准操作")
            row = self._drafts[token]
            return {"token": token, "state": "ready" if row["profile"] else "collecting",
                    "count": row["count"], "minimum_events": MIN_EVENTS,
                    "template_events": row["profile"]["events"] if row["profile"] else None,
                    "participant": participant, "device": device,
                    "expires_in_seconds": max(0, round(1800-(now-row["created"])))}

    def binding(self, token, participant, device, preview=None):
        with self._lock:
            if not isinstance(token, str):
                raise ValueError("请使用本次个人眨眼校准 token")
            row = self._drafts.get(token)
            if (not row or time.monotonic()-row["created"] >= 1800 or row["participant"] != participant
                    or row["device"] != device or not row["profile"]):
                raise ValueError("个人眨眼校准未完成、已过期或与本次参与者/设备不匹配")
            if preview is not None:
                with preview._lock:
                    if not preview._source or preview._key != device:
                        raise ValueError("校准后的预览已关闭或来源已改变，请重新校准")
                    engine = preview._source.engine
                    snap = engine.model_snapshot()
                    if (snap.get("buffer_id") != row["buffer_id"]
                            or snap.get("connection_generation") != row["connection_generation"]):
                        raise ValueError("校准后的设备已重连，请重新校准")
                    if context(engine, row["unit_confirmation"])[3] != row["montage_id"]:
                        raise ValueError("校准后的设备配置已变化，请重新校准")
            return {key: row[key] for key in ("participant", "device", "montage_id", "unit_confirmation", "profile")}


SERVICE = BlinkCalibrationService()
