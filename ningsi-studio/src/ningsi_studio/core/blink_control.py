"""Explicit, short-lived personal blink calibration; never trains on ordinary EEG."""

import hashlib
import json
import threading
import time
import uuid

import numpy as np

from ningsi.acquisition.eeg import UNIT_FACTORS, _unit
from ningsi.signal.blinks import BlinkCalibration, calibrate_blinks
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
    units = descriptor.get("declared_units") or []
    if confirmation:
        eeg_units.verify_descriptor(confirmation, descriptor)
    if any((i >= len(units) or _unit(units[i]) not in UNIT_FACTORS) and not confirmation for i in pick):
        raise ValueError("请先确认 EEG 源单位与标定倍率，再做眨眼校准")
    operator = confirmation["scale_to_uv"] if confirmation else engine.operator_eeg_scale_to_uv
    scales = voltage_scales(units, count, engine.research_scale_to_uv, pick, operator)
    identity = {key: descriptor.get(key) for key in ("name", "source_id", "hostname", "stream_type", "uid")}
    identity.update(labels=[labels[i] for i in pick], indices=pick, scales=[scales[i] for i in pick],
                    rate=float(engine.srate))
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


class BlinkCalibrationService:
    """Keep <=120s of contiguous raw calibration in memory; profiles expire in 30min."""
    def __init__(self):
        self._lock = threading.RLock()
        self._drafts = {}

    @staticmethod
    def _capture(row, snap, engine, pick, scales):
        rate = float(engine.srate)
        raw = np.asarray(snap.get("samples"), dtype=float)
        stamps = np.asarray(snap.get("timestamps"), dtype=float)
        total = snap.get("total_samples", 0)
        if (snap.get("age") is None or snap["age"] > 1 or raw.ndim != 2 or len(stamps) < round(2.5*rate)
                or raw.shape[1] != len(stamps)):
            raise ValueError("请先预览并等到连续、新鲜的至少2.5秒原始 EEG，再开始校准")
        offset = total-len(stamps)
        start = row.get("sample_end", total-round(2.5*rate))
        if start < offset or start > total:
            raise ValueError("校准期间原始数据不连续，请重新开始并保持本页打开")
        first = start-offset
        timing = stamps[max(0, first-1):]
        if (not np.isfinite(timing).all() or np.any(np.diff(timing) <= 0)
                or np.any(np.diff(timing) > 1.75/rate)):
            raise ValueError("校准期间出现时间戳间断，请重新开始")
        wave = raw[np.array(pick), first:] * np.asarray(scales)[pick, None]
        if not np.isfinite(wave).all():
            raise ValueError("当前原始波形无效，请检查接触并重新开始")
        if "start_index" not in row:
            row["start_index"] = start
        if total-row["start_index"] > round(120*rate):
            raise ValueError("本次眨眼采集超过120秒，请重新开始，或取消后再评估")
        if wave.shape[1]:
            row["chunks"].append(wave)
        row["sample_end"] = total

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
                confirmation = eeg_units.confirmation(payload.get("eeg_calibration"))
                preview.start(device)
                engine = preview._source.engine
                if confirmation:
                    engine.operator_eeg_scale_to_uv = confirmation["scale_to_uv"]
                pick, scales, labels, montage = context(engine, confirmation)
                token = uuid.uuid4().hex
                snap = engine.model_snapshot()
                self._drafts[token] = {"created": now, "participant": participant, "device": device,
                    "unit_confirmation": confirmation, "montage_id": montage, "labels": labels,
                    "chunks": [], "peaks": [], "last_peak_time": None, "profile": None,
                    "buffer_id": snap.get("buffer_id"), "connection_generation": snap.get("connection_generation")}
                try:
                    self._capture(self._drafts[token], snap, engine, pick, scales)
                except ValueError:
                    del self._drafts[token]
                    raise
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
                if row["profile"] is None:
                    self._capture(row, snap, engine, pick, scales)
                if action == "mark":
                    if row["profile"] is not None or len(row["peaks"]) >= 20:
                        raise ValueError("本次标记已完成，请结束校准或重新开始")
                    rate = float(engine.srate)
                    size = round(2.5*rate)
                    raw = np.asarray(snap.get("samples"), dtype=float)
                    stamps = np.asarray(snap.get("timestamps"), dtype=float)
                    if (snap.get("age") is None or snap["age"] > 1 or raw.ndim != 2 or raw.shape[1] < size
                            or len(stamps) < size or not np.isfinite(stamps[-size:]).all()
                            or np.any(np.diff(stamps[-size:]) <= 0)
                            or np.any(np.diff(stamps[-size:]) > 1.75/rate)):
                        raise ValueError("需要连续、新鲜的至少2.5秒原始 EEG；请稍后再标记")
                    wave = raw[np.array(pick), -size:] * np.asarray(scales)[pick, None]
                    if not np.isfinite(wave).all():
                        raise ValueError("当前原始波形无效，请检查接触")
                    width = max(1, round(.06*rate))
                    slow = np.stack([np.convolve(ch-np.median(ch), np.ones(width)/width, mode="same") for ch in wave])
                    # User clicks 0.5–1.3s after the blink. Never label the click as a physiological peak.
                    first, last = size-round(1.3*rate), size-round(.5*rate)
                    strength = np.min(np.abs(slow[:, first:last]), axis=0)
                    peak = first+int(np.argmax(strength))
                    peak_time = float(stamps[-size+peak])
                    if row["last_peak_time"] is not None and peak_time-row["last_peak_time"] < 1.5:
                        raise ValueError("同一次眨眼不能重复标记；请间隔至少3秒再自然短眨眼")
                    if strength[peak-first] < 35:
                        raise ValueError("未找到清楚的双通道短眨眼，请眨眼后等约半秒再标记")
                    row["last_peak_time"] = peak_time
                    row["peaks"].append(snap["total_samples"]-size+peak-row["start_index"])
                elif action == "finish":
                    if len(row["peaks"]) < 15:
                        raise ValueError("请至少标记15次自然短眨眼（建议15–20次）")
                    if row["profile"] is None:
                        profile = calibrate_blinks(np.concatenate(row["chunks"], axis=1), engine.srate,
                            row["peaks"], channel_labels=labels, montage_id=montage)
                        row["profile"] = profile.as_dict()
                        row["chunks"] = []  # raw calibration data is transient, not persisted
                elif action != "status":
                    raise ValueError("未知眨眼校准操作")
            row = self._drafts[token]
            return {"token": token, "state": "ready" if row["profile"] else "collecting",
                    "count": len(row["peaks"]), "participant": participant, "device": device,
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
