"""Read-only configuration checks shared by the product API and field tooling.

The checks never open an EEG inlet, create a session or certify signal quality.
Model compatibility delegates to the same predictor used by live sessions.
"""

from __future__ import annotations

import math

from ningsi.acquisition.eeg import UNIT_FACTORS, normalize_unit
from ningsi_studio.core import live_source, phases
from ningsi_studio.core.human_inference import HumanAttentionPredictor
from ningsi_studio.domain import eeg_units


MODEL_REASONS = {
    "simulation_source": "模拟数据不用于真人模型推断。",
    "native_rec_clock_required": "该模型依赖离线原生 REC 核对后的设备时钟，普通实时 LSL 不能提供；请换用实时来源一致的模型。",
    "device_channels_units_mismatch": "设备名称、来源标识、通道、电极标签、采样率、单位或时标策略与模型不匹配。",
    "feature_engine_mismatch": "模型特征引擎版本与当前程序不同，请统一版本后重新导出和训练。",
    "feature_implementation_mismatch": "模型预处理或特征聚合实现与当前程序不同，请统一版本后重新导出和训练。",
}


def _model_check(active_model, descriptor, calibration, protocol, *, simulated, model_error):
    if model_error:
        return {"status": "unavailable", "reason": "model_unavailable",
                "message": "已启用模型文件无法校验，真人模型将暂停；请到模型管理重新导入有效模型。"}, None
    if active_model is None:
        return {"status": "none", "message": "未启用真人专注模型；本次使用相对个人基线指数，不输出真人模型概率。"}, None
    model_id = active_model.get("model_id") if isinstance(active_model, dict) else None
    if simulated:
        return {"status": "simulation", "model_id": model_id, "reason": "simulation_source",
                "message": MODEL_REASONS["simulation_source"]}, None
    if descriptor is None:
        return {"status": "unavailable", "model_id": model_id, "reason": "device_unavailable",
                "message": "尚未取得所选设备的实时元数据，无法检查模型适配。"}, None
    try:
        predictor = HumanAttentionPredictor(active_model)
        matched = predictor.match_profile(descriptor)
        if calibration and calibration["scale_to_uv"] != matched["scale_to_uv"]:
            raise ValueError("device_channels_units_mismatch")
    except (ValueError, TypeError, KeyError, OverflowError) as error:
        reason = str(error)
        return {"status": "incompatible", "model_id": model_id, "reason": reason,
                "message": MODEL_REASONS.get(reason, "真人模型校验未通过；请在模型管理核对来源、版本与采集配置。")}, None
    keys = ("name", "source_id", "channels", "channel_labels", "srate", "source_unit", "scale_to_uv", "timestamp_policy", "reference")
    return {"status": "compatible", "model_id": model_id, "reason": "",
            "requirements": {key: matched.get(key) for key in keys},
            "scope_status": "outside_validated_task" if protocol in ("ten_min", "effective_time") else "sart_probe",
            "message": "设备配置与已启用真人模型匹配；仍需有效基线、近期合格脑电与足够历史才能出分。"}, matched


def evaluate_preflight(payload, *, descriptors=(), device_profile=None, active_model=None, model_error=None,
                       scan_error=None):
    """Pure policy evaluation. ``descriptors`` must come from the server's LSL scan.

    ``ready`` means a real EEG source has compatible metadata and usable unit
    configuration. It is never a baseline/QC pass and does not imply model efficacy.
    An incompatible optional model is a warning; relative-index assessment remains
    possible. Stored operator declarations never silently override live units.
    """
    if not isinstance(payload, dict):
        raise ValueError("检查参数必须是 JSON 对象。")
    device = payload.get("device") or ""
    if not isinstance(device, str) or len(device) > 120 or any(ord(c) < 32 for c in device):
        raise ValueError("设备来源键无效。")
    protocol = payload.get("protocol") or "ten_min"
    if not isinstance(protocol, str) or protocol not in phases.PROFILES:
        raise ValueError("请选择有效的评估方案。")
    checks = []

    def add(key, status, label, message, action=None):
        row = {"key": key, "status": status, "label": label, "message": message}
        if action:
            row["action"] = action
        checks.append(row)

    matches = [row for row in descriptors if device == "lsl:" + str(row.get("name") or "")]
    descriptor = matches[0] if len(matches) == 1 else None
    simulated = device == live_source.SIM_SOURCE or bool(descriptor and (
        descriptor.get("simulated") is True
        or live_source._is_sim_outlet(descriptor.get("source_id"), descriptor.get("name"))))
    if simulated:
        add("device", "fail", "数据来源", "这是模拟来源，请使用独立的模拟演示入口；不能作为正式真人采集。", "#/start")
    elif len(matches) > 1:
        add("device", "fail", "数据来源", "发现多个同名 LSL 流，来源有歧义；请关闭重复发布或更改流名称。", "#/devices")
    elif descriptor is None:
        add("device", "fail", "数据来源", "设备扫描失败，请检查采集程序后重新扫描。" if scan_error
            else "未发现所选实时 EEG 来源，请在采集程序开启推流后重新扫描。", "#/devices")
    elif descriptor.get("kind") != "eeg" or descriptor.get("supported") is False:
        add("device", "fail", "数据来源", "所选来源不是支持的原始 EEG 流，请选择脑电流。", "#/devices")
    else:
        try:
            rate = float(descriptor.get("nominal_srate") or 0)
            channel_count = int(descriptor.get("channel_count") or 0)
            valid = math.isfinite(rate) and 1 <= rate <= 10000 and 1 <= channel_count <= 256
        except (ValueError, TypeError, OverflowError):
            valid = False
        if valid and rate <= 90:
            add("device", "fail", "数据来源", "当前预处理使用 45 Hz 低通，采样率必须大于 90 Hz；当前设备采样率不支持此处理链，请核对采集程序设置。", "#/devices")
        elif valid:
            add("device", "pass", "数据来源", "已发现所选 EEG 流；这仅确认元数据可见，实际收数仍需实时质检。")
        else:
            add("device", "fail", "数据来源", "流的采样率或通道数无效，请核对厂商发布设置。", "#/devices")
    calibration = None
    calibration_error = None
    try:
        calibration = eeg_units.confirmation(payload.get("eeg_calibration"))
        if calibration and descriptor:
            eeg_units.verify_descriptor(calibration, descriptor)
    except (ValueError, TypeError, OverflowError) as error:
        calibration_error = str(error)
        calibration = None

    model, model_profile = _model_check(active_model, descriptor, calibration, protocol,
                                         simulated=simulated, model_error=model_error)
    if calibration_error:
        add("units", "fail", "单位与标定", calibration_error, "#/devices")
    elif descriptor is None or simulated:
        add("units", "pending", "单位与标定", "连接真实设备后，核对 EEG 单位和 ADC 换算倍率。", "#/devices")
    elif calibration:
        add("units", "pass", "单位与标定", "本次确认的 EEG 单位与设备声明未冲突；实际标定依据仍需操作者核对。")
    else:
        units = descriptor.get("declared_units") or []
        indices = eeg_units.eeg_indices(descriptor)
        known = bool(indices) and all(i < len(units) and normalize_unit(units[i]) in UNIT_FACTORS for i in indices)
        if known:
            add("units", "pass", "单位与标定", "EEG 通道已声明可换算的电压单位；仍需核对厂商输出及实际标定。")
        elif model_profile:
            add("units", "pass", "单位与标定", "沿用已启用模型内经身份匹配的同源单位配置；请核对设备设置未改变。")
        else:
            add("units", "fail", "单位与标定", "设备未完整声明 EEG 电压单位，或输出为 ADC 计数；请在本次单位设置中显式确认，ADC 必须有校准倍率。", "#/devices")

    profile = device_profile or {"saved": False, "profile": {}, "missing_fields": [], "status": "pending_verification"}
    missing = profile.get("missing_fields") or []
    add("device_profile", "warning" if not profile.get("saved") or missing else "pass", "设备资料",
        "设备资料尚不完整，请在设备连接页保存型号、导联、参考和标定依据。" if not profile.get("saved") or missing
        else "所选来源的设备资料已保存为操作员声明，仍需现场核对。", "#/devices")
    add("attention_model", "pass" if model["status"] in ("compatible", "none") else "warning",
        "真人专注模型", model["message"], "#/models")
    if active_model and protocol in ("ten_min", "effective_time"):
        add("model_scope", "warning", "模型适用任务", "当前训练方案不含 SART 独立监测；真人模型输出超出原探针验证范围，仅作探索性观察。")
    add("quality", "pending", "实时质量与个人基线", "开始后检查实际信号、新鲜度、伪迹和个人基线；不合格时暂停评分。")
    ready = not any(row["status"] == "fail" for row in checks) and descriptor is not None and not simulated
    return {"schema": "ningsi-session-preflight-v1", "device": device, "protocol": protocol,
            "ready": ready, "checks": checks, "device_profile": profile, "calibration": calibration,
            "descriptor": descriptor,
            "scoring": {"relative_indices": {"status": "pending" if ready else "blocked",
                "message": "专注、放松与负荷为相对个人基线指数；合格后按原指数提供预警与反馈。",
                "feedback_target": "relative_focus_index"}, "attention_model": model},
            "note": "此检查仅核对开始前配置；不采集 EEG、不建立基线，不表示状态识别或训练效果已通过验证。"}
