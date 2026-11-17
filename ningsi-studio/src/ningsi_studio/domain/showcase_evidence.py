"""Bounded, privacy-preserving competition evidence and device declarations.

Only the named local summaries below are read. Neither API callers nor profile
document references select a file to read. A completed declaration is still an
operator statement, never hardware certification or a field-validation pass.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

from ningsi.acquisition.eeg import UNIT_FACTORS


EVIDENCE_SCHEMA = "ningsi-showcase-evidence-v1"
PROFILE_SCHEMA = "ningsi-showcase-device-declaration-v1"
PROFILE_FILE = "showcase/device_profile.json"
MAX_EVIDENCE_BYTES = 128 * 1024
MAX_PROFILE_BYTES = 24 * 1024
SNAPSHOT_FILE = "showcase_evidence_snapshot.json"

_SOURCES = {
    "short_capture": "var/biomulti-qc-0.3.4/live-device-check.json",
    "complete_capture": "var/preprocessing-0.4.0/P001-export-check.json",
    "recent_capture": "var/preprocessing-0.4.0/live-review.json",
    "research": "var/calibration-dose-20261116/RESULTS.md",
    "software": "docs/VALIDATION_PLAN.md",
}

_TEXT_FIELDS = {
    "manufacturer": 160, "model": 160, "electrodes": 256,
    "reference": 256, "source_unit": 16, "unit_basis": 1500,
    "document_reference": 2000, "notes": 2000,
}
_NUMBER_FIELDS = {"srate", "scale_to_uv"}
_FIELD_LABELS = {
    "manufacturer": "设备厂家", "model": "实际设备型号",
    "electrodes": "实际电极位置", "reference": "参考电极",
    "srate": "标称采样率", "source_unit": "源数值单位",
    "scale_to_uv": "换算到 µV 的倍率", "unit_basis": "单位与换算依据",
    "document_reference": "设备资料与赛事允许依据",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON value: {value}")


def _read_source(root: Path | None, key: str, *, json_file=True):
    """Return only server-owned, in-root files; missing evidence is normal."""
    result = {"available": False, "availability": "missing", "sha256": None}
    if root is None:
        return None, result
    try:
        source = (root / _SOURCES[key]).resolve()
        source.relative_to(root)
        if not source.is_file():
            return None, result
        if source.stat().st_size > MAX_EVIDENCE_BYTES:
            raise ValueError("Evidence summary exceeds limit")
        data = source.read_bytes()
        if len(data) > MAX_EVIDENCE_BYTES:
            raise ValueError("Evidence summary exceeds limit")
        text = data.decode("utf-8-sig")
        value = json.loads(text, parse_constant=_reject_constant) if json_file else text
        if json_file and not isinstance(value, dict):
            raise ValueError("Evidence summary must be an object")
        return value, {"available": True, "availability": "available",
                       "sha256": hashlib.sha256(data).hexdigest()}
    except (OSError, UnicodeError, ValueError, TypeError):
        result["availability"] = "invalid"
        return None, result


def _count(value):
    return value if type(value) is int and 0 <= value <= 10**12 else None


def _number(value):
    if type(value) not in (int, float):
        return None
    try:
        return float(value) if math.isfinite(value) else None
    except OverflowError:
        return None


def _quality_metrics(total, usable):
    total, usable = _count(total), _count(usable)
    if total is None or usable is None or usable > total:
        raise ValueError("Invalid quality count")
    return {"windows_total": total, "windows_usable": usable,
            "valid_ratio": usable / total if total else None}


def _short_metrics(data):
    return {**_quality_metrics(data.get("windows_total"), data.get("all_quality_pass")),
            "duration_sec": _number(data.get("seconds_acquired")),
            "timing_pass": _count(data.get("timing_pass"))}


def _complete_metrics(data):
    readiness = data.get("training_readiness") or {}
    if not isinstance(readiness, dict) or type(data.get("baseline_valid")) is not bool:
        raise ValueError("Missing field-validation fields")
    metrics = {**_quality_metrics(data.get("windows_total"), data.get("windows_usable")),
               "baseline_valid": data["baseline_valid"],
               "formal_probes": _count(readiness.get("formal_probes")),
               "paired": _count(readiness.get("paired")),
               "train_eligible": _count(data.get("train_eligible")),
               "test_eligible": _count(data.get("test_eligible"))}
    if metrics["formal_probes"] is None or metrics["paired"] is None:
        raise ValueError("Missing paired-probe counts")
    if metrics["paired"] > metrics["formal_probes"]:
        raise ValueError("Invalid paired-probe counts")
    return metrics


def _recent_metrics(data):
    metrics = {**_quality_metrics(data.get("windows_total"), data.get("windows_usable")),
               "duration_sec": _number(data.get("duration_seconds_by_sample_count")),
               "channels": _count(data.get("channels")), "srate": _number(data.get("srate")),
               "windows_with_valid_clock": _count(data.get("windows_with_valid_history_and_clock"))}
    # Do not return the full window list, waveforms, source configuration or IDs.
    return metrics


def _record(root, key, title, parser, notes):
    data, state = _read_source(root, key)
    item = {"id": key, "title": title, "kind": "target_device_record", **state,
            "result": "unavailable", "metrics": {}, "notes": list(notes)}
    if data is None:
        return item
    try:
        metrics = parser(data)
    except (ValueError, TypeError, KeyError, OverflowError):
        item.update(available=False, availability="invalid")
        return item
    item["metrics"] = metrics
    negative = metrics.get("windows_usable") == 0
    if key == "complete_capture":
        negative |= not metrics["baseline_valid"] or metrics["paired"] == 0
    # Even a useful recording cannot establish hardware compliance or generality.
    item["result"] = "not_accepted" if negative else "partial"
    return item


def _requirements(records):
    full = next(item for item in records if item["id"] == "complete_capture")
    field_status = "not_accepted" if full["result"] == "not_accepted" else "pending"
    rows = [
        ("hardware", "真实脑电硬件采集", field_status,
         "真实采集已接入，合规与完整流程待核验", "#/devices",
         ["short_capture", "complete_capture", "recent_capture"],
         ["实际型号、导联、参考与单位仍需现场核对。", "接收成功不能代替有效基线与任务配对验收。"],
         "保存设备声明与材料出处，完成合格基线和正式任务采集。"),
        ("preprocessing", "信号降噪与伪迹处理", "partial",
         "滤波已实现，伪迹校正与拒评效果待现场验证", "#/showcase?step=preprocessing",
         ["recent_capture", "complete_capture"],
         ["同步 EOG 参考可用于回归校正；无参考时主要检测和排除污染。",
          "肌电与运动的标记、留空和拒窗不能等同于恢复无伪迹脑电。"],
         "展示处理前后信号，并用眨眼、咬牙、转头对照核查误拒和漏检。"),
        ("features", "PSD 与可解释特征", "software_only",
         "θ、α、β PSD 与时域特征已实现", "#/showcase?step=features",
         [], ["特征可计算并不证明其能可靠识别心理状态。"],
         "用同窗原始数据、固定单位和频带核对可复算特征。"),
        ("models", "可训练模型与部署", "pending",
         "训练导出与加载已实现，目标设备精度待验证", "#/models",
         ["complete_capture"],
         ["专注模型需要独立探针标签与按人隔离的开发和测试。",
          "压力和情绪候选保留研究分数，不自动判定个人状态。"],
         "导入合格采集导出，冻结模型后使用独立人员评估。"),
        ("joint", "脑电量表联合评估", "pending",
         "规则式联合评估已实现，预测有效性待验证", "#/flow",
         [], ["脑电指数、SAS/SDS、SART/PVT 按来源融合；模型分数独立展示。",
              "当前规则阈值尚未建立目标人群常模。"],
         "运行完整协议，核对证据一致、冲突和缺失的报告分支。"),
        ("report", "结构化报告与建议", "software_only",
         "报告生成与导出已实现", "#/report",
         [], ["报告中的状态含义受原始证据质量和验证范围约束。"],
         "展示评分、质量、问题分析、建议以及 JSON、HTML 导出。"),
        ("visualization", "实时波形频谱与热力图", "software_only",
         "实时可视化已实现", "#/live",
         [], ["波形刷新率与状态分析窗口分别计算，不能混用响应尺度。"],
         "显示真实或明确标注的模拟源，并核对无效窗独立留空。"),
        ("trends", "历史查询与周月趋势", "pending",
         "查询聚合已实现，长期真人追踪待积累", "#/history",
         [], ["专注曲线为相对基线指数；压力曲线来自本人当下自评。",
              "当前曲线不能表述为已验证的脑电压力识别趋势。"],
         "按同一参与者查询周月数据，保留缺失与配置不可比提示。"),
    ]
    return [{"id": key, "title": title, "software_status": "implemented",
             "verification_status": verification, "status_label": label, "route": route,
             "evidence_ids": ids, "gaps": gaps, "next_action": action}
            for key, title, verification, label, route, ids, gaps, action in rows]


def _bundled_snapshot():
    """Read only the fixed, anonymized package resource, never a user path."""
    try:
        path = Path(__file__).with_name(SNAPSHOT_FILE)
        if path.stat().st_size > MAX_EVIDENCE_BYTES:
            raise ValueError("Oversized bundled evidence snapshot")
        data = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
        if (not isinstance(data, dict) or data.get("schema") != EVIDENCE_SCHEMA
                or not isinstance(data.get("snapshot_at"), str)
                or not isinstance(data.get("evidence"), list)
                or {row.get("id") for row in data["evidence"] if isinstance(row, dict)}
                   != {"short_capture", "complete_capture", "recent_capture"}):
            raise ValueError("Unsupported bundled evidence snapshot")
        # The snapshot is a sanitized build artifact, never an original export.
        forbidden = {"samples", "windows", "session_id", "participant", "source_path", "path"}

        def check(value):
            if isinstance(value, dict):
                if value.keys() & forbidden:
                    raise ValueError("Bundled evidence contains private fields")
                for item in value.values():
                    check(item)
            elif isinstance(value, list):
                for item in value:
                    check(item)
        check(data)
        return data
    except (OSError, UnicodeError, ValueError, TypeError):
        return None


def build_evidence(workspace=None) -> dict:
    """Read fixed summaries under an internally selected workspace, never user paths.

    Source checkout is the default; frozen applications have no implicit access
    to the developer's experiment directory. They may display a clearly labelled
    anonymized development snapshot. Explicit workspace checks never fall back.
    """
    root = (Path(workspace).resolve() if workspace is not None else
            None if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[4])
    records = [
        _record(root, "short_capture", "真实短段采集核对", _short_metrics,
                ["这份记录仅覆盖短时采集。", "合格窗口有重叠，不能当作独立被试样本。"]),
        _record(root, "complete_capture", "完整研究记录按新策略复算", _complete_metrics,
                ["历史原始记录使用新预处理策略复算，原始数据保持不变。",
                 "有效基线和正式探针配对需分别满足要求。"]),
        _record(root, "recent_capture", "后续短时实采处理核对", _recent_metrics,
                ["工频抑制与整窗是否合格分别检查。", "结果仅适用于本段记录。"]),
    ]
    report, research_state = _read_source(root, "research", json_file=False)
    if report is not None and not all(word in report for word in ("固定后测", "SAM40", "FACED", "Monash")):
        research_state.update(available=False, availability="invalid")
    research = {**research_state, "scope": "public_cohort", "target_device_validated": False,
                "title": "公开真人脑电个人校准量研究",
                "conclusion": ("报告比较压力、情绪与跨研究专注的早期个人校准；"
                               "结果不能代替目标设备现场验证。") if research_state["available"]
                               else "本运行目录未提供可核对的研究摘要。"}
    software_text, software_state = _read_source(root, "software", json_file=False)
    total = (re.search(r"共\s*(\d+)\s*项软件测试通过", software_text) if software_text else None)
    software_validation = {**software_state, "kind": "recorded_development_snapshot",
                           "tests_total": int(total[1]) if total else None,
                           "note": "记录中的软件回归结果；本接口不重新执行测试，也不代表真人效果。"}
    requirements = _requirements(records)
    result = {"schema": EVIDENCE_SCHEMA, "generated_at": _utc_now(),
            "overall_status": "pending_field_validation", "requirements": requirements,
            "summary": {"requirements": len(requirements),
                        "software_implemented": len(requirements), "target_device_accepted": 0,
                        "evidence_available": sum(item["available"] for item in records)},
            "evidence": records, "research": research, "software_validation": software_validation,
            "evidence_origin": "workspace_summaries", "snapshot_at": None,
            "scope_note": "实现状态、软件回归与目标设备验收分别展示；未自动计算总体达标率。"}
    all_experiments_missing = all(row["availability"] == "missing" for row in records)
    all_experiments_missing &= research_state["availability"] == "missing"
    if workspace is None and all_experiments_missing:
        snapshot = _bundled_snapshot()
        if snapshot is not None:
            snapshot["generated_at"] = result["generated_at"]
            snapshot["evidence_origin"] = "bundled_development_snapshot"
            snapshot["snapshot_note"] = "随软件提供的脱敏开发证据快照；不代表当前设备或本次会话的实时测量。"
            return snapshot
    return result


def _empty_profile():
    return {**{key: "" for key in _TEXT_FIELDS}, **{key: None for key in sorted(_NUMBER_FIELDS)}}


def _normalize_profile(payload):
    if not isinstance(payload, dict):
        raise ValueError("设备声明必须是 JSON 对象。")
    values = dict(payload)
    # Keep one canonical output field while supporting the initial form name.
    if "materials" in values:
        if "document_reference" in values and values["materials"] != values["document_reference"]:
            raise ValueError("设备资料出处字段不一致。")
        values["document_reference"] = values.pop("materials")
    unknown = values.keys() - (_TEXT_FIELDS.keys() | _NUMBER_FIELDS)
    if unknown:
        raise ValueError("设备声明含不支持的字段。")
    profile = _empty_profile()
    for key, limit in _TEXT_FIELDS.items():
        value = values.get(key, "")
        if not isinstance(value, str) or len(value) > limit:
            raise ValueError(f"{_FIELD_LABELS.get(key, '备注')}必须是限定长度的文本。")
        if any(ord(char) < 32 and char not in "\n\r\t" for char in value):
            raise ValueError("设备声明不能包含控制字符。")
        profile[key] = value.strip()
    for key in _NUMBER_FIELDS:
        value = values.get(key)
        if value in (None, ""):
            continue
        numeric = _number(value)
        if numeric is None or numeric <= 0:
            raise ValueError(f"{_FIELD_LABELS[key]}必须是有限正数。")
        if key == "srate" and not 1 <= numeric <= 10000:
            raise ValueError("设备声明的采样率范围为 1–10000 Hz。")
        if key == "scale_to_uv" and not 1e-12 <= numeric <= 1e12:
            raise ValueError("单位换算倍率超出支持范围。")
        profile[key] = numeric
    unit = profile["source_unit"]
    if unit not in ("", "ADC", *UNIT_FACTORS):
        raise ValueError("请选择 nV、uV、mV、V 或 ADC 源单位。")
    if unit in UNIT_FACTORS and profile["scale_to_uv"] is not None and profile["scale_to_uv"] != UNIT_FACTORS[unit]:
        raise ValueError("所选电压单位与换算到 µV 的倍率冲突。")
    if len(json.dumps(profile, ensure_ascii=False).encode("utf-8")) > MAX_PROFILE_BYTES:
        raise ValueError("设备声明内容过大。")
    return profile


def _public_profile(profile, *, saved, updated_at=None, read_error=None):
    missing = [{"key": key, "label": label} for key, label in _FIELD_LABELS.items()
               if profile.get(key) in (None, "")]
    result = {"schema": PROFILE_SCHEMA, "saved": saved, "profile": profile,
              "missing_fields": missing, "status": "pending_verification",
              "declaration_status": "operator_declaration", "compliance_status": "not_verified",
              "updated_at": updated_at,
              "note": "保存的是操作员声明与材料出处；不自动认证设备合规、佩戴位置或硬件标定。"}
    if read_error:
        result["read_error"] = read_error
    return result


def load_device_profile(data_dir) -> dict:
    path = Path(data_dir) / PROFILE_FILE
    if not path.exists():
        return _public_profile(_empty_profile(), saved=False)
    try:
        if path.stat().st_size > MAX_PROFILE_BYTES:
            raise ValueError("Oversized device declaration")
        data = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
        if not isinstance(data, dict) or data.get("schema") != PROFILE_SCHEMA:
            raise ValueError("Unsupported device declaration")
        profile = _normalize_profile(data.get("profile"))
        stamp = data.get("updated_at")
        if not isinstance(stamp, str) or len(stamp) > 64:
            raise ValueError("Invalid device declaration timestamp")
        return _public_profile(profile, saved=True, updated_at=stamp)
    except (OSError, UnicodeError, ValueError, TypeError):
        return _public_profile(_empty_profile(), saved=False, read_error="设备声明文件无法读取，请重新核对并保存。")


def save_device_profile(data_dir, payload) -> dict:
    profile = _normalize_profile(payload)
    stamp = _utc_now()
    record = {"schema": PROFILE_SCHEMA, "updated_at": stamp, "profile": profile,
              "declaration_status": "operator_declaration", "compliance_status": "not_verified"}
    path = Path(data_dir) / PROFILE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    # A unique sibling prevents simultaneous saves from sharing a temporary file.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".device-profile-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(record, handle, ensure_ascii=False, allow_nan=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return _public_profile(profile, saved=True, updated_at=stamp)
