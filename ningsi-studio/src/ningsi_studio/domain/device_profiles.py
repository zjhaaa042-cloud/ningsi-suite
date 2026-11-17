"""Source-scoped operator declarations; never hardware certification or live QC."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from ningsi.acquisition.eeg import UNIT_FACTORS


SCHEMA = "ningsi-device-profile-v1"
MAX_PROFILE_BYTES = 24 * 1024
MAX_RECORD_BYTES = MAX_PROFILE_BYTES + 4096
TEXT_FIELDS = {
    "manufacturer": 160, "model": 160, "electrodes": 256, "reference": 256,
    "source_unit": 16, "unit_basis": 1500, "document_reference": 2000, "notes": 2000,
}
NUMBER_FIELDS = {"srate", "scale_to_uv"}
FIELD_LABELS = {
    "manufacturer": "设备厂家", "model": "实际设备型号", "electrodes": "实际电极位置",
    "reference": "参考电极", "srate": "标称采样率", "source_unit": "源数值单位",
    "scale_to_uv": "换算到 µV 的倍率", "unit_basis": "单位与换算依据",
    "document_reference": "设备资料出处",
}


def normalize_device(value):
    if (not isinstance(value, str) or not value.strip() or len(value) > 120
            or any(ord(char) < 32 for char in value)):
        raise ValueError("请选择或填写有效的 EEG 来源键。")
    value = value.strip()
    if value != "sim-bsense" and not (value.startswith("lsl:") and value[4:].strip()):
        raise ValueError("来源键必须为 lsl:<流名称> 或显式模拟源 sim-bsense。")
    return value


def empty_profile():
    return {**dict.fromkeys(TEXT_FIELDS, ""), **dict.fromkeys(sorted(NUMBER_FIELDS), None)}


def normalize_profile(payload):
    """Shared with the historical declaration endpoint to preserve its validation."""
    if not isinstance(payload, dict):
        raise ValueError("设备声明必须是 JSON 对象。")
    values = dict(payload)
    if "materials" in values:
        if "document_reference" in values and values["materials"] != values["document_reference"]:
            raise ValueError("设备资料出处字段不一致。")
        values["document_reference"] = values.pop("materials")
    if values.keys() - (TEXT_FIELDS.keys() | NUMBER_FIELDS):
        raise ValueError("设备声明含不支持的字段。")
    profile = empty_profile()
    for key, limit in TEXT_FIELDS.items():
        value = values.get(key, "")
        if not isinstance(value, str) or len(value) > limit:
            raise ValueError(f"{FIELD_LABELS.get(key, '备注')}必须是限定长度的文本。")
        if any(ord(char) < 32 and char not in "\n\r\t" for char in value):
            raise ValueError("设备声明不能包含控制字符。")
        profile[key] = value.strip()
    for key in NUMBER_FIELDS:
        value = values.get(key)
        if value in (None, ""):
            continue
        try:
            numeric = float(value) if type(value) in (int, float) else math.nan
        except OverflowError:
            numeric = math.nan
        if not math.isfinite(numeric) or numeric <= 0:
            raise ValueError(f"{FIELD_LABELS[key]}必须是有限正数。")
        if key == "srate" and not 1 <= numeric <= 10000:
            raise ValueError("设备声明的采样率范围为 1–10000 Hz。")
        if key == "scale_to_uv" and not 1e-12 <= numeric <= 1e12:
            raise ValueError("单位换算倍率超出支持范围。")
        profile[key] = numeric
    unit = profile["source_unit"]
    if unit not in ("", "ADC", *UNIT_FACTORS):
        raise ValueError("请选择 nV、uV、mV、V 或 ADC 源单位。")
    if unit in UNIT_FACTORS and profile["scale_to_uv"] not in (None, UNIT_FACTORS[unit]):
        raise ValueError("所选电压单位与换算到 µV 的倍率冲突。")
    if len(json.dumps(profile, ensure_ascii=False).encode("utf-8")) > MAX_PROFILE_BYTES:
        raise ValueError("设备声明内容过大。")
    return profile


def _public(device, profile, *, saved=False, updated_at=None, read_error=None):
    missing = [{"key": key, "label": label} for key, label in FIELD_LABELS.items()
               if profile.get(key) in (None, "")]
    result = {"schema": SCHEMA, "device": device, "saved": saved, "profile": profile,
              "missing_fields": missing, "status": "pending_verification", "updated_at": updated_at,
              "declaration_status": "operator_declaration", "compliance_status": "not_verified",
              "note": "资料按所选来源单独保存，仅为操作员声明；不代表设备合规、导联、标定或信号质量已验证。"}
    if read_error:
        result["read_error"] = read_error
    return result


def _path(data_dir, device):
    # Never use caller text as a file path; source keys may contain path-like characters.
    digest = hashlib.sha256(device.encode("utf-8")).hexdigest()
    return Path(data_dir) / "device_profiles" / (digest + ".json")


def load_device_profile(data_dir, device):
    device = normalize_device(device)
    path = _path(data_dir, device)
    if not path.exists():
        return _public(device, empty_profile())
    try:
        if path.stat().st_size > MAX_RECORD_BYTES:
            raise ValueError("oversized declaration")
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("schema") != SCHEMA or record.get("device") != device:
            raise ValueError("device identity mismatch")
        stamp = record.get("updated_at")
        if not isinstance(stamp, str) or len(stamp) > 64:
            raise ValueError("invalid timestamp")
        return _public(device, normalize_profile(record.get("profile")), saved=True, updated_at=stamp)
    except (OSError, UnicodeError, ValueError, TypeError, AttributeError):
        return _public(device, empty_profile(), read_error="设备资料无法读取，请核对该来源后重新保存。")


def save_device_profile(data_dir, device, payload):
    device = normalize_device(device)
    profile = normalize_profile(payload)
    stamp = datetime.now(timezone.utc).isoformat()
    record = {"schema": SCHEMA, "device": device, "profile": profile, "updated_at": stamp}
    path = _path(data_dir, device)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".profile-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(record, handle, ensure_ascii=False, allow_nan=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return _public(device, profile, saved=True, updated_at=stamp)
