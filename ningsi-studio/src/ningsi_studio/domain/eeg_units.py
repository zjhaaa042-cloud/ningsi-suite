"""Per-session confirmation for EEG channels with missing voltage-unit metadata."""

import math

from ningsi.acquisition.eeg import UNIT_FACTORS, unit_conflict
from ningsi.signal.live_artifacts import channel_layout


def confirmation(value):
    if value is None:
        return None
    if not isinstance(value, dict) or value.get("source_unit") not in (*UNIT_FACTORS, "ADC"):
        raise ValueError("请选择 EEG 电压单位或 ADC 计数")
    unit = value["source_unit"]
    if unit == "ADC":
        factor = value.get("scale_to_uv")
        if type(factor) not in (float, int) or not math.isfinite(factor) or factor <= 0:
            raise ValueError("请填写经校准的正数 µV/计数")
    else:
        factor = UNIT_FACTORS[unit]
        if "scale_to_uv" in value and value["scale_to_uv"] != factor:
            raise ValueError("换算倍率与所选电压单位不一致")
    return {"source_unit": unit, "scale_to_uv": float(factor), "origin": "operator_confirmed"}


def eeg_indices(descriptor):
    labels = descriptor.get("channel_labels") or []
    return channel_layout(labels, descriptor.get("channel_count") or len(labels))[0]


def verify_descriptor(calibration, descriptor):
    units = descriptor.get("declared_units") or []
    selected = [units[i] if i < len(units) else "" for i in eeg_indices(descriptor)]
    conflict = unit_conflict({**calibration, "declared_units": selected})
    if conflict:
        raise ValueError(conflict)
