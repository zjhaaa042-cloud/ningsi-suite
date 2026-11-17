"""EEG stream selection and conversion from device units into microvolts."""

import re


UNIT_FACTORS = {"nV": .001, "uV": 1., "mV": 1000., "V": 1000000.}
_UNITS = {
    "nv": "nV", "nanovolt": "nV", "nanovolts": "nV",
    "uv": "uV", "microvolt": "uV", "microvolts": "uV", "微伏": "uV",
    "mv": "mV", "millivolt": "mV", "millivolts": "mV", "毫伏": "mV",
    "v": "V", "volt": "V", "volts": "V", "伏": "V",
    "adc": "ADC", "count": "ADC", "counts": "ADC", "adccount": "ADC",
    "adccounts": "ADC", "adc计数": "ADC", "计数": "ADC", "lsb": "ADC",
}


def is_eeg_stream(stream_type, name, channel_format=None):
    """A vendor/name fallback must not override an explicit secondary stream."""
    if channel_format is not None and channel_format not in (1, 2, 4, 5, 6, 7):
        return False  # liblsl numeric formats; strings cannot contain EEG samples.
    kind = str(stream_type or "").strip().casefold()
    name = str(name or "").casefold()
    if re.search(r"marker|trigger|event|metric|quality|impedance|band[ _-]?power|spectrum", kind + " " + name):
        return False
    if kind in ("", "unknown", "data", "signal", "biosignal", "raw", "exg"):
        kind = name
    return "eeg" in kind or "electroencephal" in kind or "脑电" in kind


def normalize_unit(value):
    key = re.sub(r"[\s_-]", "", str(value or "").casefold().replace("μ", "u"))
    if key in ("", "unknown", "unspecified", "n/a", "na", "none", "未知", "未指定"):
        return None
    return _UNITS.get(key, key)


_unit = normalize_unit  # Existing live artifact estimators import this name.


def source_type_conflict(source):
    if source.get("type") and not is_eeg_stream(source["type"], source.get("name")):
        return f"设备声明的 {source['type']} 流与原始 EEG 类型不符。"
    return ""


def unit_conflict(source):
    """Report a proven conflict. An operator may still choose raw-only capture."""
    factor = source.get("scale_to_uv")
    if factor is None:
        return ""
    selected = normalize_unit(source.get("source_unit"))
    if selected in UNIT_FACTORS and factor != UNIT_FACTORS[selected]:
        return f"所选单位 {selected} 与记录的 µV 换算倍率 {factor} 不一致。"
    for channel, raw in enumerate(source.get("declared_units") or (), 1):
        declared = normalize_unit(raw)
        if declared is None:
            continue
        if declared in UNIT_FACTORS and factor == UNIT_FACTORS[declared] and selected in (None, declared):
            continue
        if declared == selected == "ADC":
            continue
        return f"通道 {channel} 的设备声明单位为 {raw}，与所选单位或换算倍率不符。请核对厂商 LSL 输出设置。"
    return ""
