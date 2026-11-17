"""The user's BSense-R device parameters, taken from the supplied product manual.

This is an application device configuration, not a physical calibration record.
It never rewrites live channel labels, references, timestamps or signal quality.
"""

from collections.abc import Mapping
import math
import re

from ningsi.acquisition.eeg import is_eeg_stream, unit_conflict


MODEL = "BSense-R"
SRATE = 250.0
CHANNELS = 2
SOURCE_UNIT = "uV"
SCALE_TO_UV = 1.0
ORIGIN = "manufacturer_manual"
DOCUMENT_REFERENCE = "BSense-R 产品手册.pdf，PDF第21–23页"
DOCUMENT_SHA256 = "f83664699b5986b012ea4f2b79eed6837f577dab793373cec54cf3b9ef3e7635"
_NAME = re.compile(r"^(?:biomulti[\s_-]*lite|bsense[\s_-]*r)[\s_-]+eeg(?:[\s_-].*)?$", re.I)
_SIMULATION = re.compile(r"(^|[^a-z])(sim|simulation|synthetic|demo)([^a-z]|$)", re.I)


def is_bsense_eeg(descriptor):
    """Recognise only the configured vendor's EEG, including API source rows."""
    if not isinstance(descriptor, Mapping):
        return False
    if descriptor.get("simulated") is True or descriptor.get("real") is False or descriptor.get("supported") is False:
        return False
    kind = descriptor.get("kind")
    if kind == "lsl":
        kind = descriptor.get("stream_kind") or descriptor.get("stream_type")
    else:
        kind = kind or descriptor.get("stream_kind") or descriptor.get("stream_type")
    name = str(descriptor.get("name") or descriptor.get("device") or "").strip()
    identity = f"{name} {descriptor.get('source_id') or ''}"
    return (str(kind or "").strip().casefold() == "eeg" and bool(_NAME.fullmatch(name))
            and is_eeg_stream(kind, name) and not _SIMULATION.search(identity))


def _number(value):
    if isinstance(value, bool):
        return math.nan
    try:
        return float(value)
    except (ValueError, TypeError, OverflowError):
        return math.nan


def configuration(descriptor):
    """Return fixed manual settings, or None for a different device.

    A recognised device with conflicting metadata raises instead of falling back
    to another profile. Nominal metadata takes precedence over display fallbacks.
    """
    if not is_bsense_eeg(descriptor):
        return None
    count = _number(descriptor.get("channel_count") if "channel_count" in descriptor else descriptor.get("channels"))
    rate = _number(descriptor.get("nominal_srate") if "nominal_srate" in descriptor else descriptor.get("srate"))
    if not math.isfinite(count) or count != CHANNELS:
        raise ValueError("BSense-R 手册配置为 2 个 EEG 通道，当前流通道数不匹配。")
    if not math.isfinite(rate) or not math.isclose(rate, SRATE, rel_tol=0, abs_tol=1e-6):
        raise ValueError("BSense-R 手册采样率为 250 Hz，当前流的标称采样率不匹配。")
    conflict = unit_conflict({"source_unit": SOURCE_UNIT, "scale_to_uv": SCALE_TO_UV,
                              "declared_units": descriptor.get("declared_units") or []})
    if conflict:
        raise ValueError("BSense-R 手册单位为 µV，当前流声明的单位不匹配，请检查采集软件设置。")
    return {"id": "bsense-r", "manufacturer": "荣脑科技 / LOONG BRAIN", "model": MODEL,
            "channels": CHANNELS, "srate": SRATE, "source_unit": SOURCE_UNIT,
            "scale_to_uv": SCALE_TO_UV, "origin": ORIGIN,
            "document_reference": DOCUMENT_REFERENCE, "document_sha256": DOCUMENT_SHA256}
