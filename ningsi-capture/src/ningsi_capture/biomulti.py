"""Match BioMulti Lite 1.0.9 two-channel REC packets to their original LSL samples."""

from contextlib import closing
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import socket
import sqlite3

import numpy as np
from ningsi.acquisition.eeg import UNIT_FACTORS, normalize_unit, unit_conflict


PACKET_SIZE = 10
UNIT_EVIDENCE = ("BioMulti Lite 1.0.9-E-Release: EEG absolute power is labelled uV^2/Hz; "
                 "the spectral path uses the decoded EEG after DC removal and the LSL publisher "
                 "copies the same raw samples without rescaling. Software-inferred, not hardware-calibrated.")


def confirmed_calibration(source, unit, evidence, factor=None):
    evidence = str(evidence or "").strip()
    scale = factor if unit == "ADC" else UNIT_FACTORS.get(unit)
    if not evidence or scale is None or not np.isfinite(scale) or scale <= 0:
        raise ValueError("请填写已核实的单位和确认依据。ADC 计数还需要正的 µV/计数倍率。")
    if unit != "ADC" and factor is not None and factor != scale:
        raise ValueError("电压单位与换算倍率不一致。")
    change = {"source_unit": unit, "scale_to_uv": scale, "unit_confirmation": evidence,
              "unit_confirmation_stage": "at_export"}
    conflict = unit_conflict({**source, **change})
    if conflict:
        raise ValueError(conflict)
    return change


def supported(source):
    return (source.get("kind") == "lsl" and source.get("data_origin") == "real"
            and source.get("channels") == 2 and source.get("srate") == 250
            and bool(re.fullmatch(r"BioMulti\s*Lite\s+EEG(?:-[\w-]+)?", source.get("name", ""), re.I)))


def local_publisher(source):
    if (not supported(source) or source.get("type") != "EEG" or not re.fullmatch(
            r"BioMulti Lite EEG-[a-fA-F0-9]{6}", source["name"])
            or str(source.get("hostname", "")).casefold() != socket.gethostname().casefold()):
        return False
    package = default_database().parent / "resources/app/package.json"
    try:
        app = json.loads(package.read_text(encoding="utf-8-sig"))
        return app.get("name") == "BioMultiLite" and app.get("version") == "1.0.9-E-Release"
    except (OSError, ValueError):
        return False


def automatic_unit(source):
    """Prefer explicit stream metadata; scope the fallback to the verified local publisher."""
    units = [normalize_unit(u) for u in (source.get("declared_units") or ())]
    if any(units):
        unit = units[0]
        if len(units) == source["channels"] and len(set(units)) == 1 and unit in UNIT_FACTORS:
            return {"source_unit": unit, "scale_to_uv": UNIT_FACTORS[unit], "unit_basis": "stream_metadata"}
        return {}
    if local_publisher(source):
        return {"source_unit": "uV", "scale_to_uv": 1., "unit_basis": "biomulti_software_inference",
                "unit_evidence": UNIT_EVIDENCE}
    return {}


def default_database():
    return Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "BioMultiLite" / "data.db"


def _reader(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=2)


def active_record(path):
    with closing(_reader(path)) as db:
        rows = db.execute("SELECT id,recoredCreateTime,recoredEndTime,configText FROM record "
                          "ORDER BY id DESC LIMIT 1").fetchall()
    if not rows or rows[0][2]:
        raise ValueError("请先在 BioMulti Lite 点 REC 开始原生录制，再开始凝思双录。")
    record_id, created, _, config = rows[0]
    if config and json.loads(config):
        _check_config(config)
    return {"record_id": record_id, "created_ms": created}


def _check_config(config):
    settings = json.loads(config or "{}")
    if not isinstance(settings, dict) or settings.get("eeg_channel_count") != 2 or settings.get("eeg_sample_rate") != 250:
        raise ValueError("此对时方式需要 BioMulti 原生记录为双通道 250 Hz。")


@dataclass
class Packet:
    number: int
    tick_ms: int
    start: int | None = None
    outlet_end: float | None = None
    ambiguous: bool = False


def _packets(path, record_id, created_ms):
    signatures = {}
    with closing(_reader(path)) as db:
        db.execute("BEGIN")
        record = db.execute("SELECT recoredCreateTime,recoredEndTime,configText FROM record WHERE id=?",
                            (record_id,)).fetchone()
        if not record or (created_ms is not None and created_ms != record[0]):
            raise ValueError("原生记录编号或创建时间与本次双录不符。")
        if not record[1]:
            raise ValueError("请先在 BioMulti Lite 停止并保存本次 REC，再导出。")
        _check_config(record[2])
        total = 0
        for data, in db.execute("SELECT data FROM source WHERE recordId=? ORDER BY id", (record_id,)):
            for frame in json.loads(data):
                if frame.get("pkg_type") != 1:
                    continue
                total += 1
                if (frame.get("isLosspkg") is not False or frame.get("eeg_data_num") != PACKET_SIZE
                        or frame.get("eeg_channel") != 2):
                    continue
                values = np.asarray(frame.get("brain_elec_channel"), dtype="<f4")
                if values.shape != (2, PACKET_SIZE) or not np.isfinite(values).all():
                    continue
                if not isinstance(frame.get("pkgnum"), int) or not isinstance(frame.get("time_mark"), int):
                    continue
                # This publisher reverses its two REC channels and sends float32.
                # Compare in the capture's lossless float64 representation.
                signature = values[::-1].T.astype("<f8").tobytes()
                if signature in signatures:
                    signatures[signature].ambiguous = True
                else:
                    signatures[signature] = Packet(frame["pkgnum"], frame["time_mark"])
    return signatures, total


def _match(connection, signatures):
    prefixes = {signature[:16] for signature in signatures}
    cursor = connection.execute("SELECT sample_index,values_f64,corrected_timestamp FROM samples ORDER BY sample_index")
    tail = []
    while batch := cursor.fetchmany(8192):
        rows = tail + batch
        for i in range(len(rows) - PACKET_SIZE + 1):
            start, values, _ = rows[i]
            if values not in prefixes:
                continue
            group = rows[i:i + PACKET_SIZE]
            if group[-1][0] != start + PACKET_SIZE - 1:
                continue
            packet = signatures.get(b"".join(row[1] for row in group))
            if packet is None:
                continue
            if packet.start is not None:
                packet.ambiguous = True
            else:
                packet.start = start
                if all(row[2] is not None and np.isfinite(row[2]) for row in group):
                    packet.outlet_end = group[-1][2]
        tail = rows[-(PACKET_SIZE - 1):]
    return sorted((p for p in signatures.values() if not p.ambiguous and p.outlet_end is not None),
                  key=lambda p: p.start)


def _clock_fit(device, outlet):
    # Low-delay packets in successive four-second intervals anchor the device
    # oscillator to the task's LSL clock. A release burst must not shift events.
    blocks = np.floor((device - device[0]) / 4).astype(int)
    anchors = []
    for block in np.unique(blocks):
        ids = np.flatnonzero(blocks == block)
        if len(ids) >= 50:
            delays = outlet[ids] - device[ids]
            anchors.append(ids[np.argsort(delays)[len(ids) // 10]])
    if len(anchors) < 3:
        raise ValueError("可唯一匹配的同步数据太短，需要至少三个有足够样本的 4 秒段。")
    origin = device[anchors[0]]
    x = device[anchors] - origin
    slope, intercept = np.polyfit(x, outlet[anchors], 1)
    residual = np.abs(outlet[anchors] - (intercept + slope * x))
    if not .95 <= slope <= 1.05 or np.max(residual) > .050:
        raise ValueError("设备时钟漂移或同步锚点误差超限，无法可靠对齐任务。")
    return (intercept + slope * (device - origin)), float(slope), float(np.max(residual)), len(anchors)


def align(connection, source, path, record_id, created_ms=None):
    if not supported(source):
        raise ValueError("仅支持 BioMulti Lite 双通道 250 Hz 真机 EEG 的原生 REC 对时。")
    signatures, total = _packets(path, record_id, created_ms)
    packets = _match(connection, signatures)
    if len(packets) < 150:
        raise ValueError("两份记录的双通道样本匹配不足，请确认设备、时段及原生 REC 编号。")
    device = np.array([p.tick_ms / 1000. for p in packets])
    numbers = np.array([p.number for p in packets], dtype=np.int64)
    starts = np.array([p.start for p in packets], dtype=np.int64)
    jumps = np.diff(numbers)
    intervals = np.diff(device)
    if (np.any(jumps <= 0) or np.any(intervals <= 0) or np.any(np.diff(starts) < PACKET_SIZE)):
        raise ValueError("匹配包出现序号或设备时钟重置、倒退或重叠，请分段复核。")
    contiguous = (jumps == 1) & (np.diff(starts) == PACKET_SIZE)
    if not contiguous.any():
        raise ValueError("两份记录没有连续的共同 EEG 包。")
    # Device ticks are integer milliseconds. Use the average period of actual
    # adjacent packets for within-packet points; missing packets stay missing.
    nominal = PACKET_SIZE / 250.
    stable = contiguous & (np.abs(intervals - nominal) <= .002)
    if not stable.any():
        raise ValueError("设备包内采样率与 250 Hz 不符。")
    outlet = np.array([p.outlet_end for p in packets])
    ends, slope, residual, anchors = _clock_fit(device, outlet)
    step = slope * float(np.mean(intervals[stable])) / PACKET_SIZE
    if abs(step * 250 - 1) > .05:
        raise ValueError("实测设备采样率与 250 Hz 偏差超过 5%。")
    n = connection.execute("SELECT COALESCE(MAX(sample_index)+1,0) FROM samples").fetchone()[0]
    times = np.full(n, np.nan)
    within = (np.arange(PACKET_SIZE) - (PACKET_SIZE - 1)) * step
    for start, end in zip(starts, ends):
        times[start:start + PACKET_SIZE] = end + within
    # Mark a break even if both writers lost the very same packet and there is
    # no empty sample_index between them. Every crossing context must fail QC.
    times[starts[1:][~stable & (np.diff(starts) == PACKET_SIZE)]] = np.nan
    audit = {"record_id": record_id, "native_eeg_packets": total, "matched_packets": len(packets),
             "matched_samples": len(packets) * PACKET_SIZE, "clocked_samples": int(np.isfinite(times).sum()),
             "sequence_breaks": int((jumps != 1).sum()), "continuity_breaks": int((~stable).sum()),
             "anchor_count": anchors, "device_to_lsl_slope": slope, "sample_rate_hz": 1 / step,
             "anchor_max_residual_ms": residual * 1000,
             "release_minus_device_ms_p05_p50_p95": (np.percentile(outlet - ends, [5, 50, 95]) * 1000).tolist(),
             "method": "unique exact two-channel packets; device millisecond ticks; offline affine LSL alignment",
             "absolute_latency": "transport and physical event latency not calibrated"}
    return times, audit, zip(starts, numbers, (p.tick_ms for p in packets), outlet, ends)
