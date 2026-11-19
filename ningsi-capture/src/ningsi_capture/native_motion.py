"""Offline BioMulti REC gyro evidence on the matched EEG analysis clock."""

from collections.abc import Mapping
from contextlib import closing
from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3

import numpy as np

from . import biomulti


POLICY = "biomulti-rec-gyro-clock-v1"
VENDOR_VERSION = "1.0.9-E-Release"
VENDOR_UNITS = "BioMulti vendor gyro vector magnitude; physical scale unverified"
GYRO_RATE = 12.5
MIN_GYRO_STEP_SEC = .06
MAX_GYRO_STEP_SEC = .12
ANCHOR_CHECK_SEC = 1e-6
MOTION_CONFIG = {"channel_name": ["Accel", "Gyro"], "series_name": ["X", "Y", "Z"],
                 "sample_rate": GYRO_RATE}


def _invalid(reason):
    return {"status": "unavailable", "peak": None, "samples": 0, "reason": reason}


@dataclass(frozen=True)
class NativeMotion:
    times_lsl: np.ndarray
    gyro_xyz: np.ndarray
    packet_numbers: np.ndarray

    def assess(self, eeg_times, *, threshold=None):
        """Read measured gyro packets bracketing both endpoints of an EEG span.

        The peak includes the boundary packets, at most one adjacent packet per
        edge. A missing packet, out-of-range endpoint, or bad gyro cadence leaves
        the span unavailable. The caller's cutoff is in vendor vector-magnitude
        units; without a cutoff the returned peak is observational.
        """
        try:
            eeg = np.asarray(eeg_times, dtype=float)
            cutoff = None if threshold is None else float(threshold)
        except (TypeError, ValueError, OverflowError):
            return _invalid("invalid_motion_assessment_input")
        if (eeg.ndim != 1 or len(eeg) < 2 or not np.isfinite(eeg).all()
                or np.any(np.diff(eeg) <= 0)
                or cutoff is not None and (not np.isfinite(cutoff) or cutoff <= 0)):
            return _invalid("invalid_motion_assessment_input")

        left = np.searchsorted(self.times_lsl, eeg[0], side="right") - 1
        right = np.searchsorted(self.times_lsl, eeg[-1], side="left")
        if left < 0 or right >= len(self.times_lsl):
            return _invalid("gyro_does_not_bracket_eeg_span")
        included = slice(left, right + 1)
        steps = np.diff(self.times_lsl[included])
        if (np.any((steps < MIN_GYRO_STEP_SEC) | (steps > MAX_GYRO_STEP_SEC))
                or np.any(np.diff(self.packet_numbers[included]) != 1)):
            return _invalid("gyro_packet_gap_in_eeg_span")

        peak = float(np.max(np.hypot.reduce(self.gyro_xyz[included], axis=1)))
        if not np.isfinite(peak):
            return _invalid("invalid_gyro_magnitude")
        status = "observed" if cutoff is None else "motion" if peak > cutoff else "quiet"
        return {"status": status, "peak": peak, "samples": int(right - left + 1), "reason": None}


def _eeg_clock(eeg_packets, analysis_times):
    points = np.asarray(tuple(eeg_packets), dtype=float)
    axis = np.asarray(analysis_times, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 5 or len(points) < 150
            or axis.ndim != 1 or not np.isfinite(points).all()):
        raise ValueError("missing_matched_eeg_anchors")
    metadata = points[:, :3]
    if ((metadata < 0).any() or (metadata >= 2**53).any()
            or np.any(metadata != np.floor(metadata)) or np.any(np.diff(metadata, axis=0) <= 0)
            or np.any(np.diff(metadata[:, 0]) < biomulti.PACKET_SIZE)):
        raise ValueError("invalid_matched_eeg_anchors")
    indices = metadata[:, 0].astype(np.int64) + biomulti.PACKET_SIZE - 1
    if indices[-1] >= len(axis):
        raise ValueError("eeg_anchor_index_out_of_range")
    matches = axis[indices]
    ends = points[:, 4]
    if not np.isfinite(matches).all() or np.any(np.abs(matches - ends) > ANCHOR_CHECK_SEC):
        raise ValueError("eeg_analysis_clock_anchor_mismatch")

    ticks = metadata[:, 2]
    slope = (ends[-1] - ends[0]) / (ticks[-1] - ticks[0])
    if not .00095 <= slope <= .00105:
        raise ValueError("eeg_device_clock_scale_invalid")
    projected = ends[0] + slope * (ticks - ticks[0])
    if np.any(np.abs(ends - projected) > ANCHOR_CHECK_SEC):
        raise ValueError("eeg_device_clock_not_affine")
    return metadata[:, 1:].astype(np.int64), float(ends[0]), float(slope)


def _rec_gyro(database, record_id, created_ms, eeg_metadata, origin, slope):
    package = Path(database).resolve().parent / "resources/app/package.json"
    app = json.loads(package.read_text(encoding="utf-8-sig"))
    if not isinstance(app, dict) or app.get("name") != "BioMultiLite" or app.get("version") != VENDOR_VERSION:
        raise ValueError("unsupported_biomulti_version")
    first_tick, last_tick = eeg_metadata[[0, -1], 1]
    anchors = dict(eeg_metadata.tolist())
    seen_anchors = set()
    times, gyro, packet_numbers = [], [], []
    previous = None
    with closing(biomulti._reader(database)) as connection:
        connection.execute("BEGIN")
        record = connection.execute(
            "SELECT recoredCreateTime,recoredEndTime,configText FROM record WHERE id=?", (record_id,)
        ).fetchone()
        if record is None or record[0] != created_ms or not record[1]:
            raise ValueError("native_record_not_closed_or_binding_mismatch")
        biomulti._check_config(record[2])
        config = json.loads(record[2])
        if config.get("motion") != MOTION_CONFIG:
            raise ValueError("unrecognized_native_gyro_config")

        for (encoded,) in connection.execute("SELECT data FROM source WHERE recordId=? ORDER BY id", (record_id,)):
            frames = json.loads(encoded)
            if not isinstance(frames, list):
                raise ValueError("invalid_native_packet_chunk")
            for frame in frames:
                if not isinstance(frame, dict):
                    raise ValueError("invalid_native_packet")
                kind, number, tick = (frame.get(key) for key in ("pkg_type", "pkgnum", "time_mark"))
                if type(kind) is not int or kind not in (1, 3):
                    continue
                if type(number) is not int or type(tick) is not int or number < 0 or tick < 0:
                    raise ValueError("invalid_native_packet_identity")
                if kind == 1:
                    if number in anchors:
                        if (number in seen_anchors or anchors[number] != tick or frame.get("isLosspkg") is not False
                                or frame.get("eeg_channel") != 2 or frame.get("eeg_data_num") != biomulti.PACKET_SIZE):
                            raise ValueError("native_eeg_anchor_record_mismatch")
                        seen_anchors.add(number)
                    continue
                if not first_tick <= tick <= last_tick:
                    continue
                if previous is not None and (number <= previous[0] or tick <= previous[1]):
                    raise ValueError("nonmonotonic_native_gyro_packets")
                previous = (number, tick)
                values = [frame.get(f"gyro_{axis}") for axis in "xyz"]
                if (frame.get("isLosspkg") is not False
                        or any(type(value) not in (int, float) or not np.isfinite(value) for value in values)):
                    continue
                times.append(origin + slope * (tick - first_tick))
                gyro.append(values)
                packet_numbers.append(number)

    if len(seen_anchors) != len(anchors):
        raise ValueError("matched_eeg_anchors_missing_from_native_record")
    if len(times) < 2:
        raise ValueError("insufficient_native_gyro_packets")
    arrays = (np.asarray(times), np.asarray(gyro), np.asarray(packet_numbers, dtype=np.int64))
    if not np.any(arrays[1] != arrays[1][0]):
        raise ValueError("constant_native_gyro_trace")
    for array in arrays:
        array.setflags(write=False)
    return NativeMotion(*arrays)


def load_native_motion(database, binding, eeg_packets, analysis_times):
    """Load gyro evidence from the same closed REC that matched the EEG.

    EEG packet rows are the unmodified output of `biomulti.align`. This loader
    consumes the iterable and returns `(None, audit)` if provenance cannot be
    established. It never adjusts EEG time, creates samples, or fills gyro gaps.
    """
    audit = {"policy": POLICY, "status": "unavailable", "units": VENDOR_UNITS}
    if (not isinstance(binding, Mapping) or type(binding.get("record_id")) is not int
            or type(binding.get("created_ms")) is not int or binding["record_id"] <= 0
            or binding["created_ms"] <= 0):
        return None, {**audit, "reason": "native_record_binding_missing"}
    record_id = binding["record_id"]
    audit["record_id"] = record_id
    try:
        eeg, origin, slope = _eeg_clock(eeg_packets, analysis_times)
        motion = _rec_gyro(database, record_id, binding["created_ms"], eeg, origin, slope)
    except (OSError, sqlite3.Error) as error:
        return None, {**audit, "reason": "native_motion_record_unreadable", "detail": str(error)}
    except (ValueError, OverflowError, TypeError) as error:
        return None, {**audit, "reason": str(error)}
    return motion, {**audit, "status": "available", "vendor_package_version": VENDOR_VERSION,
                    "sample_rate_hz": GYRO_RATE, "packets": len(motion.times_lsl),
                    "time_method": "matched REC EEG device-tick clock; no gyro interpolation"}
