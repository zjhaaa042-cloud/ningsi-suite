"""Timing estimates for packet-timestamped EEG using available release times.

Raw timestamps and sample values are never rewritten. This is an optional
analysis policy, not packet-loss recovery or a claim of hardware timing accuracy.
"""

import numpy as np


STRICT_TIMING = "source-timestamps-v1"
PACKET_TIMING = "bounded-nominal-50ms-v1"
BIOMULTI_REC_TIMING = "biomulti-rec-device-clock-v1"
BIOMULTI_LSL_TIMING = "biomulti-lsl-packet-clock-v1"
# The verified publisher emits 10 float32 samples per chunk with one release
# timestamp. This bound is on transport jitter, not an allowable EEG gap.
BIOMULTI_MAX_DELAY = .250
BIOMULTI_CONFIRM_SAMPLES = 250
BIOMULTI_WARM_SAMPLES = 1000


def biomulti_packet_clock(timestamps, *, first_index=0):
    """Estimate the sample clock of this 2 x 250 Hz outlet on fixed one-second epochs.

    An epoch uses its own releases and the preceding anchor. A burst may need
    up to one additional second of *release times* to confirm catch-up. Callers
    doing live scoring restrict these observations to the probe-time snapshot.
    Missing or unresolved time is NaN.
    """
    raw = np.asarray(timestamps, dtype=float)
    result = np.full(raw.shape, np.nan)
    grid = np.arange(250) / 250.
    previous_anchor = previous_release = previous_phase = None
    last_valid = -np.inf
    for start in range((-first_index) % 250, len(raw) - 249, 250):
        epoch = raw[start:start + 250]
        if not np.isfinite(epoch).all():
            previous_anchor = previous_release = previous_phase = None
            continue
        # In a ten-sample chunk each of the nine internal intervals is 4 ms.
        # Detect the phase, including a recording that started mid-packet.
        edges = np.flatnonzero(np.abs(np.diff(epoch) - .004) > .0002)
        phases = np.unique(edges % 10)
        if len(phases) > 1:
            previous_anchor = previous_release = previous_phase = None
            continue
        phase = int(phases[0]) if len(phases) else (previous_phase if previous_phase is not None else 9)
        ends = np.arange(phase, 250, 10)
        release = epoch[ends]
        offsets = release - grid[ends]
        # The early arrivals have the least transport delay. Joining successive
        # anchors distributes oscillator drift without following delivery bursts.
        anchor = float(np.quantile(offsets, .10)) + 1.
        left = previous_anchor if previous_anchor is not None else anchor - 1.
        intervals = np.diff(release, prepend=previous_release if previous_release is not None else release[0] - .04)
        coherent = (np.all((intervals > 0) & (intervals <= BIOMULTI_MAX_DELAY))
                    and np.ptp(offsets) <= BIOMULTI_MAX_DELAY and abs(anchor - left - 1.) <= .04
                    and (previous_phase is None or phase == previous_phase))
        if coherent:
            estimated = left + grid * (anchor - left)
            if estimated[0] > last_valid and np.max(np.abs(estimated - epoch)) <= BIOMULTI_MAX_DELAY:
                result[start:start + 250] = estimated
                # A long silence is explained by batching only if subsequent
                # releases actually catch up. Preserve an unresolved gap at the
                # first affected packet, including gaps near the end of an epoch.
                for packet in np.flatnonzero(intervals > .100):
                    before = (offsets[packet - 1] if packet else
                              previous_release - (grid[ends[0]] - .040))
                    end_id = start + ends[packet]
                    later = np.arange(end_id, min(len(raw), end_id + BIOMULTI_CONFIRM_SAMPLES + 1), 10)
                    # Only consecutive packet releases can explain a burst.
                    times = raw[later]
                    deltas = np.diff(times)
                    breaks = np.flatnonzero(~np.isfinite(deltas) | (deltas <= 0) | (deltas > BIOMULTI_MAX_DELAY))
                    count = int(breaks[0] + 1) if breaks.size else len(later)
                    offsets_after = times[:count] - (later[:count] - start) / 250.
                    if np.min(offsets_after) - before > .040:
                        result[start + max(0, ends[packet] - 9):start + ends[packet] + 1] = np.nan
                valid = result[start:start + 250]
                finite = valid[np.isfinite(valid)]
                if finite.size:
                    last_valid = finite[-1]
        previous_anchor, previous_release, previous_phase = anchor, release[-1], phase
    return result


def analysis_timestamps(timestamps, rate, policy=None, *, first_index=0):
    stamps = np.asarray(timestamps, dtype=float)
    audit = {"policy": policy or STRICT_TIMING, "applied": False, "max_shift_sec": 0.}
    if policy in (None, STRICT_TIMING):
        return stamps, audit
    if policy == BIOMULTI_LSL_TIMING:
        if rate != 250:
            raise ValueError("BioMulti 分包时钟仅适用于 250 Hz EEG。")
        estimated = biomulti_packet_clock(stamps, first_index=first_index)
        valid = np.isfinite(estimated)
        audit.update(applied=bool(valid.all() and valid.size),
                     max_shift_sec=float(np.max(np.abs(estimated[valid] - stamps[valid]))) if valid.any() else 0.)
        return estimated, audit
    if policy != PACKET_TIMING:
        raise ValueError("未知的脑电时间处理方式。")
    if (not np.isfinite(rate) or rate <= 0 or stamps.size < 2
            or not np.isfinite(stamps).all()):
        return stamps, audit
    diffs = np.diff(stamps)
    # Keep the original failure visible to the strict feature gate. Do not
    # close a large gap, bridge a reset, resample values, or invent samples.
    if (np.any(diffs > .1) or np.any(diffs < -.05)
            or abs((stamps[-1] - stamps[0]) * rate / (len(stamps) - 1) - 1) > .05):
        return stamps, audit
    grid = np.arange(len(stamps), dtype=float) / rate
    estimated = float(np.median(stamps - grid)) + grid
    shift = float(np.max(np.abs(estimated - stamps)))
    if shift > .05:
        return stamps, audit
    audit.update(applied=shift > 1e-8, max_shift_sec=shift)
    return estimated, audit
