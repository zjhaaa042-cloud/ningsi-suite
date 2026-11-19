"""Session-local audit of prompted artifact signatures. EEG is never corrected here."""

from collections import Counter

import numpy as np

from ningsi.signal import conditioning

from .spectrum_quality import MOTION_CLOCK_MARGIN_SEC, MOTION_QUIET_CUTOFF


POLICY = "capture-prompted-artifact-audit-v1"
ACTIONS = ("blink", "eyes", "jaw", "earclip", "forehead", "head")
FIT_REPS = (1, 2, 3)
CHECK_REPS = (4, 5)
FRAME_SEC = .75
FRAME_HOP_SEC = .25
CUE_SEC = 2.
REST_NAMES = ("artifact_calibration_rest_before", "artifact_calibration_rest_after")
FEATURES = ("peak_to_peak_uv", "jump_20ms_uv", "channel_difference_uv", "slow_rms_uv",
            "beta_rms_uv", "wideband_rms_uv", "lead_imbalance", "lead_correlation")


def _interval_starts(start, stop, width, hop):
    last_offset = stop - start - width + 1e-6
    if last_offset >= 0.:
        for index in range(int(last_offset // hop) + 1):
            yield start + index * hop


def _phase(payload):
    return payload.get("phase", payload.get("name"))


def _event_time(event):
    try:
        value = float(event.get("time_lsl"))
    except (TypeError, ValueError, OverflowError):
        return None
    return value if np.isfinite(value) else None


def _events(events):
    starts, ends = [], []
    rests = {name: {"phase_start": [], "phase_end": []} for name in REST_NAMES}
    cues = {action: {} for action in ACTIONS}
    for event in events:
        payload = event.get("payload")
        payload = payload if isinstance(payload, dict) else {}
        kind, stamp = event.get("kind"), _event_time(event)
        if stamp is None:
            continue
        if kind == "artifact_calibration_start":
            starts.append(stamp)
        elif kind == "artifact_calibration_complete":
            ends.append(stamp)
        elif kind == "artifact_calibration_cue":
            action, repetition = payload.get("action"), payload.get("repetition")
            if action in ACTIONS and type(repetition) is int and repetition in (*FIT_REPS, *CHECK_REPS):
                cues[action].setdefault(repetition, []).append(stamp)
        elif kind in ("phase_start", "phase_end") and _phase(payload) in REST_NAMES:
            rests[_phase(payload)][kind].append(stamp)
    return starts, ends, rests, cues


def _rms_bands(data, rate, bands):
    length = data.shape[1]
    taper = np.hanning(length)
    power = np.abs(np.fft.rfft((data - data.mean(axis=1, keepdims=True)) * taper)) ** 2
    freqs = np.fft.rfftfreq(length, 1. / rate)
    normalization = length * (taper @ taper)
    results = []
    for low, high in bands:
        energy = power[:, (freqs >= low) & (freqs < high)].sum(axis=1)
        results.append(float(np.sqrt(2 * energy.max() / normalization)))
    return results


def _features(wave, raw, rate):
    spans = np.ptp(wave, axis=1)
    jump = max(1, round(.02 * rate))
    slow, beta = _rms_bands(wave, rate, ((.5, 4.), (13., 30.)))
    high, very_high = _rms_bands(raw, rate, ((30., 45.), (66., 94.)))
    centered = wave - wave.mean(axis=1, keepdims=True)
    energy = np.linalg.norm(centered, axis=1)
    correlation = centered[0] @ centered[1] / max(1e-9, energy[0] * energy[1])
    measures = [spans.max(), np.max(np.abs(wave[:, jump:] - wave[:, :-jump])),
                np.max(np.abs(wave[0] - wave[1])), slow, beta, max(high, very_high)]
    return np.r_[np.log1p(measures), abs(np.log((spans[0] + 1.) / (spans[1] + 1.))), correlation]


class _Frames:
    def __init__(self, raw, stamps, rate, motion):
        self.raw, self.stamps, self.rate, self.motion = raw, stamps, rate, motion
        self.finite = np.flatnonzero(np.isfinite(stamps))
        self.times = stamps[self.finite]
        self.length = int(FRAME_SEC * rate)
        self.history = round(6 * rate) - round(4 * rate)
        self.samples = {}

    def sample(self, left):
        if left not in self.samples:
            if len(self.samples) >= 1024:
                self.samples.clear()
            self.samples[left] = self._sample(left)
        return self.samples[left]

    def _sample(self, left):
        first, stop = left - self.history, left + self.length
        if first < 0 or stop > len(self.stamps):
            return None
        raw, stamps = self.raw[:, first:stop], self.stamps[first:stop]
        if not np.isfinite(raw).all() or not np.isfinite(stamps).all():
            return None
        steps = np.diff(stamps)
        if np.any((steps <= 0) | (steps > 1.75 / self.rate)):
            return None
        if abs((stamps[-1] - stamps[0]) * self.rate / (len(stamps) - 1) - 1) > .05:
            return None
        wave = conditioning.filter_uv(raw, self.rate)
        if not np.isfinite(wave).all() or np.any(np.std(wave, axis=1) < 2.):
            return None
        for lead in raw:
            if any(end - start >= max(3, round(.1 * self.rate))
                   for start, end in conditioning._runs(np.diff(lead) == 0)):
                return None
        start, end = float(stamps[self.history]), float(stamps[-1] + 1. / self.rate)
        gyro_status = "unavailable"
        if self.motion is not None:
            gyro_status = self.motion.assess([start - MOTION_CLOCK_MARGIN_SEC, end + MOTION_CLOCK_MARGIN_SEC],
                                             threshold=MOTION_QUIET_CUTOFF)["status"]
        return start, end, _features(wave, raw[:, self.history:], self.rate), gyro_status

    def span(self, start, stop, *, hop=FRAME_HOP_SEC):
        for requested in _interval_starts(start, stop, FRAME_SEC, hop):
            position = np.searchsorted(self.times, requested - 1e-8)
            if position == len(self.times) or self.times[position] - requested > 1.75 / self.rate:
                yield None
            else:
                yield self.sample(int(self.finite[position]))


def _strength(values):
    return float(np.max(values[:6]))


def _distance(value, center):
    return float(np.sqrt(np.mean((value - center) ** 2)))


def _observations(frames, start, stop, center, scale):
    samples = list(frames.span(start, stop))
    return len(samples), [(a, b, (vector - center) / scale, gyro)
                          for a, b, vector, gyro in (sample for sample in samples if sample is not None)]


def _matches(value, patterns):
    return [action for action, (center, radius) in patterns.items() if _distance(value, center) <= radius]


def _rests(starts, ends, rest_events):
    if len(starts) != 1 or len(ends) != 1 or starts[0] >= ends[0]:
        return None
    blocks = []
    for name in REST_NAMES:
        start = rest_events[name]["phase_start"]
        end = rest_events[name]["phase_end"]
        if len(start) != 1 or len(end) != 1 or end[0] - start[0] < 8.:
            return None
        blocks.append((start[0], end[0]))
    before, after = blocks
    if not starts[0] <= before[0] < before[1] < after[0] < after[1] <= ends[0]:
        return None
    return before, after


def _cue_locations(cues, before, after):
    observed = []
    for action, repetitions in cues.items():
        for repetition, stamps in repetitions.items():
            observed.extend((stamp, action, repetition) for stamp in stamps)
    observed.sort()
    results = {action: {} for action in ACTIONS}
    for index, (stamp, action, repetition) in enumerate(observed):
        previous_separated = index == 0 or stamp - observed[index - 1][0] >= CUE_SEC
        next_separated = index + 1 == len(observed) or observed[index + 1][0] - stamp >= CUE_SEC
        if (len(cues[action][repetition]) == 1 and before[1] <= stamp <= after[0] - CUE_SEC
                and previous_separated and next_separated):
            results[action][repetition] = stamp
    return results


def _preferred(observations, action, threshold):
    eligible = [frame for frame in observations if action != "head" or frame[3] == "motion"]
    best = max(eligible, key=lambda frame: _strength(frame[2]), default=None)
    return best if best is not None and _strength(best[2]) >= threshold else None


def _fit_actions(frames, locations, cues, norm, threshold):
    results, patterns, checking = {}, {}, {}
    for action in ACTIONS:
        slots = locations[action]
        selected = {}
        assessable = []
        for rep, stamp in slots.items():
            count, observations = _observations(frames, stamp, stamp + CUE_SEC, *norm)
            if count and len(observations) == count:
                assessable.append(rep)
                selected[rep] = _preferred(observations, action, threshold)
        fitted = [selected[rep][2] for rep in FIT_REPS if selected.get(rep) is not None]
        checks = [selected.get(rep) for rep in CHECK_REPS]
        checking[action] = checks
        row = {"status": "unavailable", "reason": "incomplete_or_ambiguous_cues",
               "cues": sum(len(values) for values in cues[action].values()), "eligible_cues": len(slots),
               "train_count": sum(rep in assessable for rep in FIT_REPS),
               "validation_count": sum(rep in assessable for rep in CHECK_REPS),
               "available_count": len(assessable), "fit_signature_count": len(fitted), "validation_hits": 0}
        results[action] = row
        if action == "head" and frames.motion is None:
            row["reason"] = "verified_gyro_required"
        elif len(slots) == len(FIT_REPS) + len(CHECK_REPS):
            row["reason"] = "insufficient_repeatable_signature"
            if len(fitted) == len(FIT_REPS):
                center = np.median(fitted, axis=0)
                spread = max(_distance(sample, center) for sample in fitted)
                if spread <= 2.5:
                    patterns[action] = center, max(.85, spread * 1.3)
    return results, patterns, checking


def _validate_actions(results, patterns, checking, post, threshold):
    supported = {}
    for action in patterns:
        row = results[action]
        own = checking[action]
        row["validation_hits"] = sum(frame is not None and _matches(frame[2], patterns) == [action]
                                     for frame in own)
        rest_matches = sum(any(_strength(value) >= threshold and action in _matches(value, patterns)
                               and (action != "head" or gyro == "motion")
                               for _, _, value, gyro in interval) for interval in post)
        cross = sum(frame is not None and action in _matches(frame[2], patterns)
                    and (action != "head" or frame[3] == "motion")
                    for other, held in checking.items() if other != action for frame in held)
        row.update(rest_check_count=len(post), rest_check_matches=rest_matches, cross_action_matches=cross)
        if row["validation_hits"] != len(CHECK_REPS) or cross:
            row["reason"] = "held_out_inconsistent_or_overlapping"
        elif rest_matches >= 2 and rest_matches / len(post) > .15:
            row["reason"] = "frequent_during_rest_check"
        else:
            row.update(status="available", reason=None)
            supported[action] = patterns[action]
    return supported


def _review_windows(frames, cutoff, norm, threshold, patterns, supported):
    rows = []
    for start in _interval_starts(cutoff, frames.times[-1] + 1. / frames.rate, 4., 2.):
        start, stop = float(start), float(start + 4.)
        count, observations = _observations(frames, start, stop, *norm)
        row = {"start_lsl": start, "end_lsl": stop, "status": "no_signature", "actions": [],
               "reason": None, "events": [], "assessable_frames": len(observations), "planned_frames": count}
        rows.append(row)
        if len(observations) < count:
            row.update(status="unavailable", reason="incomplete_eeg_or_clock")
            continue
        uncertain = False
        for left, right, value, gyro in observations:
            if _strength(value) < threshold:
                continue
            matching = _matches(value, patterns)
            if (len(matching) != 1 or matching[0] not in supported
                    or matching[0] == "head" and gyro != "motion"):
                uncertain = True
                continue
            row["events"].append({"action": matching[0], "start_lsl": left, "end_lsl": right,
                                  "gyro_status": gyro})
        row["actions"] = list(dict.fromkeys(event["action"] for event in row["events"]))
        if row["actions"]:
            row.update(status="action_like", reason="other_unclassified_excursion" if uncertain else None)
        elif uncertain or not supported:
            row.update(status="uncertain", reason="unclassified_excursion" if uncertain else "no_supported_action_pattern")
    return rows


def build_artifact_report(raw_uv, stamps, rate, events, *, motion=None):
    """Audit cue-aligned EEG patterns and flag similarities in later 4 s windows.

    Cues identify an instruction, with a fixed 0–2 s reaction interval. They do
    not establish the source of a transient. The separate rest check can veto
    weak separation. This report has no effect on QC, EEG, or training.
    """
    starts, ends, rest_events, cues = _events(events)
    report = {"policy": POLICY, "status": "unavailable", "reason": None, "actions": {}, "windows": [],
              "fit_repetitions": list(FIT_REPS), "validation_repetitions": list(CHECK_REPS),
              "cue_response_sec": CUE_SEC, "frame_sec": FRAME_SEC, "features": list(FEATURES),
              "sources_are_inferred": True, "changes_waveform_or_eligibility": False}
    if not starts:
        return {**report, "reason": "calibration_not_recorded"}
    blocks = _rests(starts, ends, rest_events)
    if blocks is None:
        return {**report, "reason": "incomplete_calibration_or_rest"}
    raw, stamps = np.asarray(raw_uv, dtype=float), np.asarray(stamps, dtype=float)
    try:
        rate = float(rate)
    except (TypeError, ValueError, OverflowError):
        return {**report, "reason": "unsupported_eeg_input"}
    if stamps.ndim != 1 or raw.shape != (2, len(stamps)) or not np.isfinite(rate) or not 100. <= rate <= 2000.:
        return {**report, "reason": "unsupported_eeg_input"}
    frames = _Frames(raw, stamps, rate, motion)
    if not len(frames.times) or np.any(np.diff(frames.times) <= 0):
        return {**report, "reason": "unverified_eeg_clock"}
    nominal_duration = len(stamps) / rate
    audit_span = max(ends[0], float(frames.times[-1])) - starts[0]
    # Bound the number of scheduled intervals by the recorded samples. Normal gaps may
    # still mark individual intervals unavailable. With most timestamps absent, the
    # elapsed axis cannot support a session-wide diagnostic or a dense window report.
    if audit_span > 2. * nominal_duration + 4.:
        return {**report, "reason": "clock_span_exceeds_observed_samples",
                "audit_span_sec": audit_span if np.isfinite(audit_span) else None,
                "nominal_sample_duration_sec": nominal_duration}
    before, after = blocks
    # Use disjoint frames for background counts and keep filter history inside the rest block.
    initial = [frame[2] for frame in frames.span(before[0] + 2., before[1], hop=FRAME_SEC) if frame is not None]
    if len(initial) < 6:
        return {**report, "reason": "insufficient_initial_rest"}
    initial = np.stack(initial)
    center = np.median(initial, axis=0)
    scale = np.maximum(np.asarray([.18] * 6 + [.20, .20]), 1.4826 * np.median(np.abs(initial - center), axis=0))
    threshold = max(2.5, float(np.max(np.percentile((initial - center)[:, :6] / scale[:6], 95, axis=0))))
    norm = center, scale
    post = []
    for start in _interval_starts(after[0] + 2., after[1], CUE_SEC, CUE_SEC):
        count, observations = _observations(frames, start, start + CUE_SEC, *norm)
        if count and len(observations) == count:
            post.append(observations)
    if len(post) < 3:
        return {**report, "reason": "insufficient_validation_rest"}
    actions, patterns, checking = _fit_actions(frames, _cue_locations(cues, before, after), cues, norm, threshold)
    supported = _validate_actions(actions, patterns, checking, post, threshold)
    status = "available" if len(supported) == len(ACTIONS) else "partial" if supported else "unavailable"
    report.update(status=status, reason=None if supported else "no_held_out_supported_action",
                  actions=actions, before_rest_frames=len(initial), after_rest_intervals=len(post),
                  windows=_review_windows(frames, ends[0], norm, threshold, patterns, supported))
    report["window_counts"] = dict(Counter(row["status"] for row in report["windows"]))
    return report
