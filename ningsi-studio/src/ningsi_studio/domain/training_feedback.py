"""Current-window spectral feedback and its separate descriptive audit trail."""

from dataclasses import replace
import math
import time

import numpy as np

from ningsi import config
from ningsi.signal.conditioning import EYE, MOVEMENT, MUSCLE, SIGNAL_FAULT
from ningsi.signal.graded import describe_fragments, union_seconds


_SPECTRAL_REASONS = {"amplitude", "eog", "emg", "channel_span", "flat_channel"}
_LOCALIZED_REASONS = {"ocular_transient": EYE, "muscle_burst": MUSCLE,
                      "motion_transient": MOVEMENT, "contact_dropout": SIGNAL_FAULT}
_LOCALIZED_BLINKS = {"blink_boundary", "blink_dense", "blink_uncalibrated", "blink_shape_mismatch",
                     "blink_channel_mismatch", "blink_residual"}


def local_eeg_feedback(step, *, srate):
    """Return screened per-lead fragments only; no score or inferred target state."""
    if step.window is None or step.raw is None:
        return None
    reasons = set(step.window.quality.reasons)
    wave = np.asarray(step.raw, dtype=float)
    if wave.ndim != 2 or not np.isfinite(wave).all():
        return None
    mask = getattr(step, "fragment_artifact_mask", None)
    eligible = None
    allowed = _SPECTRAL_REASONS
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape not in (wave.shape[-1:], wave.shape) or mask.dtype.kind not in "iu":
            return None
        # Whole-span temporal/physiological confidence comes first. A mask on
        # either lead vetoes both leads at that time; an untouched half can pass.
        global_mask = np.bitwise_or.reduce(np.atleast_2d(mask), axis=0)
        eligible = np.broadcast_to(global_mask == 0, wave.shape)
        allowed = _SPECTRAL_REASONS | {reason for reason, bit in _LOCALIZED_REASONS.items()
                                       if np.any(global_mask & bit)}
        if np.any(global_mask & EYE):
            allowed.update(_LOCALIZED_BLINKS.intersection(getattr(step, "fragment_localized_reasons", ())))
    if reasons - allowed:
        return None
    result = describe_fragments(wave, srate, eligible=eligible, include_fragment_bands=True)
    return {"kind": "relative_band_power", "channels": result["channels"]} if result["channels"] else None


class TrainingFeedbackSignal:
    """Track current-window identity and persist distinct descriptive observations."""

    def __init__(self, runtime):
        self.engine = runtime.engine
        self.srate = runtime.source.srate
        self.real = runtime.source.kind == "lsl"
        self.interval = config.STEP_SEC * runtime.scale
        self.width = round(config.WINDOW_SEC * self.srate)
        anchor = self.engine.source.assessment_timing_anchor() if self.real else None
        self.stream = self._stream(anchor) if anchor is not None else None
        self.origin = anchor["latest_sample_end"] if anchor is not None else 0
        self.last = self.origin
        self.observations = {}
        self.measured_focus_updates = 0
        self.measured_focus_only_updates = 0

    @staticmethod
    def _stream(timing):
        return timing["buffer_id"], timing["connection_generation"]

    def _fresh(self, timing):
        if timing is None or timing["received_at"] is None or not math.isfinite(timing["received_at"]):
            return False
        if time.monotonic() - timing["received_at"] >= self.interval:
            return False
        current = self._stream(timing)
        if current != self.stream:
            self.stream = current
            self.origin = self.last = timing["latest_sample_end"]
            return False
        end = timing["sample_end"]
        if end > timing["latest_sample_end"] or end < self.origin + self.width or end <= self.last:
            return False
        self.last = end
        return True

    def read(self, state):
        if not self.real:
            return self.engine.step(state)
        step = self.engine.step(state, assessment_timing=True, defer_alerts=True, fragment_masks=True)
        if not self._fresh(step.sample_timing):
            window = step.window
            if window is not None:
                quality = replace(window.quality, ok=False, reasons=tuple(dict.fromkeys((
                    *window.quality.reasons, "training_window_not_fresh"))))
                window = replace(window, quality=quality)
            step = replace(step, window=window, indicators=None, scores={}, focus_basis={},
                           graded={"grade": "C", "channels": [], "reason": "training_window_not_fresh"})
        self.engine.set_step_alerts(step)
        return step

    def _expires_at(self, step):
        now = time.monotonic()
        if not self.real:
            return now + self.interval
        timing = step.sample_timing
        received = timing.get("received_at") if timing is not None else None
        if received is None or not math.isfinite(received) or not 0 <= now - received < self.interval:
            return None
        return received + self.interval

    def capture(self, step, score, elapsed):
        if score is not None:
            return None
        expires = self._expires_at(step)
        if expires is None:
            return None
        local = local_eeg_feedback(step, srate=self.srate)
        ttl = expires - time.monotonic()
        if local is None or ttl <= 0:
            return None
        timing = step.sample_timing if self.real else None
        stream = self._stream(timing) if self.real else (None, None)
        sample_end = timing["sample_end"] if self.real else step.index * self.width
        for channel in local["channels"]:
            for (left, right), bands in zip(channel["intervals"], channel.pop("fragment_bands"), strict=True):
                start, stop = sample_end + round(left * self.srate), sample_end + round(right * self.srate)
                identity = (*stream, channel["channel"], start, stop)
                self.observations.setdefault(identity, {
                    "buffer_id": stream[0], "connection_generation": stream[1], "channel": channel["channel"],
                    "sample_start_index": start, "sample_stop_index": stop,
                    "feedback_elapsed_seconds": round(elapsed, 4), "relative": bands,
                })
        return {**local, "source": "lsl" if self.real else "simulation",
                "valid_until_unix_ms": int((time.time() + ttl) * 1000)}

    def presentation(self, step, mode, rewarded_score, elapsed, *, baseline_valid):
        """Keep the current calibrated index separate from the trainer's smoothed reward."""
        local = self.capture(step, rewarded_score, elapsed)
        measured = None
        presentation_mode = "local_eeg" if local else mode
        value = step.focus
        if (baseline_valid and step.graded.get("grade") == "A" and step.usable
                and value is not None and math.isfinite(value) and 0 <= value <= 1):
            expiry = self._expires_at(step)
            ttl = expiry - time.monotonic() if expiry is not None else 0
            if ttl > 0:
                measured = {"value": float(value), "source": "lsl" if self.real else "simulation",
                            "valid_until_unix_ms": int((time.time() + ttl) * 1000)}
                self.measured_focus_updates += 1
                if rewarded_score is None:
                    presentation_mode = "measured_focus"
                    self.measured_focus_only_updates += 1
        return {"presentation_mode": presentation_mode, "measured_focus_index": measured, "local_eeg_feedback": local}

    def summary(self):
        grouped = {}
        all_channels = {}
        for (buffer_id, generation, channel, start, stop) in self.observations:
            interval = (start / self.srate, stop / self.srate)
            grouped.setdefault(channel, {}).setdefault((buffer_id, generation), []).append(interval)
            all_channels.setdefault((buffer_id, generation), []).append(interval)
        return {"kind": "relative_band_power", "source": "lsl" if self.real else "simulation",
                "sample_rate_hz": self.srate, "observations": list(self.observations.values()),
                "observations_count": len(self.observations),
                "support_seconds": round(sum(map(union_seconds, all_channels.values())), 3),
                "support_seconds_by_channel": {
                    str(channel): round(sum(map(union_seconds, by_connection.values())), 3)
                    for channel, by_connection in grouped.items()}}
