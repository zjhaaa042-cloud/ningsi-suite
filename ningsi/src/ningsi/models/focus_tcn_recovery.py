"""Recover a bipolar TCN window when a shared input transient dominates both wires."""

from __future__ import annotations

import numpy as np

from ningsi import config
from ningsi.signal.preprocess import preprocess

from . import attlapse_shadow
from .logistic import features_from_window
from .research_features import CONTRACT, condition_window


POLICY = "focus-tcn-shared-input-recovery-v1"


def _has_raw_plateau(values, rate):
    # Reject a held contact/dropout anywhere in the context. Also reject a
    # shorter plateau specifically at a channel's recorded extrema.
    repeated = np.diff(values, axis=1) == 0
    hold = max(2, round(.1 * rate) - 1)
    if np.lib.stride_tricks.sliding_window_view(repeated, hold, axis=1).all(axis=-1).any():
        return True
    extrema = ((values == values.min(axis=1, keepdims=True))
               | (values == values.max(axis=1, keepdims=True)))
    return bool(np.any(repeated[:, :-1] & repeated[:, 1:] & extrema[:, 1:-1]))


def _has_shared_mains(values, rate):
    n = values.shape[1]
    frequency = np.fft.rfftfreq(n, 1 / rate)
    spectrum = np.fft.rfft((values - values.mean(axis=1, keepdims=True)) * np.hanning(n), axis=1)
    line = spectrum[:, (49 <= frequency) & (frequency <= 51)]
    line_energy = (np.abs(line) ** 2).sum(axis=1)
    total_energy = (np.abs(spectrum[:, (1 <= frequency) & (frequency <= 100)]) ** 2).sum(axis=1)
    if np.any(total_energy <= 0) or np.any(line_energy < .65 * total_energy):
        return False
    # A real positive cross-spectrum is required. An inverted or mostly
    # independent mains component is not a shared nuisance in the difference.
    alignment = np.vdot(line[0], line[1])
    coherence = abs(alignment) / np.sqrt(line_energy[0] * line_energy[1])
    return coherence >= .985 and abs(np.angle(alignment, deg=True)) < 8


def _has_shared_excess(values, rate):
    size = round(CONTRACT["window_sec"] * rate)
    filtered = np.stack([preprocess(channel - channel.mean(), rate)[0][-size:] for channel in values])
    limit = config.QUALITY["amp_max_uv"]
    magnitude = np.abs(filtered)
    over = magnitude > limit
    if not over.all(axis=0).any() or magnitude.max() > 4 * limit:
        return False
    suspect = over.any(axis=0)
    differential = filtered[0] - filtered[1]
    if (np.any(np.signbit(filtered[0, suspect]) != np.signbit(filtered[1, suspect]))
            or np.any(np.abs(differential[suspect]) > config.QUALITY["span_max_uv"] / 2)):
        return False
    # Use the normal conditioning branch's movement proxy on the observed
    # difference. A sharp unilateral transient is not common interference.
    step = max(1, round(.020 * rate))
    return bool(np.all(np.abs(differential[step:] - differential[:-step]) <= 75.))


def compute_window(values, timestamps, rate, factor, indices):
    """Reuse frozen bipolar features for a limited, separately annotated recovery."""
    result = attlapse_shadow.compute_window(values, timestamps, rate, factor, indices)
    original = result["physical_reasons"]
    result.update(recovery_policy=POLICY, recovered=False, original_physical_reasons=original)
    if original != ["input_amplitude"] or any(reason.startswith("bipolar_") for reason in result["strict_reasons"]):
        return result
    pair = np.asarray(values, dtype=float)[list(indices)]
    calibrated = pair * float(factor)
    size = round(CONTRACT["window_sec"] * rate)
    if _has_raw_plateau(calibrated, rate) or not _has_shared_mains(calibrated[:, -size:], rate):
        return result
    if not _has_shared_excess(calibrated, rate):
        return result
    difference = condition_window(pair[0:1] - pair[1:2], timestamps, rate, factor)
    if difference["window"].quality.ok:
        result.update(physical_reasons=[], features=features_from_window(difference["window"]), recovered=True)
    return result
