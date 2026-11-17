"""Causal antialiased 512-to-250 Hz sensitivity analysis on the recording time grid."""

import numpy as np

from attlapse_two_channel_data import iter_probes as raw_probes


SOURCE_RATE = 512
TARGET_RATE = 250
FILTER_LENGTH = 129
DELAY = (FILTER_LENGTH - 1) // 2
CUTOFF = 102.0


def _weights():
    # Each output uses only the last 129 samples, including the latest available.
    # The reduced 256:125 ratio gives 125 distinct fractional sample positions.
    phases = (np.arange(125) * 256 % 125) / 125
    delta = np.arange(FILTER_LENGTH)[None, :] - DELAY - phases[:, None]
    weights = (2 * CUTOFF / SOURCE_RATE * np.sinc(2 * CUTOFF / SOURCE_RATE * delta)
               * np.kaiser(FILTER_LENGTH, 8.6)[None, :])
    return weights / weights.sum(axis=1, keepdims=True)


WEIGHTS = _weights()


def resample(probe):
    if probe["rate"] != SOURCE_RATE:
        raise ValueError("Expected original 512 Hz samples.")
    start = int(probe["start_sample"])
    stop = int(probe["stop_sample"])
    first = ((start + FILTER_LENGTH - 1) * TARGET_RATE + SOURCE_RATE - 1) // SOURCE_RATE
    limit = (stop * TARGET_RATE + SOURCE_RATE - 1) // SOURCE_RATE
    samples = np.arange(first, limit, dtype=np.int64)
    base = samples * SOURCE_RATE // TARGET_RATE
    windows = np.lib.stride_tricks.sliding_window_view(probe["values_uv"], FILTER_LENGTH, axis=1)
    values = np.empty((2, len(samples)), dtype=float)
    for phase, weights in enumerate(WEIGHTS):
        offset = (phase - first) % 125
        # A view with stride 256 in the source grid avoids gathering a large tensor.
        if offset >= len(samples):
            continue
        indices = base[offset::125] - start - FILTER_LENGTH + 1
        block = windows[:, indices[0]:indices[-1] + 1:256, :]
        values[:, offset::125] = np.einsum("ctk,k->ct", block, weights)
    return {**probe, "rate": TARGET_RATE, "start_sample": first, "stop_sample": limit,
            "onset_sample": round(probe["onset_sec"] * TARGET_RATE),
            "source_onset_sample": probe["onset_sample"], "timestamps": samples / TARGET_RATE,
            "values_uv": values}


def iter_probes(root, participant=None):
    for probe in raw_probes(root, participant=participant):
        yield resample(probe)
