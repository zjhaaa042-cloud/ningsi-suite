"""Analyze an already conditioned EEG window without acquisition or persistence.

The caller owns unit conversion, filtering and the clock. This service shares
the strict quality → acquisition evidence → baseline index sequence; it does
not replace the separately versioned feature pipeline of a trained model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ningsi import config
from ningsi.signal.artifacts import Verdict
from ningsi.signal.indicators import IndicatorResult, compute_indicators
from ningsi.signal.live_artifacts import with_artifact_evidence
from ningsi.signal.window import WindowFeatures, analyze_window


@dataclass(frozen=True)
class ProcessedWindow:
    window: WindowFeatures
    indicators: IndicatorResult


def process_window(window_uv, srate: float, *, t_end: float = 0.0,
                   baseline=None, artifact_evidence: dict | None = None) -> ProcessedWindow:
    """Return strict features and relative baseline indices for C × samples in µV.

    Missing, short and non-finite inputs produce unavailable results. Quality
    evidence from acquisition must veto scoring before indices are computed;
    an unavailable result never becomes a zero attention score.
    """
    data = (np.atleast_2d(np.asarray(window_uv, dtype=float))
            if window_uv is not None else np.zeros((1, 0)))
    minimum = max(1, int(round(config.WINDOW_SEC * srate)))
    if data.ndim == 2 and (data.shape[1] < minimum or not np.isfinite(data).all()):
        samples = int(data.shape[1])
        reason = "no_data" if samples == 0 else "signal_too_short"
        if data.size and not np.isfinite(data).all():
            reason = "non_finite_signal"
        window = WindowFeatures(
            t_end=float(t_end), srate=float(srate), channels=int(data.shape[0]) or 1,
            rel={}, powers={},
            quality=Verdict(False, (reason,), {"samples": samples}), spectrum={},
        )
    else:
        window = analyze_window(data, srate, t_end=t_end)
    if artifact_evidence:
        from dataclasses import replace
        window = replace(window, quality=with_artifact_evidence(window.quality, artifact_evidence))
    return ProcessedWindow(window, compute_indicators(window, baseline))
