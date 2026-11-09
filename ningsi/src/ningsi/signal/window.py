"""单个 4 秒窗的完整特征：伪迹判定 + Welch 频谱 + 频带相对功率 + 时域特征。

一个窗只估计一次功率谱，质量判定、指标计算与报告证据都引用同一次结果。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ningsi import config
from ningsi.signal.artifacts import Verdict, check_channels
from ningsi.signal.bands import band_powers, relative_band_powers
from ningsi.signal.spectrum import Spectrum, welch_psd


def time_domain_features(data: np.ndarray) -> dict:
    """时域特征：波幅（峰峰值、RMS）与通道间相关系数（多通道时给出最小值）。"""
    out = {
        "rms_uv": round(float(np.sqrt(np.mean(data ** 2))), 4),
        "ptp_uv": round(float(np.max(data) - np.min(data)), 4),
    }
    if data.shape[0] >= 2:
        correlations = []
        for first in range(data.shape[0]):
            for second in range(first + 1, data.shape[0]):
                a, b = data[first], data[second]
                if np.std(a) < 1e-9 or np.std(b) < 1e-9:
                    continue
                correlations.append(float(np.corrcoef(a, b)[0, 1]))
        out["channel_corr_min"] = round(min(correlations), 4) if correlations else None
        out["channel_corr_max"] = round(max(correlations), 4) if correlations else None
    return out


@dataclass(frozen=True)
class WindowFeatures:
    t_end: float
    srate: float
    channels: int
    rel: dict
    powers: dict
    quality: Verdict
    spectrum: dict
    time_domain: dict = field(default_factory=dict)
    spec: str = config.INDICATOR_SPEC

    @property
    def usable(self) -> bool:
        return self.quality.ok

    def as_dict(self) -> dict:
        return {
            "t_end": self.t_end,
            "srate": self.srate,
            "channels": self.channels,
            "usable": self.usable,
            "unusable_reasons": list(self.quality.reasons),
            "relative_band_powers": {k: round(v, 6) for k, v in self.rel.items()},
            "band_powers_uv2": {k: round(v, 6) for k, v in self.powers.items()},
            "time_domain": dict(self.time_domain),
            "quality_metrics": self.quality.metrics,
            "spectrum": self.spectrum,
        }


def analyze_window(window_uv, srate: float, t_end: float = 0.0, quality: dict | None = None) -> WindowFeatures:
    """window_uv 形状为 (通道, 采样点)；多通道时对功率谱取平均。"""
    data = np.atleast_2d(np.asarray(window_uv, dtype=float))
    spectra: list[Spectrum] = []
    for channel in data:
        spectra.append(welch_psd(channel, srate))
    freqs = spectra[0].freqs
    psd = np.mean([s.psd for s in spectra], axis=0)
    verdict = check_channels(data, srate, quality=quality, spectrum=spectra[0])
    return WindowFeatures(
        t_end=float(t_end),
        srate=float(srate),
        channels=int(data.shape[0]),
        rel=relative_band_powers(freqs, psd),
        powers=band_powers(freqs, psd),
        quality=verdict,
        spectrum=spectra[0].meta(),
        time_domain=time_domain_features(data),
    )
