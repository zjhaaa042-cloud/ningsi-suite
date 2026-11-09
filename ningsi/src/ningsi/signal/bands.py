"""频带功率积分、相对功率与比值指标（文档 8.3.2）。"""

from __future__ import annotations

import numpy as np

from ningsi import config


def band_power(freqs, psd, low: float, high: float) -> float:
    """对功率谱在 [low, high) 上做梯形积分，单位 µV²。"""
    f = np.asarray(freqs, dtype=float)
    p = np.asarray(psd, dtype=float)
    mask = (f >= low) & (f < high)
    if np.count_nonzero(mask) < 2:
        return 0.0
    return float(np.trapezoid(p[mask], f[mask]))


def band_powers(freqs, psd, bands: dict | None = None) -> dict[str, float]:
    bands = bands or config.BANDS
    return {name: band_power(freqs, psd, low, high) for name, (low, high) in bands.items()}


def relative_band_powers(freqs, psd, bands: dict | None = None) -> dict[str, float]:
    """各频带功率占 0.5–45 Hz 总功率的比例；总功率为 0 时全部返回 0。"""
    powers = band_powers(freqs, psd, bands)
    total = band_power(freqs, psd, *config.TOTAL_BAND)
    if total <= 0:
        return {name: 0.0 for name in powers}
    return {name: value / total for name, value in powers.items()}


def theta_beta_ratio(freqs, psd) -> float:
    powers = band_powers(freqs, psd)
    beta = powers.get("beta", 0.0)
    return powers.get("theta", 0.0) / beta if beta > 0 else 0.0


def alpha_beta_ratio(freqs, psd) -> float:
    powers = band_powers(freqs, psd)
    beta = powers.get("beta", 0.0)
    return powers.get("alpha", 0.0) / beta if beta > 0 else 0.0
