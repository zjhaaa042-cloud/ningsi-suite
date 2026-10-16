"""Welch 功率谱估计（纯 numpy 实现，参数与文档 8.3.1 一致）。

4 秒分析窗在 2 秒分段、50% 重叠下得到 3 个分段；可用长度不足一个分段时不计算 PSD，
不补齐数据、不缩短分段长度。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ningsi import config


@dataclass(frozen=True)
class Spectrum:
    freqs: np.ndarray
    psd: np.ndarray
    segments: int
    segment_sec: float
    taper: str
    nfft: int = 0
    spec: str = config.SPECTRUM_SPEC

    def meta(self) -> dict:
        return {
            "spec": self.spec,
            "segments": self.segments,
            "segment_sec": self.segment_sec,
            "taper": self.taper,
            "nfft": self.nfft,
            "delta_f_hz": round(float(self.freqs[1] - self.freqs[0]), 4) if self.freqs.size > 1 else None,
            "fmax": float(self.freqs[-1]) if self.freqs.size else 0.0,
        }


def hann(length: int) -> np.ndarray:
    if length < 2:
        raise ValueError("hann 窗长度至少为 2")
    n = np.arange(length, dtype=float)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * n / (length - 1))


def welch_psd(
    samples,
    srate: float,
    window_sec: float | None = None,
    segment_sec: float | None = None,
    overlap: float | None = None,
    fmax: float | None = None,
    detrend: str | None = None,
) -> Spectrum:
    """返回单边功率谱密度（µV²/Hz）。样本不足一个分段时抛 ValueError。"""
    x = np.asarray(samples, dtype=float).ravel()
    nperseg = int(round((segment_sec if segment_sec is not None else config.WELCH["segment_sec"]) * srate))
    ov = overlap if overlap is not None else config.WELCH["overlap"]
    fmax = fmax if fmax is not None else config.WELCH["fmax"]
    detrend = detrend or config.WELCH["detrend"]
    if nperseg < 8:
        raise ValueError("分段长度过短，无法估计功率谱")
    if x.size < nperseg:
        raise ValueError(f"可用样本 {x.size} 少于分段长度 {nperseg}，不计算 PSD")

    step = max(1, int(round(nperseg * (1.0 - ov))))
    window = hann(nperseg)
    win_power = float(np.sum(window ** 2))
    n_segments = 1 + (x.size - nperseg) // step
    # NFFT 取不小于分段点数的 2 的整数次幂（250 Hz、2 秒分段时为 512 点）
    nfft = 1
    while nfft < nperseg:
        nfft *= 2
    n_freqs = nfft // 2 + 1
    freqs = np.fft.rfftfreq(nfft, d=1.0 / srate)
    acc = np.zeros(n_freqs, dtype=float)

    for index in range(n_segments):
        start = index * step
        segment = x[start:start + nperseg].copy()
        if detrend == "mean":
            segment -= segment.mean()
        spectrum = np.fft.rfft(segment * window, n=nfft)
        power = (np.abs(spectrum) ** 2) / (srate * win_power)
        power[1:-1] *= 2.0
        acc += power

    acc /= n_segments
    keep = freqs <= fmax + 1e-9
    return Spectrum(
        freqs=freqs[keep],
        psd=acc[keep],
        segments=n_segments,
        segment_sec=nperseg / srate,
        taper=config.WELCH["taper"],
        nfft=nfft,
    )


def window_psd(prefix_uv, srate: float) -> Spectrum:
    """对单个 4 秒窗（已通过伪迹粗筛）做 Welch 估计。"""
    return welch_psd(prefix_uv, srate, window_sec=config.WINDOW_SEC)
