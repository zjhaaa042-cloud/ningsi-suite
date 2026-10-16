"""四级信号处理链（文档 8.2）：基线漂移校正 → 带通滤波 → 工频陷波 → 伪迹粗筛。

滤波采用双二阶（RBJ biquad）零相位实现：先正向再反向各滤一次，避免引入相位延迟影响
时域指标与反应时对齐。每一步参数都写入处理记录，使同一段原始数据可以被重新复算。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

MAINS_FREQS = (50.0, 60.0)


def _biquad(kind: str, f0: float, srate: float, q: float = 0.7071) -> tuple:
    w0 = 2.0 * math.pi * f0 / srate
    cos_w0, sin_w0 = math.cos(w0), math.sin(w0)
    alpha = sin_w0 / (2.0 * q)
    if kind == "lowpass":
        b = ((1 - cos_w0) / 2, 1 - cos_w0, (1 - cos_w0) / 2)
        a = (1 + alpha, -2 * cos_w0, 1 - alpha)
    elif kind == "highpass":
        b = ((1 + cos_w0) / 2, -(1 + cos_w0), (1 + cos_w0) / 2)
        a = (1 + alpha, -2 * cos_w0, 1 - alpha)
    elif kind == "notch":
        b = (1.0, -2 * cos_w0, 1.0)
        a = (1 + alpha, -2 * cos_w0, 1 - alpha)
    else:
        raise ValueError(f"未知滤波器类型：{kind}")
    return tuple(value / a[0] for value in b), tuple(value / a[0] for value in a)


def apply_biquad(samples, b, a) -> np.ndarray:
    x = np.asarray(samples, dtype=float)
    y = np.zeros_like(x)
    x1 = x2 = y1 = y2 = 0.0
    for index, value in enumerate(x):
        out = b[0] * value + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2
        x2, x1 = x1, value
        y2, y1 = y1, out
        y[index] = out
    return y


def filtfilt(samples, b, a, pad: int = 32) -> np.ndarray:
    """镜像填充后前后向各滤一次，避免多一次正向滤波引入相位偏移。"""
    x = np.asarray(samples, dtype=float)
    if x.size < 2:
        raise ValueError("零相位滤波至少需要两个样本")
    pad = min(max(1, int(pad)), x.size - 1)
    head = 2 * x[0] - x[1:pad + 1][::-1]
    tail = 2 * x[-1] - x[-pad - 1:-1][::-1]
    padded = np.concatenate([head, x, tail])
    forward = apply_biquad(padded, b, a)
    backward = apply_biquad(forward[::-1], b, a)[::-1]
    return backward[pad:-pad]


@dataclass
class ChainLog:
    stages: list = field(default_factory=list)
    spec: str = "preprocess-v2"

    def add(self, stage: str, **params) -> None:
        self.stages.append({"stage": stage, **params})

    def as_dict(self) -> dict:
        return {"spec": self.spec, "stages": self.stages}


def preprocess(samples, srate: float, drift_hz: float = 0.5, lowpass_hz: float = 45.0,
               notch_q: float = 30.0, mains=MAINS_FREQS) -> tuple:
    """返回 (滤波后信号, 处理链记录)。"""
    log = ChainLog()
    x = np.asarray(samples, dtype=float)

    b, a = _biquad("highpass", drift_hz, srate)
    x = filtfilt(x, b, a)
    log.add("drift_correction", type="highpass", fc_hz=drift_hz, zero_phase=True)

    for f0 in mains:
        if f0 < srate / 2.0:
            b, a = _biquad("notch", f0, srate, q=notch_q)
            x = filtfilt(x, b, a)
            log.add("mains_notch", f0_hz=f0, q=notch_q, zero_phase=True)

    b, a = _biquad("lowpass", lowpass_hz, srate)
    x = filtfilt(x, b, a)
    log.add("band_pass", type="lowpass", fc_hz=lowpass_hz, band_hz=(drift_hz, lowpass_hz), zero_phase=True)
    return x, log
