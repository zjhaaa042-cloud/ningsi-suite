"""仿真脑电源：在没有真实设备时用于联调、演示与自动化测试。

按状态生成四类可复现的 4 秒窗：静息、专注、困倦、高负荷；另可生成伪迹窗用于质量门控测试。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ningsi import config

# 状态 → 各频带目标幅度（µV 峰值）：(theta 6 Hz, alpha 10 Hz, beta 18 Hz, gamma 35 Hz)
STATE_PRESETS = {
    "rest": {"theta": 6.0, "alpha": 16.0, "beta": 6.0, "gamma": 2.0},
    "eyes_closed": {"theta": 5.0, "alpha": 24.0, "beta": 4.0, "gamma": 1.5},
    "focused": {"theta": 2.0, "alpha": 14.0, "beta": 16.0, "gamma": 4.0},
    "drowsy": {"theta": 9.0, "alpha": 20.0, "beta": 4.0, "gamma": 1.5},
    "loaded": {"theta": 16.0, "alpha": 7.0, "beta": 8.0, "gamma": 2.0},
}
BAND_FREQ = {"theta": 6.0, "alpha": 10.0, "beta": 18.0, "gamma": 35.0}


@dataclass
class SyntheticEEG:
    srate: float = 250.0
    channels: int = 1
    noise_uv: float = 3.0
    seed: int = 7

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def window(self, state: str = "rest", seconds: float | None = None, artifact: bool = False) -> np.ndarray:
        """返回形状 (通道, 采样点) 的窗；artifact=True 时叠加尖峰与高幅肌电。"""
        seconds = seconds or config.WINDOW_SEC
        preset = STATE_PRESETS.get(state)
        if preset is None:
            raise ValueError(f"未知状态：{state}")
        count = int(round(seconds * self.srate))
        t = np.arange(count) / self.srate
        out = np.zeros((self.channels, count), dtype=float)
        for channel in range(self.channels):
            phase = self._rng.uniform(0, 2 * np.pi)
            signal = np.zeros(count, dtype=float)
            for band, amplitude in preset.items():
                jitter = 1.0 + 0.05 * self._rng.standard_normal()
                signal += amplitude * jitter * np.sin(2 * np.pi * BAND_FREQ[band] * t + phase + 0.1 * channel)
            signal += self._rng.normal(0.0, self.noise_uv, count)
            out[channel] = signal
        if artifact:
            out[0, count // 3] += 400.0
            out[:, count // 2:count // 2 + int(0.2 * self.srate)] += 60.0 * np.sin(
                2 * np.pi * 40.0 * t[: int(0.2 * self.srate)]
            )
        return out

    def series(self, plan, step_sec: float | None = None):
        """plan: [(state, 窗数), ...]；逐窗产出 (t_end, 窗数据)。"""
        step = step_sec or config.STEP_SEC
        t_end = 0.0
        for state, count in plan:
            for _ in range(count):
                t_end += step
                yield t_end, self.window(state)
