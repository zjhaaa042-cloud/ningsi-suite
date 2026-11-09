"""LSL 实时数据源：把 BioMultiLite / 各类 LSL 设备的流接入凝思。

依赖可选的 pylsl；未安装或未发现设备时给出明确提示，并回退到仿真源用于联调。
按厂家无关的方式解析：只要设备通过 LSL 发布 EEG 流即可接入，无需改动上层代码。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_STREAM_TYPE = "EEG"


class LslUnavailable(RuntimeError):
    """pylsl 未安装或未发现可用 EEG 流。"""


@dataclass
class LslSource:
    stream_type: str = DEFAULT_STREAM_TYPE
    stream_name: str | None = None
    buffer_sec: float = 8.0
    _inlet: object = None
    _info: object = None
    _buffer: list = field(default_factory=list)

    def connect(self, timeout: float = 5.0) -> dict:
        try:
            from pylsl import StreamInlet, resolve_byprop
        except ImportError as error:
            raise LslUnavailable("未安装 pylsl，无法接入真实设备；可先使用仿真源联调") from error
        streams = resolve_byprop("type", self.stream_type, timeout=timeout)
        if self.stream_name:
            streams = [s for s in streams if s.name() == self.stream_name] or streams
        if not streams:
            raise LslUnavailable(f"未发现 type={self.stream_type} 的 LSL 流，请确认设备已 Start")
        self._info = streams[0]
        self._inlet = StreamInlet(self._info)
        return {
            "name": self._info.name(),
            "type": self._info.type(),
            "channels": self._info.channel_count(),
            "srate": self._info.nominal_srate(),
        }

    @property
    def srate(self) -> float:
        return float(self._info.nominal_srate()) if self._info is not None else 0.0

    @property
    def channels(self) -> int:
        return int(self._info.channel_count()) if self._info is not None else 0

    def pull_window(self, seconds: float) -> np.ndarray:
        """拉取最近的 seconds 秒数据，返回 (通道, 采样点)；不足时按实际长度返回。"""
        if self._inlet is None or self._info is None:
            raise LslUnavailable("尚未连接 LSL 流")
        needed = int(round(seconds * self.srate))
        while len(self._buffer) < needed:
            chunk, _ = self._inlet.pull_chunk(timeout=0.5, max_samples=needed)
            if not chunk:
                break
            self._buffer.extend(chunk)
        tail = self._buffer[-needed:] if needed else list(self._buffer)
        if not tail:
            raise LslUnavailable("未取到样本，请检查设备是否在发布数据")
        array = np.asarray(tail, dtype=float).T
        return array
