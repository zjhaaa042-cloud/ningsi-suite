"""相对基线归一化：把绝对功率变成相对个体基线的变化量（文档 8.4）。

基线本身质量不合格、或设备/采样率/通道配置发生变化时，系统明确拒绝给出趋势结论。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ningsi import config


# 基线离散度下限：比值型指标用对数尺度，0.15 nat 约等于 16% 的比值变化；
# 同时不低于基线均值的 10%，避免基线过于平稳时把微小波动放大成极端 z 值。
STD_FLOOR = 0.15
STD_REL_FLOOR = 0.10
Z_CLIP = 3.0


@dataclass(frozen=True)
class Baseline:
    spec: str
    srate: float
    channels: int
    device: str
    bands: dict = field(default_factory=dict)     # 频带相对功率统计量
    indices: dict = field(default_factory=dict)   # 比值指标统计量
    n_windows: int = 0
    valid: bool = True
    reasons: tuple = ()

    def _z(self, table: dict, name: str, value: float) -> float | None:
        stat = table.get(name)
        if stat is None:
            return None
        mean = float(stat["mean"])
        std = max(float(stat["std"]), STD_REL_FLOOR * abs(mean), STD_FLOOR)
        z = (float(value) - mean) / std
        return max(-Z_CLIP, min(Z_CLIP, z))

    def z_band(self, name: str, value: float) -> float | None:
        if not self.valid:
            return None
        return self._z(self.bands, name, value)

    def z_index(self, name: str, value: float) -> float | None:
        if not self.valid:
            return None
        return self._z(self.indices, name, value)

    def index_stat(self, name: str) -> dict | None:
        return self.indices.get(name)

    def as_dict(self) -> dict:
        return {
            "spec": self.spec,
            "srate": self.srate,
            "channels": self.channels,
            "device": self.device,
            "n_windows": self.n_windows,
            "valid": self.valid,
            "reasons": list(self.reasons),
            "bands": self.bands,
            "indices": self.indices,
        }


def _stats(values: list[float]) -> dict:
    ordered = sorted(values)
    n = len(ordered)
    mean = sum(ordered) / n
    if n > 1:
        var = sum((v - mean) ** 2 for v in ordered) / (n - 1)
        std = var ** 0.5
    else:
        std = 0.0
    mid = n // 2
    median = ordered[mid] if n % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])
    return {"mean": round(mean, 6), "std": round(std, 6), "median": round(median, 6), "n": n}


def _index_values(window) -> dict:
    """按 config.INDICATORS 定义的频带比值，取自然对数。"""
    out = {}
    for name, (num, den) in config.INDICATORS.items():
        top = float(window.rel.get(num, 0.0))
        bottom = max(float(window.rel.get(den, 0.0)), config.INDEX_FLOOR)
        out[name] = round(__import__("math").log(max(top, config.INDEX_FLOOR) / bottom), 6)
    return out


def build_baseline(windows, device: str = "unknown", srate: float = 250.0, min_windows: int = 5) -> Baseline:
    """windows: 静息基线窗（睁眼/闭眼）序列；只用质量合格的窗。"""
    usable = [w for w in windows if w.usable]
    if not usable:
        return Baseline(config.BASELINE_SPEC, srate, 0, device, {}, {}, 0, False, ("no_usable_window",))
    names = sorted(usable[0].rel)
    bands = {name: _stats([float(w.rel[name]) for w in usable]) for name in names}
    raw_indices = [_index_values(w) for w in usable]
    indices = {name: _stats([vals[name] for vals in raw_indices]) for name in raw_indices[0]}
    reasons = () if len(usable) >= min_windows else ("insufficient_baseline_windows",)
    return Baseline(
        spec=config.BASELINE_SPEC,
        srate=float(srate),
        channels=int(usable[0].channels),
        device=device,
        bands=bands,
        indices=indices,
        n_windows=len(usable),
        valid=len(usable) >= min_windows,
        reasons=reasons,
    )


def index_values(window) -> dict:
    return _index_values(window)


def can_compare(baseline: Baseline, srate: float, channels: int, device: str) -> tuple[bool, str]:
    """跨天/跨设备趋势的可比性判定；不可比时返回原因。"""
    if baseline is None or not baseline.valid:
        return False, "baseline_invalid"
    if abs(float(srate) - float(baseline.srate)) > 1e-6:
        return False, "sample_rate_changed"
    if int(channels) != int(baseline.channels):
        return False, "channel_count_changed"
    if device and baseline.device and device != baseline.device:
        return False, "device_changed"
    return True, "comparable"
