"""伪迹粗筛：幅度、恒定/贴轨、通道跨度、肌电与眼电代理指标（文档 7.2.15）。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ningsi import config
from ningsi.signal.bands import relative_band_powers
from ningsi.signal.spectrum import welch_psd


@dataclass(frozen=True)
class Verdict:
    ok: bool
    reasons: tuple[str, ...] = ()
    metrics: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"ok": self.ok, "reasons": list(self.reasons), "metrics": dict(self.metrics)}


def check_channels(window_uv, srate: float, quality: dict | None = None, spectrum=None, spectra=None) -> Verdict:
    """window_uv: 形状 (通道, 采样点) 的 4 秒窗。返回是否可用与不可用原因。

    spectrum 可传入已算好的 Spectrum，避免逐窗重复估计功率谱。
    """
    q = {**config.QUALITY, **(quality or {})}
    data = np.atleast_2d(np.asarray(window_uv, dtype=float))
    if data.ndim != 2 or not data.size:
        return Verdict(False, ("no_data",), {})
    if not np.isfinite(data).all():
        return Verdict(False, ("non_finite_signal",), {})
    reasons: list[str] = []
    metrics: dict = {}

    peak = float(np.max(np.abs(data)))
    metrics["peak_uv"] = round(peak, 3)
    if peak > q["amp_max_uv"]:
        reasons.append("amplitude")

    stds = np.std(data, axis=1)
    metrics["min_std_uv"] = round(float(np.min(stds)), 3)
    if float(np.min(stds)) < q["flat_std_uv"]:
        reasons.append("flat_channel")

    span = float(np.max(np.max(data, axis=0) - np.min(data, axis=0)))
    metrics["span_uv"] = round(span, 3)
    if span > q["span_max_uv"]:
        reasons.append("channel_span")

    if spectra is None:
        spectra = [spectrum if index == 0 and spectrum is not None else welch_psd(channel, srate)
                   for index, channel in enumerate(data)]
    if len(spectra) != data.shape[0]:
        raise ValueError("频谱数量与通道数不一致")
    channel_metrics = []
    for index, item in enumerate(spectra):
        rel = relative_band_powers(item.freqs, item.psd)
        channel_metrics.append({"channel": index, "emg_rel": float(rel.get("gamma", 0.0)),
                                "eog_rel": float(rel.get("low", 0.0))})
    metrics["channels"] = channel_metrics
    # 任一通道不合格就拒绝整个窗；避免平均功率掩盖局部污染。
    metrics["emg_rel"] = max(item["emg_rel"] for item in channel_metrics)
    metrics["eog_rel"] = max(item["eog_rel"] for item in channel_metrics)
    metrics["screening_note"] = "逐通道频带代理粗筛，不能保证排除眼电或肌电污染；阈值待验证"
    if metrics["emg_rel"] > q["emg_rel_max"]:
        reasons.append("emg")
    if metrics["eog_rel"] > q["eog_rel_max"]:
        reasons.append("eog")

    return Verdict(ok=not reasons, reasons=tuple(reasons), metrics=metrics)


def window_quality_ok(window_uv, srate: float) -> bool:
    return check_channels(window_uv, srate).ok
