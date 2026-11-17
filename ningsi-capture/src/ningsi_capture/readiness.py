"""Operator diagnostics on raw EEG and formal probe rows."""

from dataclasses import dataclass

import numpy as np

from ningsi.models.attlapse import channel_indices


# At the current 4-second window and 2-second hop, 14 windows span 30 seconds.
RECENT_WINDOWS = 14


def line_power_ratios(samples, rate, center=50):
    """Per channel, center ±1 Hz / 1–min(100, Nyquist) Hz on sample order."""
    values = np.asarray(samples, dtype=float)
    if values.ndim != 2:
        raise ValueError("工频检查需要逐样本、逐通道的二维数据。")
    size = round(4 * rate)
    if size < 2 or len(values) < size or rate / 2 < center + 1:
        return (None,) * values.shape[1]
    frequencies = np.fft.rfftfreq(size, 1 / rate)
    total_band = (frequencies >= 1) & (frequencies <= 100)
    line_band = (frequencies >= center - 1) & (frequencies <= center + 1)
    taper = np.hanning(size)
    result = []
    for channel in values[-size:].T:
        ratio = None
        if np.isfinite(channel).all() and np.any(channel != channel[0]):
            # Ratios are scale invariant, including when the source unit is unknown.
            scaled = channel / np.max(np.abs(channel))
            power = np.abs(np.fft.rfft((scaled - scaled.mean()) * taper)) ** 2
            if size % 2 == 0:
                power[-1] *= .5  # A one-sided real spectrum has only one Nyquist bin.
            total = power[total_band].sum()
            if total > 0:
                ratio = float(power[line_band].sum() / total)
        result.append(ratio)
    return tuple(result)


@dataclass(frozen=True)
class Preflight:
    labels: tuple
    ratios: tuple
    rate: float
    passed: int
    count: int
    end_index: int
    filtered_ratios: tuple = ()
    raw_60: tuple = ()
    filtered_60: tuple = ()

    def text(self):
        if self.rate / 2 < 51:
            spectrum = "标称采样率不足以测量 49–51 Hz"
        else:
            readings = [f"{label} {value:.1%}" if value is not None else f"{label} 无法估计"
                        for label, value in zip(self.labels, self.ratios)]
            spectrum = "原始 50 Hz 占比  " + "、".join(readings)
            if any(value is not None and value > .5 for value in self.ratios):
                spectrum += " · 工频占多数，检查电极接触与线缆"
            if self.filtered_ratios and any(r is not None for r in self.filtered_ratios):
                spectrum += " · 处理后 " + "、".join(f"{r:.2%}" if r is not None else "—" for r in self.filtered_ratios)
            if any(r is not None and r > .1 for r in self.raw_60):
                spectrum += " · 60 Hz 原始→处理后 " + "、".join(
                    f"{r:.1%}→{f:.2%}" if r is not None and f is not None else "待测"
                    for r, f in zip(self.raw_60, self.filtered_60))
        return f"{spectrum}\n最近 {self.count} 窗合格 {self.passed}/{self.count}（{self.passed/self.count:.0%}）"

    def as_dict(self):
        return {"sample_end_index": self.end_index, "labels": self.labels,
                "line_power_ratios": self.ratios, "line_band_hz": [49, 51],
                "total_band_hz": [1, min(100., self.rate / 2)], "nominal_srate": self.rate,
                "recent_windows": self.count, "recent_usable": self.passed,
                "filtered_50hz_ratios": self.filtered_ratios, "raw_60hz_ratios": self.raw_60,
                "filtered_60hz_ratios": self.filtered_60}


def cohort_use(cohort, origin):
    if origin != "real":
        return "模拟数据 · 训练与测试均为 0 行"
    return {"pilot": "预试 · 训练与测试均为 0 行", "development": "开发组 · 合格探针用于训练",
            "test": "独立测试组 · 合格探针用于测试"}.get(cohort, "未分组 · 训练与测试均为 0 行")


def montage_hint(labels, reference):
    details = []
    try:
        channel_indices(labels)
    except ValueError:
        details.append("现场候选评分需按实际顺序填写 Fp1、Fp2")
    if str(reference or "").strip().lower() in ("", "unknown", "未确认", "未知"):
        details.append("参考电极待核对")
    return " · ".join(details)


def formal_probe_summary(rows):
    formal = [r for r in rows if not r["practice"]]
    labelled = [r for r in formal if r["binary_label_valid"]]
    eligible = [r for r in labelled if r["train_eligible"] or r["test_eligible"]]
    return {"formal_probes": len(formal), "on_task": sum(r["label"] == 0 for r in labelled),
            "internal_mind_wandering": sum(r["label"] == 1 for r in labelled),
            "usable_eeg": sum(r["feature_usable"] for r in formal),
            "paired": sum(r["feature_usable"] for r in labelled),
            "eligible_on_task": sum(r["label"] == 0 for r in eligible),
            "eligible_internal_mind_wandering": sum(r["label"] == 1 for r in eligible)}
