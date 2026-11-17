"""闭环神经反馈训练（文档 7.2.8、7.2.9、8.6）。

反馈以专注度评分驱动；训练分段进行，每段记录平均专注度、达标时间占比与波动幅度；
下一段目标按上一段表现自适应调整；训练前后各采集一次静息基线用于对比。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from ningsi import config
from ningsi.signal.baseline import Baseline, can_compare
from ningsi.signal.indicators import score_from_z


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def initial_target(baseline: Baseline | None = None) -> tuple[float, str]:
    """以专注比值的基线中位数，经过同一标准化和评分函数确定目标。"""
    settings = config.TRAINING
    base_score = score_from_z(0.0)
    stat = baseline.index_stat("focus") if baseline is not None and baseline.valid else None
    if stat is not None:
        z = baseline.z_index("focus", stat["median"])
        if z is not None:
            base_score = score_from_z(z)
    target = clamp(base_score, settings["target_min"], settings["target_max"])
    rationale = f"基线中位数对应评分 {base_score:.2f}，裁剪到 [{settings['target_min']:.2f}, {settings['target_max']:.2f}] 后取 {target:.2f}"
    if baseline is not None and not baseline.valid:
        rationale += "；注意本次基线质量不合格，目标仅作演示用途"
    return target, rationale


@dataclass
class Segment:
    index: int
    duration_sec: float
    target: float
    hold_sec: float
    samples: list = field(default_factory=list)   # [(t, focus_score)]
    excluded_windows: int = 0
    interruptions: list = field(default_factory=list)  # 无效反馈时间点，绝不以零分替代

    def stats(self) -> dict:
        events = sorted([(float(t), float(v)) for t, v in self.samples
                         if math.isfinite(t) and math.isfinite(v) and 0 <= t < self.duration_sec]
                        + [(float(t), None) for t in self.interruptions
                           if math.isfinite(t) and 0 <= t < self.duration_sec],
                        key=lambda item: item[0])
        values = [value for _, value in events if value is not None]
        valid_seconds = target_seconds = 0.0
        for index, (timestamp, value) in enumerate(events):
            # A value describes at most the next nominal update interval.
            # Invalid events stop that interval; gaps are never interpolated.
            right = min(timestamp + config.STEP_SEC, self.duration_sec,
                        events[index + 1][0] if index + 1 < len(events) else self.duration_sec)
            seconds = max(0.0, right - max(0.0, timestamp))
            if value is not None:
                valid_seconds += seconds
                if value >= self.target:
                    target_seconds += seconds
        duration = max(0.0, self.duration_sec)
        timing = {"valid_seconds": round(valid_seconds, 4),
                  "on_target_seconds": round(target_seconds, 4),
                  "valid_time_ratio": round(valid_seconds / duration, 4) if duration else 0.0,
                  "on_target_ratio": round(target_seconds / duration, 4) if duration and values else None,
                  "on_target_valid_ratio": round(target_seconds / valid_seconds, 4) if valid_seconds else None,
                  "coverage_sufficient": bool(duration and valid_seconds / duration >= config.TRAINING["min_coverage_ratio"]),
                  "timing_spec": config.TRAINING_SPEC}
        if not values:
            return {"mean": None, "volatility": None, "n": 0,
                    "achieved": False, "max_hold_sec": 0.0, "current_hold_sec": 0.0,
                    "on_target_sample_ratio": None, **timing}
        mean = sum(values) / len(values)
        if len(values) > 1:
            volatility = (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5
        else:
            volatility = 0.0
        on_target = sum(1 for v in values if v >= self.target)
        current_hold = max_hold = 0.0
        previous = None
        previous_on_target = False
        for timestamp, value in events:
            if value is not None and value >= self.target:
                delta = timestamp - previous if previous is not None else 0.0
                if previous_on_target and 0 < delta <= config.STEP_SEC * 1.5:
                    current_hold += delta
                else:
                    current_hold = 0.0
                max_hold = max(max_hold, current_hold)
            else:
                current_hold = 0.0
            previous, previous_on_target = timestamp, value is not None and value >= self.target
        return {
            "mean": round(mean, 4),
            "on_target_sample_ratio": round(on_target / len(values), 4),
            "volatility": round(volatility, 4),
            "n": len(values),
            "max_hold_sec": round(max_hold, 4),
            "current_hold_sec": round(current_hold, 4),
            "achieved": max_hold >= self.hold_sec,
            **timing,
        }

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "duration_sec": self.duration_sec,
            "target": round(self.target, 4),
            "hold_sec": self.hold_sec,
            "excluded_windows": self.excluded_windows,
            "interruptions": list(self.interruptions),
            "stats": self.stats(),
        }


@dataclass
class NeurofeedbackSession:
    participant: str
    session: str
    run: str
    mode: str = "quick"
    target: float = 0.5
    hold_sec: float = config.TRAINING["hold_sec"]
    segments: list = field(default_factory=list)
    rationale: str = ""
    baseline_before: dict = field(default_factory=dict)
    baseline_after: dict = field(default_factory=dict)
    spec: str = config.TRAINING_SPEC

    def plan(self) -> dict:
        return dict(config.TRAINING[self.mode])

    def next_target(self, stats: dict) -> float:
        """自适应：达标时间占比高且波动小 → 提高要求；占比低或波动大 → 降低要求。"""
        settings = config.TRAINING
        step = settings["target_step"]
        target, hold = self.target, self.hold_sec
        if not stats.get("coverage_sufficient"):
            return target
        ratio = stats["on_target_valid_ratio"]
        if stats["achieved"] and ratio >= settings["on_target_raise"] and stats["volatility"] <= settings["volatility_high"]:
            target = clamp(target + step, settings["target_min"], settings["target_max"])
            self.hold_sec = min(self.hold_sec + 2.0, 20.0)
        elif ratio < settings["on_target_lower"] or stats["volatility"] > settings["volatility_high"]:
            target = clamp(target - step, settings["target_min"], settings["target_max"])
            self.hold_sec = max(self.hold_sec - 2.0, 6.0)
        return target

    def add_segment(self, samples, duration_sec: float, excluded_windows: int = 0, interruptions=None) -> Segment:
        segment = Segment(
            index=len(self.segments) + 1,
            duration_sec=float(duration_sec),
            target=self.target,
            hold_sec=self.hold_sec,
            samples=[(float(t), float(v)) for t, v in samples],
            excluded_windows=int(excluded_windows),
            interruptions=list(interruptions or []),
        )
        self.segments.append(segment)
        self.target = self.next_target(segment.stats())
        return segment

    def summary(self, baseline_before: Baseline | None = None, baseline_after: Baseline | None = None) -> dict:
        stats = [segment.stats() for segment in self.segments]
        # Missing segments are retained in the record but are not zero scores.
        # Keep the existing equal weighting of segments that contain data.
        usable_stats = [s for s in stats if s["n"] > 0]
        means = [s["mean"] for s in usable_stats]
        duration = sum(segment.duration_sec for segment in self.segments)
        valid_seconds = sum(s["valid_seconds"] for s in stats)
        target_seconds = sum(s["on_target_seconds"] for s in stats)
        change = (round(stats[-1]["mean"] - stats[0]["mean"], 4)
                  if len(stats) > 1 and stats[0]["n"] > 0 and stats[-1]["n"] > 0 else None)
        comparable, reason = (True, "no_baseline")
        if baseline_before is not None and baseline_after is not None:
            comparable, reason = can_compare(baseline_before, baseline_after.srate, baseline_after.channels, baseline_after.device)
            if not baseline_after.valid:
                comparable, reason = False, "invalid_after_baseline"
        return {
            "spec": self.spec,
            "mode": self.mode,
            "segments": [segment.as_dict() for segment in self.segments],
            "target_plan": self.plan(),
            "target_rationale": self.rationale,
            "initial_target": round(self.target, 4) if not self.segments else self.segments[0].target,
            "final_target": round(self.target, 4),
            "mean_focus": round(sum(means) / len(means), 4) if means else None,
            "on_target_ratio": round(target_seconds / duration, 4) if duration and usable_stats else None,
            "valid_time_ratio": round(valid_seconds / duration, 4) if duration else 0.0,
            "on_target_valid_ratio": round(target_seconds / valid_seconds, 4) if valid_seconds else None,
            "valid_seconds": round(valid_seconds, 4),
            "on_target_seconds": round(target_seconds, 4),
            "volatility": (round(sum(s["volatility"] for s in usable_stats) / len(usable_stats), 4)
                           if usable_stats else None),
            "first_to_last_change": change,
            "baseline_comparable": comparable,
            "baseline_comparability": reason,
            "baseline_before": baseline_before.as_dict() if baseline_before else None,
            "baseline_after": baseline_after.as_dict() if baseline_after else None,
        }
