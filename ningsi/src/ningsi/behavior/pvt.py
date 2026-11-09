"""PVT-B（精神运动警觉任务）：3 分钟、刺激间隔 1–4 秒。

指标：平均反应时、中位数、慢反应（lapse，>0.5 s）次数与占比、抢答次数。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ningsi.behavior import metrics

LAPSE_SEC = 0.5
DURATION_SEC = 180.0
ISI_RANGE = (1.0, 4.0)


@dataclass
class PvtTrial:
    onset: float
    responded: bool = False
    rt: float | None = None
    false_start: bool = False


@dataclass
class PvtResult:
    trials: list = field(default_factory=list)
    duration_sec: float = DURATION_SEC

    def score(self) -> dict:
        rts = [t.rt for t in self.trials if t.responded and t.rt is not None]
        lapses = [rt for rt in rts if rt > LAPSE_SEC]
        false_starts = sum(1 for t in self.trials if t.false_start)
        return {
            "task": "pvt-b",
            "duration_sec": self.duration_sec,
            "trials": len(self.trials),
            "responded": len(rts),
            "missed": sum(1 for t in self.trials if not t.responded),
            "rt_mean": round(metrics.mean(rts), 4),
            "rt_median": round(metrics.percentile(rts, 0.5), 4),
            "rt_sd": round(metrics.std(rts), 4),
            "rt_p90": round(metrics.percentile(rts, 0.9), 4),
            "lapses": len(lapses),
            "lapse_rate": metrics.ratio(len(lapses), len(rts)),
            "false_starts": false_starts,
            "valid": len(rts) >= 5,
        }


def build_schedule(seed: int = 1, duration_sec: float = DURATION_SEC) -> list:
    """确定性生成刺激起始时刻；间隔在 1–4 秒之间。"""
    state = seed or 1
    onsets = []
    now = 2.0
    while now < duration_sec:
        state = (1103515245 * state + 12345) % (2 ** 31)
        gap = ISI_RANGE[0] + (ISI_RANGE[1] - ISI_RANGE[0]) * (state % 1000) / 1000.0
        onsets.append(round(now, 3))
        now += gap
    return onsets


def simulate_result(seed: int = 1, base_rt: float = 0.28, lapse_probability: float = 0.05) -> PvtResult:
    """用于演示与测试的确定性作答模拟；真实运行由界面采集作答时刻。"""
    state = seed or 1
    trials = []
    for onset in build_schedule(seed):
        state = (1103515245 * state + 12345) % (2 ** 31)
        draw = (state % 1000) / 1000.0
        jitter = ((state // 1000) % 1000) / 1000.0
        if draw < lapse_probability:
            trials.append(PvtTrial(onset=onset, responded=True, rt=round(base_rt + LAPSE_SEC + jitter * 0.4, 3)))
        else:
            trials.append(PvtTrial(onset=onset, responded=True, rt=round(base_rt + jitter * 0.12, 3)))
    return PvtResult(trials=trials)
