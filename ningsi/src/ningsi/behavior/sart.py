"""SART（持续注意反应任务）：可复现的平衡序列与试次级指标。

固定 180 个正式试次、20 个 No-Go（数字 3，占 11%）；8 套序列按匿名被试/会话/Run
确定性轮换，记录 sequence_set_id、随机种子与全部 No-Go 位置，便于跨被试复现。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from ningsi import config
from ningsi.behavior import metrics

SEQUENCE_SETS = 8
NOGO_DIGIT = 3
TRIALS = 180
NOGO_TRIALS = 20
PRACTICE_TRIALS = 12


@dataclass(frozen=True)
class SartSequence:
    digits: tuple
    nogo_positions: tuple
    set_id: int
    seed: int
    practice: tuple = ()

    @property
    def trials(self) -> int:
        return len(self.digits)

    def as_dict(self) -> dict:
        return {
            "trials": self.trials,
            "nogo_trials": len(self.nogo_positions),
            "nogo_positions": list(self.nogo_positions),
            "sequence_set_id": self.set_id,
            "seed": self.seed,
            "digits": list(self.digits),
        }


def _seed_from(participant: str, session: str, run: str) -> int:
    digest = hashlib.sha256(f"{participant}|{session}|{run}|sart".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _rng(seed: int):
    state = seed or 1

    def next_int(bound: int) -> int:
        nonlocal state
        state = (1103515245 * state + 12345) % (2 ** 31)
        return state % bound

    return next_int


def build_sequence(participant: str, session: str, run: str,
                   trials: int | None = None, nogo_trials: int | None = None,
                   practice_trials: int | None = None) -> SartSequence:
    """构建 SART 序列。

    试次数可覆盖（短协议用一半）：**默认值就是完整协议的 180 / 20 / 12**，
    不传参数时行为与历史版本完全一致（验收与 CLI 都不受影响）。
    No-Go 位置仍按"等间距 + 抖动"生成，短协议下等间距自动跟着缩小。
    """
    total_trials = int(trials or TRIALS)
    total_nogo = int(nogo_trials or NOGO_TRIALS)
    total_practice = int(practice_trials if practice_trials is not None else PRACTICE_TRIALS)
    if total_trials < 5 or total_nogo < 1 or total_nogo >= total_trials:
        raise ValueError(f"SART 试次数不合理：trials={total_trials} nogo={total_nogo}")
    seed = _seed_from(participant, session, run)
    set_id = seed % SEQUENCE_SETS
    next_int = _rng(seed)
    spacing = total_trials / total_nogo
    positions = []
    for index in range(total_nogo):
        base = int(round(index * spacing))
        jitter = next_int(max(1, int(spacing) - 2)) - (max(1, int(spacing) - 2) // 2)
        positions.append(min(total_trials - 1, max(0, base + jitter)))
    positions = sorted(set(positions))
    while len(positions) < total_nogo:
        candidate = next_int(total_trials)
        if candidate not in positions:
            positions.append(candidate)
    positions = tuple(sorted(positions))

    digits = []
    for index in range(total_trials):
        if index in positions:
            digits.append(NOGO_DIGIT)
        else:
            value = 1 + next_int(9)
            while value == NOGO_DIGIT:
                value = 1 + next_int(9)
            digits.append(value)
    practice = tuple(1 + next_int(9) for _ in range(max(0, total_practice)))
    return SartSequence(digits=tuple(digits), nogo_positions=positions, set_id=set_id, seed=seed, practice=practice)


@dataclass
class SartResult:
    participant: str
    session: str
    run: str
    sequence: SartSequence
    responded: tuple = ()          # 与 digits 等长：True/False
    rts: tuple = ()                # 命中试次的反应时（秒），未反应为 None
    extra: dict = field(default_factory=dict)

    def score(self) -> dict:
        go_hits, go_misses, commissions = [], 0, 0
        for index, digit in enumerate(self.sequence.digits):
            responded = self.responded[index] if index < len(self.responded) else False
            rt = self.rts[index] if index < len(self.rts) else None
            if digit == NOGO_DIGIT:
                if responded:
                    commissions += 1
            else:
                if responded and rt is not None:
                    go_hits.append(float(rt))
                else:
                    go_misses += 1
        go_total = self.sequence.trials - len(self.sequence.nogo_positions)
        nogo_total = len(self.sequence.nogo_positions)
        return {
            "task": "sart",
            "spec": config.LABEL_SPEC,
            "sequence_set_id": self.sequence.set_id,
            "seed": self.sequence.seed,
            "trials": self.sequence.trials,
            "go_trials": go_total,
            "nogo_trials": nogo_total,
            "go_accuracy": metrics.ratio(len(go_hits), go_total),
            "commission_errors": commissions,
            "commission_rate": metrics.ratio(commissions, nogo_total),
            "omission_errors": go_misses,
            "omission_rate": metrics.ratio(go_misses, go_total),
            "rt_mean": round(metrics.mean(go_hits), 4),
            "rt_sd": round(metrics.std(go_hits), 4),
            "rt_p50": round(metrics.percentile(go_hits, 0.5), 4) if go_hits else 0.0,
            "rt_variability": round(metrics.std(go_hits) / metrics.mean(go_hits), 4) if go_hits and metrics.mean(go_hits) else 0.0,
            "valid": go_total > 0 and nogo_total > 0,
            "nogo_positions": list(self.sequence.nogo_positions),
            "extra": dict(self.extra),
        }

    def as_jsonl(self) -> str:
        return json.dumps(self.score(), ensure_ascii=False)
