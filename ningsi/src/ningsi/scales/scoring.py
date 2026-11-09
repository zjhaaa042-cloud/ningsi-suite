"""量表计分与分级（粗分 → 标准分 → 程度分级）。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ningsi import config
from ningsi.scales.instruments import Scale, get_scale


@dataclass(frozen=True)
class ScaleResult:
    code: str
    name: str
    version: str
    raw_score: int
    standard_score: int
    level: str
    reverse_items: tuple = ()
    answered: int = 0
    missing_items: tuple = ()
    items: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "name": self.name,
            "version": self.version,
            "raw_score": self.raw_score,
            "standard_score": self.standard_score,
            "level": self.level,
            "reverse_items": list(self.reverse_items),
            "answered": self.answered,
            "missing_items": list(self.missing_items),
            "items": self.items,
        }


def level_of(standard_score: int, boundaries: dict | None = None) -> str:
    b = boundaries or config.SCALE_BOUNDARY
    if standard_score < 50:
        return "正常范围"
    if standard_score <= b["mild_max"]:
        return "轻度"
    if standard_score <= b["moderate_max"]:
        return "中度"
    return "重度"


def score_scale(code: str, responses) -> ScaleResult:
    """responses: 长度 20 的作答序列（1–4），也可为 {题号: 值} 字典；缺答按 1 分计入并记录。"""
    scale: Scale = get_scale(code)
    if isinstance(responses, dict):
        seq = [responses.get(index + 1) for index in range(scale.size)]
    else:
        seq = list(responses)
    if len(seq) != scale.size:
        raise ValueError(f"{scale.code} 需要 {scale.size} 题作答，收到 {len(seq)}")

    raw = 0
    missing = []
    per_item = {}
    for index, (text, reverse) in enumerate(scale.items):
        value = seq[index]
        if value is None:
            missing.append(index + 1)
            value = 1
        value = int(value)
        if not 1 <= value <= 4:
            raise ValueError(f"第 {index + 1} 题作答超出 1–4：{value}")
        effective = (5 - value) if reverse else value
        per_item[index + 1] = {"text": text, "answer": value, "reverse": reverse, "effective": effective}
        raw += effective

    standard = int(math.floor(raw * scale.factor + 0.5))
    return ScaleResult(
        code=scale.code,
        name=scale.name,
        version=scale.version,
        raw_score=raw,
        standard_score=standard,
        level=level_of(standard),
        reverse_items=scale.reverse_indices(),
        answered=scale.size - len(missing),
        missing_items=tuple(missing),
        items=per_item,
    )
