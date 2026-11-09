"""行为指标的公共统计口径。"""

from __future__ import annotations


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def std(values) -> float:
    values = list(values)
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return (sum((v - m) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def percentile(values, fraction: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, int(round(fraction * (len(values) - 1)))))
    return values[index]


def ratio(part: float, whole: float) -> float:
    return round(part / whole, 4) if whole else 0.0
