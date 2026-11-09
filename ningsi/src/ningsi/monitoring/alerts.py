"""三类实时预警：低专注、高负荷、信号质量不足（文档 7.2.17）。

计时以连续越界时长为准；伪迹窗（质量不合格窗）不参与计时，等于把时钟暂停，
避免"因为动作伪迹而报警"。每类预警都有独立的触发、持续与解除条件。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ningsi import config


@dataclass(frozen=True)
class AlertEvent:
    kind: str
    state: str        # triggered / active / released
    t: float
    value: float
    sustained_sec: float
    message: str

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "state": self.state,
            "t": round(self.t, 3),
            "value": round(self.value, 4),
            "sustained_sec": round(self.sustained_sec, 3),
            "message": self.message,
        }


@dataclass
class _RuleState:
    breached: float = 0.0
    clear: float = 0.0
    active: bool = False


MESSAGES = {
    "low_focus": "专注度持续偏低，建议先做 1 分钟放松再继续任务",
    "high_load": "认知负荷持续偏高，建议休息并放慢节奏",
    "poor_signal": "信号质量不足，请检查电极接触与佩戴位置",
}


class AlertEngine:
    """逐窗输入（窗特征 + 指标结果），输出预警事件序列。"""

    def __init__(self, step_sec: float | None = None, rules: dict | None = None) -> None:
        self.step_sec = step_sec or config.STEP_SEC
        self.rules = rules or config.ALERTS
        self._state = {name: _RuleState() for name in self.rules}
        self.invalid_streak = 0
        self.history: list = []

    def feed(self, window, indicators=None) -> list:
        events: list[AlertEvent] = []
        usable = bool(window is not None and window.usable)
        step = self.step_sec
        t = float(window.t_end) if window is not None else 0.0

        if not usable:
            self.invalid_streak += 1
            for name in ("low_focus", "high_load"):
                events.extend(self._update(name, t, None, breached=False, usable=False))
            events.extend(self._update(
                "poor_signal", t, float(self.invalid_streak), breached=self.invalid_streak >= config.QUALITY["consecutive_invalid_max"], usable=True,
            ))
            self.history.extend(events)
            return events

        self.invalid_streak = 0
        events.extend(self._update("poor_signal", t, 0.0, breached=False, usable=True))

        if indicators is None or not indicators.available:
            for name in ("low_focus", "high_load"):
                events.extend(self._update(name, t, None, breached=False, usable=False))
            self.history.extend(events)
            return events

        focus = indicators.score("focus")
        load = indicators.score("load")
        events.extend(self._update("low_focus", t, focus, breached=focus < self.rules["low_focus"]["threshold"], usable=True, clear=focus >= self.rules["low_focus"]["release"]))
        events.extend(self._update("high_load", t, load, breached=load >= self.rules["high_load"]["threshold"], usable=True, clear=load <= self.rules["high_load"]["release"]))
        self.history.extend(events)
        return events

    def _update(self, name: str, t: float, value, breached: bool, usable: bool, clear: bool = True) -> list:
        rule = self.rules[name]
        state = self._state[name]
        events: list[AlertEvent] = []
        if not usable:
            state.breached = 0.0
            state.clear = 0.0
            return events

        if breached:
            state.breached += self.step_sec
            state.clear = 0.0
            if not state.active and state.breached >= rule["sustain_sec"]:
                state.active = True
                events.append(AlertEvent(name, "triggered", t, float(value), state.breached, MESSAGES[name]))
        else:
            state.breached = 0.0
            if state.active:
                if clear:
                    state.clear += self.step_sec
                    if state.clear >= rule.get("release_sec", 10.0):
                        state.active = False
                        state.clear = 0.0
                        events.append(AlertEvent(name, "released", t, float(value), 0.0, MESSAGES[name] + "（已解除）"))
                else:
                    state.clear = 0.0
        if state.active and not events:
            events.append(AlertEvent(name, "active", t, float(value), state.breached, MESSAGES[name]))
        return events

    def active_kinds(self) -> list:
        return [name for name, state in self._state.items() if state.active]
