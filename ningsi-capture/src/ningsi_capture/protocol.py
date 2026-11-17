"""A timer-driven research protocol, independent of Tk and EEG acquisition.

SART uses a fixed SOA. Probe screens pause trial progression, and have a
separate recovery interval. PVT is a documented response-triggered 1–4 s ISI
variant; it is not asserted equivalent to a validated clinical PVT instrument.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from math import ceil as math_ceil
import random

from .bootstrap import ensure_engine
ensure_engine()
from ningsi.models.capture_protocols import FULL_PROTOCOL, SHORT_PROTOCOL

PROBE_CHOICES = (
    "正在处理当前任务", "正在评价表现或想着任务何时结束", "正在想个人生活中的事情",
    "被环境或身体感觉吸引", "想象或白日梦等任务无关的事情", "脑子空白", "不确定",
)


@dataclass(frozen=True)
class ProtocolConfig:
    name: str = FULL_PROTOCOL
    calibration_sec: float = 120
    holdout_sec: float = 120
    sart_practice_trials: int = 12
    pvt_practice_sec: float = 30
    pvt_sec: float = 180
    blocks: int = 4
    sart_trials: int = 180
    nogo_trials: int = 20
    sart_soa: float = 2.2
    probes_per_block: int = 8
    probe_timeout: float = 8
    recovery_sec: float = 1
    rest_sec: float = 60
    artifact_still_sec: float = 60
    artifact_action_sec: float = 20
    artifact_rest_sec: float = 10
    seed: int = 20261113

    def __post_init__(self):
        if not 5 <= self.sart_practice_trials <= 10000:
            raise ValueError("SART练习至少需要5个试次。")
        if not 5 <= self.sart_trials <= 10000 or not 1 <= self.nogo_trials < self.sart_trials:
            raise ValueError("SART正式试次数或No-Go数量无效。")
        if not 1 <= self.blocks <= 20 or not 0 <= self.probes_per_block < self.sart_trials:
            raise ValueError("区块或探针数量无效。")
        optional = ("pvt_practice_sec", "pvt_sec", "artifact_still_sec", "artifact_action_sec", "artifact_rest_sec")
        for name in ("calibration_sec", "holdout_sec", "sart_soa", "probe_timeout", "recovery_sec", "rest_sec", *optional):
            value = getattr(self, name)
            if name in optional and value == 0 and self.name == SHORT_PROTOCOL:
                continue
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"时长必须为有限正数：{name}")

    def as_dict(self):
        return asdict(self)

    @classmethod
    def short(cls):
        return cls(name=SHORT_PROTOCOL, holdout_sec=45, sart_practice_trials=24,
                   blocks=3, sart_trials=115, nogo_trials=13, probes_per_block=5, rest_sec=45,
                   pvt_practice_sec=0, pvt_sec=0, artifact_still_sec=0, artifact_action_sec=0, artifact_rest_sec=0)

    @classmethod
    def demo(cls):
        return cls(name="simulation-check-v1", calibration_sec=34, holdout_sec=4,
                   sart_practice_trials=6, pvt_practice_sec=3, pvt_sec=5,
                   blocks=2, sart_trials=36, nogo_trials=4, sart_soa=.6,
                   probes_per_block=1, probe_timeout=5, recovery_sec=.4, rest_sec=2,
                   artifact_still_sec=3, artifact_action_sec=2, artifact_rest_sec=1)


def stages(config):
    items = [
        {"name": "baseline_calibration", "kind": "static", "seconds": config.calibration_sec,
         "title": "睁眼基线", "text": "自然呼吸，注视屏幕中央的十字，保持舒适坐姿。"},
        {"name": "baseline_holdout", "kind": "static", "seconds": config.holdout_sec,
         "title": "独立静息记录", "text": "继续睁眼注视十字，自然呼吸。"},
        {"name": "sart_practice", "kind": "sart", "title": "数字任务练习", "practice": True, "block": 0},
        {"name": "probe_practice", "kind": "practice_probe", "title": "探针作答练习", "practice": True, "block": 0},
        {"name": "pvt_practice", "kind": "pvt", "title": "光点反应练习", "seconds": config.pvt_practice_sec, "practice": True},
        {"name": "pvt_pre", "kind": "pvt", "title": "任务前光点反应", "seconds": config.pvt_sec},
    ]
    for block in range(1, config.blocks + 1):
        if block > 1:
            items.append({"name": f"rest_{block-1}", "kind": "static", "seconds": config.rest_sec,
                          "title": "区块间休息", "text": "可以放松双手和眼睛，请保持设备佩戴位置。"})
        items.append({"name": f"sart_{block}", "kind": "sart", "title": f"数字任务 第 {block} 区块", "block": block})
    items.append({"name": "pvt_post", "kind": "pvt", "title": "任务后光点反应", "seconds": config.pvt_sec})
    items.append({"name": "artifact_still", "kind": "static", "seconds": config.artifact_still_sec,
                  "title": "动作对照  静坐", "text": "睁眼静坐，自然呼吸，保持设备位置。"})
    for action, text in (("blink", "按舒适的节奏重复眨眼。"),
                         ("jaw", "轻微皱眉或轻轻活动下颌，不要用力。"),
                         ("head", "缓慢、小幅左右转头，保持舒适。")):
        items.append({"name": f"artifact_{action}", "kind": "static", "seconds": config.artifact_action_sec,
                      "title": "动作对照", "text": text + "不适时可立即停止。"})
        items.append({"name": f"artifact_rest_{action}", "kind": "static", "seconds": config.artifact_rest_sec,
                      "title": "动作间休息", "text": "回到自然静坐。"})
    return [stage for stage in items if stage.get("seconds") != 0]


class TaskRunner:
    def __init__(self, store, render, config=None, clock=None):
        self.store, self.render = store, render
        self.config = config or ProtocolConfig()
        self.clock = clock or store.clock
        self.plan = stages(self.config)
        seed = int(hashlib.sha256(f"{store.metadata['participant']}|{store.metadata['visit']}|{self.config.seed}".encode()).hexdigest()[:12], 16)
        self.rng = random.Random(seed)
        self.running = False
        self.finished = False
        self.phase_index = -1
        self.phase = {}
        self.state = "idle"
        self.abort_reason = None
        self.trial_index = 0
        self.trial = None
        self.probe = None
        self.last_view = {}

    def emit(self, kind, payload=None, stamp=None):
        return self.store.event(kind, {"phase": self.phase.get("name"),
                                      "practice": bool(self.phase.get("practice")),
                                      "block": self.phase.get("block", 0), **(payload or {})}, stamp)

    def show(self, title, text, footer="", choices=False):
        self.last_view = {"title": title, "text": text, "footer": footer, "choices": choices}
        return self.render(self.last_view) or self.clock.now()

    def start(self):
        if self.state != "idle":
            raise RuntimeError("该次采集的实验已经开始过，请新建采集以重新实验。")
        self.running = True
        self.emit("protocol_start", {"config": self.config.as_dict()})
        self.next_phase()

    def next_phase(self):
        if self.phase:
            self.emit("phase_end")
        self.phase_index += 1
        self.trial = None
        self.probe = None
        if self.phase_index >= len(self.plan):
            self.running, self.finished, self.state = False, True, "done"
            self.emit("protocol_complete")
            self.show("实验已完成", "请通知实验员。原始脑电仍在连续保存。")
            return
        self.phase = self.plan[self.phase_index]
        self.phase_started = self.clock.now()
        self.emit("phase_start", {"spec": self.phase}, self.phase_started)
        kind = self.phase["kind"]
        if kind == "static":
            self.state = "static"
            self.deadline = self.phase_started.perf + self.phase["seconds"]
            self.show(self.phase["title"], "+", self.phase["text"])
        elif kind == "sart":
            self.prepare_sart()
            self.next_sart_trial()
        elif kind == "practice_probe":
            self.show_probe()
        elif kind == "pvt":
            self.phase_deadline = self.phase_started.perf + self.phase["seconds"]
            self.trial_index = 0
            self.next_pvt_trial()

    def phase_remaining(self, now_perf):
        """Remaining seconds (low, high) on the phase clock, None for an untimed phase."""
        if not self.running or self.phase["kind"] == "practice_probe":
            return None
        if "seconds" in self.phase:
            remaining = max(0., self.phase_started.perf + self.phase["seconds"] - now_perf)
            return remaining, remaining

        # SART advances after each full SOA. Prompts pause the digits until a
        # response or timeout and each prompt has a separate recovery interval.
        trials_left = len(self.digits) - self.trial_index
        current_remaining = max(0., self.deadline - now_perf)
        if self.state == "sart":
            trials_left -= 1
            current_min = current_max = current_remaining
        elif self.state == "probe":
            current_min = self.config.recovery_sec
            current_max = current_min + current_remaining
        else:  # recovery after the current prompt
            current_min = current_max = current_remaining
        prompts_left = sum(position > self.trial_index for position in self.probe_positions)
        scheduled = trials_left * self.config.sart_soa + prompts_left * self.config.recovery_sec
        return scheduled + current_min, scheduled + current_max + prompts_left * self.config.probe_timeout

    def prepare_sart(self):
        from .bootstrap import ensure_engine
        ensure_engine()
        from ningsi.behavior.sart import build_sequence
        practice = self.phase.get("practice", False)
        trials = self.config.sart_practice_trials if practice else self.config.sart_trials
        nogo = max(1, trials // 9) if practice else self.config.nogo_trials
        sequence = build_sequence(self.store.metadata["participant"], self.store.metadata["visit"],
                                  self.phase["name"], trials=trials, nogo_trials=nogo, practice_trials=0)
        self.digits = sequence.digits
        self.trial_index = 0
        self.probe_positions = set()
        if not practice:
            spacing = trials / (self.config.probes_per_block + 1)
            jitter = max(0, min(3, int(spacing / 5)))
            for index in range(1, self.config.probes_per_block + 1):
                position = round(index * spacing) + self.rng.randint(-jitter, jitter)
                # The first demo probe also needs 12 seconds of same-task history.
                position = max(position, math_ceil(14 / self.config.sart_soa))
                self.probe_positions.add(min(trials - 1, position))
        self.emit("sart_sequence", {**sequence.as_dict(), "probe_after_trials": sorted(self.probe_positions),
                                    "soa_sec": self.config.sart_soa})

    def next_sart_trial(self, scheduled_perf=None):
        if self.trial_index >= len(self.digits):
            self.next_phase()
            return
        digit = self.digits[self.trial_index]
        trial_id = f"{self.phase['name']}-{self.trial_index:04d}"
        scheduled = self.clock.now()
        planned = scheduled.perf if scheduled_perf is None else scheduled_perf
        self.emit("stimulus_requested", {"task": "sart", "trial_id": trial_id, "digit": digit,
                                          "scheduled_perf": planned}, scheduled)
        onset = self.show("持续注意任务" + ("  练习" if self.phase.get("practice") else ""), str(digit),
                          "除数字 3 外，看到其他数字请按空格。数字 3 不按键。")
        self.trial = {"trial_id": trial_id, "task": "sart", "index": self.trial_index,
                      "digit": digit, "nogo": digit == 3, "onset_lsl": onset.lsl,
                      "onset_perf": onset.perf, "responded": False, "rt_sec": None,
                      "scheduled_perf": planned}
        self.emit("stimulus_onset", {**self.trial, "onset_basis": "tk_idle_estimate"}, onset)
        self.state = "sart"
        self.deadline = onset.perf + self.config.sart_soa

    def finish_sart_trial(self):
        trial = self.trial
        early = trial["responded"] and trial["rt_sec"] < .1
        self.emit("trial_result", {**trial, "valid": not early,
                                   "invalid_reason": "anticipatory_response" if early else "",
                                   "correct": trial["responded"] != trial["nogo"]})
        self.trial_index += 1
        self.trial = None
        if self.trial_index in self.probe_positions:
            self.show_probe()
        else:
            self.next_sart_trial(self.deadline)

    def show_probe(self):
        probe_id = f"{self.phase['name']}-probe-{self.trial_index:04d}"
        learning = self.phase["kind"] == "practice_probe"
        onset = self.show("刚才的思维在哪里" + ("  练习" if self.phase.get("practice") else ""),
                          "提示出现前约10秒，你的思维主要在哪里？",
                          "练习：请读完选项后按数字1–7，不计正式成绩。" if learning else "按数字 1–7 作答；请如实选择。", True)
        self.probe = {"probe_id": probe_id, "after_trial": self.trial_index,
                      "onset_lsl": onset.lsl, "onset_perf": onset.perf}
        self.emit("probe_onset", {**self.probe, "choices": list(PROBE_CHOICES),
                                  "timeout_sec": None if learning else self.config.probe_timeout,
                                  "onset_basis": "tk_idle_estimate"}, onset)
        self.state = "probe"
        self.deadline = math.inf if learning else onset.perf + self.config.probe_timeout

    def finish_probe(self, choice, stamp):
        self.emit("probe_response", {**self.probe, "choice": choice,
                                     "response_rt_sec": stamp.perf - self.probe["onset_perf"],
                                     "timeout": choice is None}, stamp)
        self.probe = None
        if self.phase["kind"] == "practice_probe":
            self.next_phase()
        else:
            self.state = "recovery"
            self.deadline = stamp.perf + self.config.recovery_sec
            self.show("请继续", "+", "准备继续数字任务。")
            self.emit("probe_recovery_start")

    def next_pvt_trial(self):
        now = self.clock.now()
        if now.perf >= self.phase_deadline:
            self.next_phase()
            return
        self.state = "pvt_wait"
        self.trial = None
        self.deadline = now.perf + self.rng.uniform(1, 4)
        self.show("光点反应任务" + ("  练习" if self.phase.get("practice") else ""), "+",
                  "等待光点出现；出现后尽快按空格，请勿提前按键。")
        self.emit("pvt_wait_start", {"index": self.trial_index, "planned_onset_perf": self.deadline})

    def onset_pvt(self):
        scheduled = self.deadline
        onset = self.show("光点反应任务", "●", "现在请按空格。")
        self.trial = {"trial_id": f"{self.phase['name']}-{self.trial_index:04d}",
                      "task": "pvt", "index": self.trial_index, "onset_lsl": onset.lsl,
                      "onset_perf": onset.perf, "scheduled_perf": scheduled,
                      "responded": False, "rt_sec": None}
        self.state = "pvt_stimulus"
        self.deadline = onset.perf + 3
        self.emit("stimulus_onset", {**self.trial, "onset_basis": "tk_idle_estimate"}, onset)

    def finish_pvt(self, reason=""):
        if self.trial is not None:
            trial = self.trial
            early = trial["responded"] and trial["rt_sec"] < .1
            reason = reason or ("anticipatory_response" if early else "")
            self.emit("trial_result", {**trial, "valid": not bool(reason), "invalid_reason": reason,
                                       "omission": not trial["responded"] and not reason,
                                       "lapse_500ms": trial["rt_sec"] is not None and trial["rt_sec"] > .5})
        self.trial = None
        self.trial_index += 1
        self.next_pvt_trial()

    def tick(self):
        if not self.running:
            return
        now = self.clock.now()
        if self.phase.get("kind") == "pvt" and now.perf >= self.phase_deadline:
            if self.trial is not None:
                self.emit("trial_result", {**self.trial, "valid": False, "invalid_reason": "phase_deadline"})
            self.next_phase()
        elif now.perf >= self.deadline:
            if self.state == "static":
                self.next_phase()
            elif self.state == "sart":
                self.finish_sart_trial()
            elif self.state == "probe":
                self.finish_probe(None, now)
            elif self.state == "recovery":
                self.next_sart_trial(self.deadline)
            elif self.state == "pvt_wait":
                self.onset_pvt()
            elif self.state == "pvt_stimulus":
                self.finish_pvt()

    def key(self, key, stamp=None, os_time_ms=None):
        if not self.running:
            return
        stamp = stamp or self.clock.now()
        # A late GUI callback cannot turn an expired trial into a valid response.
        previous = (self.state, self.trial and self.trial["trial_id"], self.probe and self.probe["probe_id"])
        self.tick()
        current = (self.state, self.trial and self.trial["trial_id"], self.probe and self.probe["probe_id"])
        self.emit("key_down", {"key": key, "state": self.state,
                               "os_event_time_ms": os_time_ms,
                               "ignored_after_transition": previous != current}, stamp)
        if previous != current:
            return
        if self.state == "probe" and key in "1234567" and len(key) == 1:
            if stamp.perf >= self.probe["onset_perf"]:
                self.finish_probe(int(key), stamp)
        elif key == "space" and self.state in ("sart", "pvt_stimulus"):
            if self.trial and not self.trial["responded"] and stamp.perf >= self.trial["onset_perf"]:
                self.trial.update(responded=True, rt_sec=stamp.perf - self.trial["onset_perf"])
                self.emit("trial_response", dict(self.trial), stamp)
                if self.state == "pvt_stimulus":
                    self.finish_pvt()
        elif key == "space" and self.state == "pvt_wait":
            self.emit("false_start", {"task": "pvt", "index": self.trial_index}, stamp)

    def abort(self, reason="operator_stop"):
        if not self.running:
            return
        if self.trial is not None:
            self.emit("trial_result", {**self.trial, "valid": False, "invalid_reason": reason})
        if self.probe is not None:
            self.emit("probe_cancelled", {**self.probe, "reason": reason})
        self.emit("phase_end", {"incomplete": True})
        self.emit("protocol_abort", {"reason": reason})
        self.abort_reason = reason
        self.running, self.state = False, "aborted"
        self.show("实验已停止", "请通知实验员。已采集数据保留。")
