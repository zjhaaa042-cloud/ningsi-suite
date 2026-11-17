"""Fast, explicitly synthetic recording fixture for reproducible acceptance."""

from .acquisition import simulation_source, synthetic_samples
from .clock import Stamp
from .protocol import ProtocolConfig, TaskRunner
from .storage import SessionStore

import numpy as np


class VirtualClock:
    domain = "synthetic_virtual_clock"

    def __init__(self, seconds=1000.):
        self.seconds = seconds

    def now(self):
        return Stamp(self.seconds, self.seconds, "synthetic-virtual-time")


def create_short_demo(root):
    """Exercise the real short-protocol scheduler with an explicitly simulated clock, EEG and keys."""
    clock = VirtualClock()
    start = clock.seconds
    config = ProtocolConfig.short()
    source = simulation_source()
    store = SessionStore(root, participant="DEMO_SHORT", cohort="pilot", source=source,
                         protocol=config.as_dict(), clock=clock, notes="短版自检。脑电、按键和时钟均为模拟。")
    try:
        correction = store.correction(0.)
        runner = TaskRunner(store, lambda view: clock.now(), config, clock)
        runner.start()
        formal_answers = iter([1, 3, 1, 5, 2] * config.blocks)
        rate = source["srate"]
        next_sample = 0
        while runner.running:
            key = None
            if runner.state == "probe":
                practice = runner.phase.get("practice")
                until = clock.seconds + (7 if practice else 3)
                key = "1" if practice else str(next(formal_answers))
            elif runner.state == "sart" and not runner.trial["nogo"] and not runner.trial["responded"]:
                until, key = clock.seconds + .25, "space"
            else:
                until = runner.deadline
            # Emit continuous raw samples in bounded chunks, including the quiet baselines.
            end_sample = round((until - start) * rate)
            while next_sample < end_sample:
                last = min(next_sample + int(rate), end_sample)
                indices = np.arange(next_sample, last)
                clock.seconds = min(until, start + last / rate)
                store.append(synthetic_samples(indices, rate, source["channels"]), start + indices / rate,
                             offset=0., correction_id=correction)
                next_sample = last
            clock.seconds = until
            if key is None:
                runner.tick()
            else:
                runner.key(key)
        store.close("completed")
        return store.path
    except BaseException:
        store.close("failed", "short_simulation_failed")
        raise


def create_demo(root, participant="DEMO001", visit="01", seconds=104):
    clock = VirtualClock()
    source = simulation_source()
    config = ProtocolConfig().as_dict()
    config.update(name="synthetic-fixture-v1", calibration_sec=40, holdout_sec=4)
    store = SessionStore(root, participant=participant, visit=visit, cohort="pilot",
                         source=source, protocol=config, clock=clock,
                         notes="Explicitly synthetic fixture, not participant data or hardware validation.")
    correction = store.correction(0.)
    schedule = [
        (0, "protocol_start", {"config": config}),
        (0, "phase_start", {"phase": "baseline_calibration"}),
        (40, "phase_end", {"phase": "baseline_calibration"}),
        (40, "phase_start", {"phase": "baseline_holdout"}),
        (44, "phase_end", {"phase": "baseline_holdout"}),
        (44, "phase_start", {"phase": "sart_1"}),
    ]
    for index, (onset, choice) in enumerate(((62, 1), (80, 3), (98, 5))):
        probe = {"phase": "sart_1", "practice": False, "probe_id": f"demo-{index}", "block": 1,
                 "onset_lsl": 1000 + onset, "onset_perf": 1000 + onset}
        schedule.append((onset, "probe_onset", probe))
        schedule.append((onset + .5, "probe_response", {**probe, "choice": choice, "timeout": False, "response_rt_sec": .5}))
    schedule.extend([(101, "phase_end", {"phase": "sart_1"}), (101, "protocol_complete", {})])
    schedule.sort(key=lambda item: item[0])
    event_index = 0
    rate = source["srate"]
    try:
        for first in range(0, int(seconds * rate), 125):
            last = min(first + 125, int(seconds * rate))
            end = last / rate
            while event_index < len(schedule) and schedule[event_index][0] <= end:
                when, kind, payload = schedule[event_index]
                clock.seconds = 1000 + when
                store.event(kind, payload)
                event_index += 1
            clock.seconds = 1000 + end
            indices = np.arange(first, last)
            store.append(synthetic_samples(indices), 1000 + indices / rate,
                         offset=0., correction_id=correction)
            if first % 250 == 0:
                store.event("clock_pair", clock.now().as_dict())
        clock.seconds = 1000 + seconds
        store.close("completed" if seconds >= 101 else "aborted")
        return store.path
    except BaseException:
        store.close("failed", "synthetic_fixture_failed")
        raise
