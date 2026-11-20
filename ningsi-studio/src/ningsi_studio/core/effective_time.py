"""Bounded effective EEG time; never changes quality or scoring eligibility.

Baseline time is the union of complete, phase-contained usable windows. Training
time starts after server feedback publication and is capped at one update
interval. Client rendering and physical display latency are not measured.
"""
from collections import Counter
from dataclasses import replace
from statistics import median
import math
import time

from ningsi import config
from ningsi.signal.baseline import build_baseline, index_values
from ningsi.signal.graded import FeedbackGate, robust_reference, union_seconds
from ningsi_studio.core.phases import EFFECTIVE_POLICY, TEN_DURATIONS
from ningsi_studio.core.ten_min import LABELS
from ningsi_studio.core.control import SessionCancelled
from ningsi_studio.domain.results import training_indicator_overview
from ningsi_studio.domain.training import StudioTraining
from ningsi_studio.domain.training_feedback import TrainingFeedbackSignal

SPEC = EFFECTIVE_POLICY["spec"]
# Wake before the previous feedback expires, leaving time for the next analysis
# and ordinary scheduler jitter. This changes no quality or support threshold.
# Accelerated fixtures cap the lead at a quarter of one protocol update.
SCHEDULING_MARGIN_WALL_SEC = .05
SCHEDULING_MARGIN_STEP_FRACTION = .25


def progress(*, target, valid, elapsed, maximum, paused=False, reason=None, eligible=True, scale=1, actual_elapsed=None):
    reached = valid + 1e-6 >= target
    return {"spec": SPEC, "target_seconds": target, "valid_seconds": round(valid, 4),
            "elapsed_seconds": round(elapsed, 4), "max_seconds": maximum,
            "extension_seconds": round(max(0.0, elapsed-target), 4),
            "quality_paused": bool(paused), "completion_reason": reason,
            "target_reached": reached, "completed": bool(reached and eligible and reason == "target_reached"),
            "time_basis": "protocol_seconds", "time_scale": scale,
            "actual_elapsed_seconds": round(elapsed*scale if actual_elapsed is None else actual_elapsed, 4)}


class DisplayClock:
    """Server feedback ledger. A score supports at most the next STEP seconds."""
    def __init__(self, target=50.0, maximum=100.0, scale=1):
        self.target, self.maximum = target, maximum
        self.scale = scale
        self.valid = 0.0
        self.at = None
        self.until = None
        self.usable = False
        self.mode = None
        self.mode_seconds = Counter()

    def settle(self, now):
        if self.at is not None:
            end = min(float(now), self.maximum, self.until)
            seconds = max(0.0, end-self.at)
            if self.mode is not None:
                self.mode_seconds[self.mode] += seconds
            if self.usable:
                self.valid += seconds
            # Keep the original expiry while consuming the interval.
            self.at = end

    def display(self, now, *, usable, mode):
        self.settle(now)
        self.at, self.usable, self.mode = float(now), bool(usable), mode
        self.until = self.at + config.STEP_SEC

    def snapshot(self, elapsed, reason=None, *, actual_elapsed=None):
        return progress(target=self.target, valid=self.valid, elapsed=elapsed,
                        maximum=self.maximum, paused=not self.usable, reason=reason, scale=self.scale,
                        actual_elapsed=actual_elapsed)


def _robust_baseline(r, steps):
    windows = [s.window for s in steps]
    base = build_baseline(windows, device=r.source.device, srate=r.source.srate)
    usable = [index_values(w) for w in windows if w.usable]
    indices = {}
    for name, stat in base.indices.items():
        vals = [v[name] for v in usable]
        center = median(vals)
        indices[name] = {**stat, "mean": center, "std": 1.4826*median(abs(v-center) for v in vals)}
    return replace(base, indices=indices, spec=base.spec + "/effective-robust-v1",
                   quality={**base.quality, "estimator": "median_MAD", "validated": False})


def _emit_progress(r, phase, state, *, segment=None):
    payload = {"phase": phase, "key": phase, "effective_time": state}
    if segment is not None:
        payload["segment"] = segment
    r.effective_time = payload
    r.bus.publish("effective_progress", payload)


def collect_baseline(r, key, run_id):
    """No bad-window filtering before build_baseline: its denominator is intact."""
    target, maximum = EFFECTIVE_POLICY["baseline_target_seconds"], EFFECTIVE_POLICY["baseline_max_seconds"]
    begin, steps, intervals = r.protocol_now(), [], []
    wall_started = time.monotonic()
    base, valid, reason, tick = _robust_baseline(r, []), 0.0, None, 0
    state = progress(target=target, valid=0, elapsed=0, maximum=maximum, paused=True, scale=r.scale)
    try:
        while True:
            r._check_cancel()
            elapsed = (r.protocol_now()-begin)/r.scale
            if elapsed >= maximum:
                reason = "max_duration"
                break
            step = r.engine.step("rest")
            elapsed = (r.protocol_now()-begin)/r.scale
            # Use the phase clock, including simulation, so no future or previous
            # phase samples can create effective baseline seconds.
            step.window = replace(step.window, t_end=elapsed)
            step.graded["t_end"] = elapsed
            if config.WINDOW_SEC <= elapsed <= maximum:
                steps.append(step)
                if step.window.usable:
                    intervals.append((elapsed-config.WINDOW_SEC, elapsed))
                valid = union_seconds(intervals)
                base = _robust_baseline(r, steps)
            r._publish_window(step, phase=key, include_signal=tick % 5 == 0)
            state = progress(target=target, valid=valid, elapsed=elapsed, maximum=maximum,
                             paused=not step.usable, eligible=base.valid, scale=r.scale,
                             actual_elapsed=time.monotonic()-wall_started)
            _emit_progress(r, key, state)
            r._progress(key, min(1.0, valid/target), effective_time=state)
            if valid + 1e-6 >= target and base.valid:
                reason = "target_reached"
                break
            tick += 1
            r._sleep_real(max(0, min(begin+maximum*r.scale,
                                    begin+tick*config.STEP_SEC*r.scale)-r.protocol_now()))
    except SessionCancelled:
        reason = "cancelled"
        raise
    finally:
        elapsed = (r.protocol_now()-begin)/r.scale
        state = progress(target=target, valid=valid, elapsed=elapsed, maximum=maximum,
                         paused=state["quality_paused"], reason=reason, eligible=base.valid, scale=r.scale,
                         actual_elapsed=time.monotonic()-wall_started)
        state.update(candidate_windows=len(steps), usable_windows=sum(s.usable for s in steps),
                     baseline_valid=base.valid, baseline_reasons=list(base.reasons),
                     minimum_baseline_seconds=config.BASELINE_PROTOCOL["min_usable_sec"],
                     minimum_valid_ratio=config.QUALITY["valid_ratio_min"])
        r.effective_time_result["phase_results"][key] = state
        r.effective_time_result["timings"][key] = round(elapsed, 3)
        _emit_progress(r, key, state)
        r._merge_run_payload(run_id, {"baseline": base.as_dict(), "effective_time": state})
    return steps, base


def training_round(r, trainer, base, index, run_id):
    # GetTickCount64 (time.monotonic on Windows / Python 3.12) can quantize
    # consecutive accelerated updates to one timestamp. QPC is also monotonic,
    # and keeps publication order / interval reconciliation at full precision.
    now = r.precise_now
    begin, gate, smooth = now(), FeedbackGate(), None
    wall_started = time.perf_counter()
    signal = TrainingFeedbackSignal(r)
    clock = DisplayClock(EFFECTIVE_POLICY["segment_target_seconds"], EFFECTIVE_POLICY["segment_max_seconds"], r.scale)
    limit = clock.maximum if base.valid else clock.target
    support_gaps = []
    recorded_expiries = set()
    next_analysis_at = None
    processing_seconds = 0.0
    analysis_durations, feedback_times, scheduling_margins = [], [], []

    def settle(elapsed):
        # The shared hold statistic tolerates timing jitter between samples.
        # This protocol has an explicit support expiry, so any later interval
        # needs an interruption even when that gap is within the old tolerance.
        # A processing/scheduling gap is not another rejected EEG window.
        expiry = min(clock.until, clock.maximum) if clock.until is not None else None
        if clock.usable and expiry is not None and elapsed > expiry + 1e-9 and expiry not in recorded_expiries:
            trainer.interruptions.append(expiry)
            support_gaps.append(expiry)
            recorded_expiries.add(expiry)
        clock.settle(elapsed)

    reason = None
    try:
        while True:
            r._check_cancel()
            elapsed = (now()-begin)/r.scale
            settle(elapsed)
            if clock.valid + 1e-6 >= clock.target:
                reason = "target_reached"
                break
            if elapsed >= limit:
                reason = "max_duration" if base.valid else "baseline_invalid"
                break
            if next_analysis_at is not None:
                wake = min(next_analysis_at, limit)
                remaining = max(0.0, clock.target-clock.valid)
                # If the currently published score can finish the target before
                # another analysis is expected to finish, settle that real tail
                # instead of doing an unnecessary analysis or pre-crediting it.
                if clock.usable and remaining <= max(0.0, clock.until-clock.at) + 1e-9:
                    goal_at = clock.at + remaining
                    if goal_at <= next_analysis_at + processing_seconds:
                        wake = min(goal_at, limit)
                r._sleep_real(max(0.0, (wake-elapsed)*r.scale))
                elapsed = (now()-begin)/r.scale
                settle(elapsed)
                if clock.valid + 1e-6 >= clock.target:
                    reason = "target_reached"
                    break
                if elapsed >= limit:
                    reason = "max_duration" if base.valid else "baseline_invalid"
                    break
            analysis_started = now()
            step = signal.read("focused")
            r._record_diagnostic(step, "training")
            heatmap = r._record_heatmap(step, "training")
            elapsed = (now()-begin)/r.scale
            processing_seconds = (now()-analysis_started)/r.scale
            analysis_durations.append(processing_seconds)
            settle(elapsed)
            if clock.valid + 1e-6 >= clock.target:
                reason = "target_reached"
                break
            if elapsed >= limit:
                reason = "max_duration" if base.valid else "baseline_invalid"
                break
            mode = gate.update(step.graded["grade"], base.valid)
            score = step.focus if mode == "eeg" and step.usable else None
            if score is not None and not math.isfinite(score):
                score = None
            smooth = None if score is None else score if smooth is None else .3*score+.7*smooth
            trainer.feed(smooth, elapsed, usable=smooth is not None)
            clock.display(elapsed, usable=smooth is not None, mode=mode)
            feedback_times.append(elapsed)
            state = clock.snapshot(elapsed, actual_elapsed=time.perf_counter()-wall_started)
            state["baseline_valid"] = base.valid
            _emit_progress(r, "training", state, segment=index+1)
            presentation = signal.presentation(step, mode, smooth, elapsed, baseline_valid=base.valid)
            payload = {"segment": index+1, "feedback_mode": mode, "mode_label": LABELS[presentation["presentation_mode"]],
                       **presentation,
                       "heatmap_cell": heatmap, "analysis_window": r.latest_analysis_window,
                       "affect_model": r.current_affect(),
                       "attention_model": step.attention_model, "grade": step.graded["grade"],
                       "score": smooth, "target": trainer.target if smooth is not None else None,
                       "usable": smooth is not None, "on_target": smooth is not None and smooth >= trainer.target,
                       "effective_time": state, "segment_stats": trainer.segment_stats(elapsed),
                       **trainer.feedback_status(elapsed)}
            r.latest_feedback = payload
            r.bus.publish("feedback", payload)
            # Analyze ahead of the *actual* prior feedback expiry. Sleeping a
            # full STEP before starting analysis creates a gap on every update.
            # Slow analysis / late wakeups still expire normally in settle();
            # neither planned deadlines nor missing intervals are credited.
            margin = (min(config.STEP_SEC*SCHEDULING_MARGIN_STEP_FRACTION,
                          SCHEDULING_MARGIN_WALL_SEC/r.scale)
                      if processing_seconds > 1e-9 else 0.0)
            scheduling_margins.append(margin)
            next_analysis_at = elapsed + max(0.0, config.STEP_SEC-processing_seconds-margin)
    except SessionCancelled:
        reason = "cancelled"
        raise
    finally:
        elapsed = (now()-begin)/r.scale
        settle(elapsed)
        state = clock.snapshot(elapsed, reason, actual_elapsed=time.perf_counter()-wall_started)
        state["baseline_valid"] = base.valid
        segment = trainer.close_segment(duration_sec=elapsed)
        segment["local_eeg_feedback"] = signal.summary()
        segment["measured_focus_updates"] = signal.measured_focus_updates
        segment["measured_focus_only_updates"] = signal.measured_focus_only_updates
        # Independently reconcile the live clock with the existing engine's
        # persisted interval accounting; never hide a disagreement by replacing
        # the clock counter with a convenient final number.
        state["timing_consistent"] = math.isclose(clock.valid, segment["stats"]["valid_seconds"], abs_tol=1e-4)
        state["target_reached"] = state["valid_seconds"] + 1e-6 >= clock.target
        state["completed"] = state["target_reached"] and reason == "target_reached" and state["timing_consistent"]
        if not state["timing_consistent"]:
            state["completion_reason"] = "timing_issue"
        segment.update(effective_time=state,
                       support_gap_count=len(support_gaps), support_gap_interruptions=support_gaps,
                       mode_seconds={k: round(v, 4) for k,v in clock.mode_seconds.items()},
                       mode_seconds_scope="strict_feedback_gate",
                       unobserved_seconds=round(max(0, elapsed-sum(clock.mode_seconds.values())), 4))
        intervals = [right-left for left, right in zip(feedback_times, feedback_times[1:])]
        segment["feedback_schedule"] = {
            "spec": "ahead-of-expiry-v1", "time_basis": "protocol_seconds", "time_scale": r.scale,
            "clock": "perf_counter",
            "nominal_step_seconds": config.STEP_SEC, "support_cap_seconds": config.STEP_SEC,
            "scheduling_margin_seconds": max(scheduling_margins, default=0.0),
            "actual_scheduling_margin_seconds": max(scheduling_margins, default=0.0)*r.scale,
            "analysis_seconds_max": max(analysis_durations, default=0.0),
            "feedback_interval_seconds_mean": sum(intervals)/len(intervals) if intervals else None,
            "feedback_interval_seconds_max": max(intervals, default=None),
            "note": "按上次处理耗时提前分析并留少量调度余量；反馈时刻取实际时钟，更新可略早于名义2秒，超时仍记缺口。",
        }
        r._persist_training_segment(segment)
        r.effective_time_result["segments"].append(segment)
        r._merge_run_payload(run_id, {"segments": r.effective_time_result["segments"], "effective_time": state})
        _emit_progress(r, "training", state, segment=index+1)
        r.bus.publish("segment", segment)
    return segment


def run(r, auto):
    r.engine.graded_enabled = True
    started = r.protocol_now()
    wall_started = time.monotonic()
    r.effective_time_result = {**EFFECTIVE_POLICY, "phase_results": {}, "timings": {}, "segments": [],
        "acquisition_complete": False, "collection_status": "in_progress", "time_scale": r.scale,
        "time_basis": "protocol_seconds", "elapsed_sec_basis": "actual_wall_seconds",
        "note": "有效时长不代表注意力或疗效；质量、基线及升级门控保持原规则。连接及报告不计阶段上限。"}
    try:
        _run(r, auto, started, wall_started)
    except SessionCancelled:
        wall_elapsed = round(time.monotonic()-wall_started, 4)
        r.effective_time_result.update(collection_status="cancelled", acquisition_complete=False,
                                       elapsed_sec=wall_elapsed,
                                       actual_elapsed_seconds=wall_elapsed,
                                       protocol_elapsed_seconds=round((r.protocol_now()-started)/r.scale, 4))
        # Partial outcome is also retained when no final report can be generated.
        r.summary = {"protocol": "effective_time", "effective_time": r.effective_time_result}
        raise


def _run(r, auto, started, wall_started):
    result = r.effective_time_result
    r.skipped_phases = ["baseline_closed", "scales", "sart", "monitor", "model"]
    r.training_skipped, r.model_skipped = False, True
    with r._phase("qc") as ctx:
        begin, steps = r.protocol_now(), []
        for tick in range(int(TEN_DURATIONS["qc"]/config.STEP_SEC)):
            r._check_cancel()
            if r.protocol_now() >= begin+30*r.scale:
                break
            step = r.engine.step("rest")
            steps.append(step)
            r._publish_window(step, phase="qc", include_signal=tick % 5 == 0)
            r._sleep_real(max(0, begin+(tick+1)*config.STEP_SEC*r.scale-r.protocol_now()))
        r._sleep_real(max(0, begin+30*r.scale-r.protocol_now()))
        usable = sum(s.usable for s in steps)
        qc = {"windows": len(steps), "usable": usable,
              "passed": bool(steps and usable/len(steps) >= config.QUALITY["valid_ratio_min"])}
        result["timings"]["qc"] = (r.protocol_now()-begin)/r.scale
        r._merge_run_payload(ctx["run_id"], qc)
        r.bus.publish("quality", qc)
    with r._phase("baseline_open") as ctx:
        before, base = collect_baseline(r, "baseline_open", ctx["run_id"])
        r.baselines, r.engine.baseline = {"baseline_open": base}, base
        r.bus.publish("baseline", {"key": "baseline_open", "baseline": base.as_dict(),
                                  "effective_time": result["phase_results"]["baseline_open"]})
    for key, practice in (("practice", True), ("pvt", False)):
        with r._phase(key) as ctx:
            begin = r.protocol_now()
            task_result = r._run_pvt(auto, practice=practice)
            r._sleep_real(max(0, begin+TEN_DURATIONS[key]*r.scale-r.protocol_now()))
            result["timings"][key] = (r.protocol_now()-begin)/r.scale
            r._merge_run_payload(ctx["run_id"], {"practice": practice, **task_result})
            if not practice:
                pvt = task_result
                r.bus.publish("behavior", {"task": "pvt", "result": pvt})
    trainer = StudioTraining(r.subject["public_id"], "01", "001", base, mode="ten_min")
    with r._phase("training") as ctx:
        training_run_id, begin = ctx["run_id"], r.precise_now()
        training_wall_started = time.perf_counter()
        r.bus.publish("training_start", {**trainer.describe(), "mode": "effective_time", "effective_time": dict(EFFECTIVE_POLICY)})
        for index in range(EFFECTIVE_POLICY["segments"]):
            segment = training_round(r, trainer, base, index, training_run_id)
            rest_payload = {"segment": index+1, "feedback_mode": "rest", "mode_label": LABELS["rest"],
                            "score": None, "usable": False, "local_eeg_feedback": None, "measured_focus_index": None,
                            "effective_time": {**segment["effective_time"], "quality_paused": False},
                            "segment_stats": {**segment["stats"], "duration_sec": segment["duration_sec"]}}
            r.latest_feedback = rest_payload
            r.bus.publish("feedback", rest_payload)
            r._sleep(EFFECTIVE_POLICY["rest_seconds"])
            r._progress("training", (index+1)/EFFECTIVE_POLICY["segments"])
        elapsed = (r.precise_now()-begin)/r.scale
        segments = result["segments"]
        completed = all(s["effective_time"]["completed"] for s in segments)
        completion_reason = ("timing_issue" if any(not s["effective_time"]["timing_consistent"] for s in segments) else
                             "target_reached" if completed else
                             "baseline_invalid" if not base.valid else "max_duration")
        training_state = progress(target=200, valid=sum(s["stats"]["valid_seconds"] for s in segments),
                                  elapsed=elapsed, maximum=440,
                                  reason=completion_reason, scale=r.scale, actual_elapsed=time.perf_counter()-training_wall_started)
        training_state["completed"] = completed
        training_state["baseline_valid"] = base.valid
        training_state["extension_seconds"] = round(max(0, elapsed-240), 4)
        result["phase_results"]["training"] = training_state
        result["timings"]["training"] = elapsed
        result["indicator_overview"] = training_indicator_overview(r.diagnostic_windows)
        r._merge_run_payload(ctx["run_id"], {"indicator_overview": result["indicator_overview"], "effective_time": training_state})
    with r._phase("baseline_after") as ctx:
        after, post = collect_baseline(r, "baseline_after", ctx["run_id"])
        r.bus.publish("baseline", {"key": "baseline_after", "baseline": post.as_dict(),
                                  "effective_time": result["phase_results"]["baseline_after"]})
    training = trainer.summary(r.engine, baseline_after=post)
    training.update(segments=segments, mode="effective_time", effective_time=training_state,
                    describe={**trainer.describe(), "mode": "effective_time"}, target_plan=dict(trainer.plan))
    r._merge_run_payload(training_run_id, {"training": training})
    before_ref, after_ref = robust_reference(s.graded for s in before), robust_reference(s.graded for s in after)
    complete = all(s["completed"] for s in result["phase_results"].values())
    wall_elapsed = round(time.monotonic()-wall_started, 4)
    result.update(baseline_before=before_ref, baseline_after=after_ref,
                  changes={ch: after_ref[ch]["median"]-v["median"] for ch,v in before_ref.items() if ch in after_ref},
                  elapsed_sec=wall_elapsed, acquisition_complete=complete,
                  actual_elapsed_seconds=wall_elapsed,
                  protocol_elapsed_seconds=round((r.protocol_now()-started)/r.scale, 4),
                  collection_status="complete" if complete else "insufficient",
                  comparison_note="同设备、同通道描述性log(β/θ)变化；质量组成可不同，不证明疗效或注意力变化")
    monitor = {"steps": [], "quality": r.engine.summarize([]), "summary": {}}
    with r._phase("assessment") as ctx:
        assessment = r._run_assessment(base, {}, {"pvt": pvt}, {}, monitor["quality"])
        r._merge_run_payload(ctx["run_id"], assessment.as_dict())
    with r._phase("report") as ctx:
        paths = r._write_report(base, {}, {}, pvt, monitor, training, assessment)
        r._merge_run_payload(ctx["run_id"], {k: str(v) for k,v in paths.items()})
        r.bus.publish("artifacts", {k: str(v) for k,v in r.artifacts.items()})
    r.summary = {"protocol": "effective_time", "training": training, "effective_time": result,
                 "quality": monitor["quality"], "assessment": assessment.as_dict(),
                 "artifacts": {k: str(v) for k,v in r.artifacts.items()}}
    r._finish("done", None, summary=r.summary)
