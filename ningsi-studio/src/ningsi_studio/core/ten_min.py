"""Fixed acquisition budget; descriptive salvage never supplies strict scores."""
import time
from collections import Counter
from dataclasses import replace
from statistics import median

from ningsi import config
from ningsi.signal.baseline import build_baseline, index_values
from ningsi.signal.graded import FeedbackGate, robust_reference
from ningsi_studio.domain.training import StudioTraining

LABELS = {"eeg": "探索性脑电反馈：保持稳定，不代表已验证的注意力水平",
          "quality": "质量反馈：自然呼吸、减少动作；不计注意力奖励",
          "guided": "固定引导：轻松注视前方，分心后回到目标；当前未使用脑电反馈",
          "rest": "休息10秒：放松坐姿，保持设备连接"}


def run(runtime, auto):
    r = runtime
    r.engine.graded_enabled = True
    started = time.monotonic()
    timings = {}
    r.ten_min_result = {"budget_sec": 600, "timings": timings,
                        "note": "描述性前后变化不证明训练疗效；三级规则及阈值待验证。"}
    r.skipped_phases = ["baseline_closed", "scales", "sart", "monitor", "model"]
    r.training_skipped, r.model_skipped = False, True

    def remaining(deadline):
        return max(0, deadline - time.monotonic())

    def collect(key, seconds, state="rest"):
        begin = time.monotonic()
        reference_start = begin if r.source.kind == "lsl" else r.engine.t
        deadline = begin + seconds * r.scale
        steps = []
        for i in range(int(seconds / config.STEP_SEC)):
            if time.monotonic() >= deadline:
                break
            r._check_cancel()
            step = r.engine.step(state)
            if r.source.kind == "lsl":
                step.window = replace(step.window, t_end=time.monotonic())
                step.graded["t_end"] = step.window.t_end
            # The first rolling window can include samples from the previous
            # phase. Retain it for diagnostics, never for the new baseline.
            if key == "qc" or step.window.t_end - config.WINDOW_SEC >= reference_start:
                steps.append(step)
            r._publish_window(step, phase=key, include_signal=i % 5 == 0)
            r._progress(key, (i + 1) * config.STEP_SEC / seconds)
            r._sleep_real(min(remaining(deadline), max(0, begin + (i+1)*config.STEP_SEC*r.scale-time.monotonic())))
        r._sleep_real(remaining(deadline))
        timings[key] = round((time.monotonic()-begin)/r.scale, 3)
        return steps

    def baseline(steps):
        windows = [s.window for s in steps]
        b = build_baseline(windows, device=r.source.device, srate=r.source.srate)
        # This profile alone uses robust location/scale; strict validity unchanged.
        usable = [index_values(w) for w in windows if w.usable]
        indices = {}
        for name, stat in b.indices.items():
            vals = [v[name] for v in usable]
            center = median(vals)
            indices[name] = {**stat, "mean": center, "std": 1.4826*median(abs(v-center) for v in vals)}
        return replace(b, indices=indices, spec=b.spec + "/ten-robust-v1",
                       quality={**b.quality, "estimator": "median_MAD", "validated": False})

    with r._phase("qc") as ctx:
        steps = collect("qc", 30)
        usable = sum(s.usable for s in steps)
        qc = {"windows": len(steps), "usable": usable,
              "passed": bool(steps and usable / len(steps) >= config.QUALITY["valid_ratio_min"])}
        r._merge_run_payload(ctx["run_id"], qc)
        r.bus.publish("quality", qc)
    with r._phase("baseline_open") as ctx:
        before = collect("baseline_open", 60)
        base = baseline(before)
        r.baselines = {"baseline_open": base}
        r.engine.baseline = base
        r._merge_run_payload(ctx["run_id"], {"baseline": base.as_dict()})
        r.bus.publish("baseline", {"key": "baseline_open", "baseline": base.as_dict()})
    for key, seconds, practice in (("practice", 30, True), ("pvt", 180, False)):
        with r._phase(key) as ctx:
            begin = time.monotonic()
            result = r._run_pvt(auto, practice=practice)
            r._sleep_real(max(0, begin + seconds*r.scale-time.monotonic()))
            timings[key] = round((time.monotonic()-begin)/r.scale, 3)
            r._merge_run_payload(ctx["run_id"], {"practice": practice, **result})
            if not practice:
                pvt = result
                r.bus.publish("behavior", {"task": "pvt", "result": result})

    segments = []
    gate = FeedbackGate()
    trainer = StudioTraining(r.subject["public_id"], "01", "001", base)
    trainer.plan = {"segments": 4, "segment_sec": 50}
    with r._phase("training") as ctx:
        begin = time.monotonic()
        r.bus.publish("training_start", {**trainer.describe(), "rationale": LABELS["guided"]})
        for index in range(4):
            round_start = time.monotonic()
            gate = FeedbackGate()
            counts = Counter()
            smooth = None
            for tick in range(25):
                r._check_cancel()
                if time.monotonic() >= round_start + 50*r.scale:
                    break
                step = r.engine.step("focused")
                r._record_diagnostic(step, "training")
                mode = gate.update(step.graded["grade"], base.valid)
                score = step.focus if mode == "eeg" else None
                if score is None:
                    smooth = None
                else:
                    smooth = score if smooth is None else .3*score+.7*smooth
                # Empty windows must not bridge a previous reward hold.
                sample_time = (time.monotonic() - round_start) / r.scale
                trainer.feed(smooth, sample_time, usable=smooth is not None)
                display_start = time.monotonic()
                r.bus.publish("feedback", {"segment": index+1, "feedback_mode": mode,
                    "mode_label": LABELS[mode], "grade": step.graded["grade"],
                    "score": smooth, "target": trainer.target if mode == "eeg" else None,
                    "usable": smooth is not None, "on_target": smooth is not None and smooth >= trainer.target,
                    **trainer.feedback_status()})
                r._sleep_real(max(0, round_start+(tick+1)*2*r.scale-time.monotonic()))
                counts[mode] += max(0, min(time.monotonic(), round_start+50*r.scale)-display_start)/r.scale
            segment = trainer.close_segment()
            segment["mode_seconds"] = {key: round(value, 3) for key,value in counts.items()}
            segment["unobserved_seconds"] = round(max(0, 50-sum(counts.values())), 3)
            if not segment["n"]:
                segment.update(mean_score=None, on_target_ratio=None, volatility=None)
                segment["stats"].update(mean=None, on_target_ratio=None, volatility=None)
            segments.append(segment)
            r._persist_training_segment(segment)
            r.bus.publish("segment", segment)
            r.bus.publish("feedback", {"segment": index+1, "feedback_mode": "rest", "mode_label": LABELS["rest"], "score": None, "usable": False})
            r._sleep_real(max(0, round_start+60*r.scale-time.monotonic()))
            r._progress("training", (index+1)/4)
        timings["training"] = round((time.monotonic()-begin)/r.scale, 3)
        r._merge_run_payload(ctx["run_id"], {"segments": segments})
    with r._phase("baseline_after") as ctx:
        after = collect("baseline_after", 60)
        post = baseline(after)
        r._merge_run_payload(ctx["run_id"], {"baseline": post.as_dict()})
        r.bus.publish("baseline", {"key": "baseline_after", "baseline": post.as_dict()})
    training = trainer.summary(r.engine, baseline_after=post)
    training["segments"] = segments
    training["mode"] = "ten_min"
    training["describe"] = {**trainer.describe(), "mode": "ten_min", "rest_sec": 10}
    training["target_plan"] = {"segments": 4, "segment_sec": 50, "rest_sec": 10}
    if not any(seg["n"] for seg in segments):
        for name in ("mean_focus", "on_target_ratio", "volatility", "first_to_last_change"):
            training[name] = None
    if not segments[0]["n"] or not segments[-1]["n"]:
        training["first_to_last_change"] = None
    before_ref = robust_reference(s.graded for s in before)
    after_ref = robust_reference(s.graded for s in after)
    r.ten_min_result.update({"baseline_before": before_ref, "baseline_after": after_ref,
        "changes": {ch: after_ref[ch]["median"]-v["median"] for ch, v in before_ref.items() if ch in after_ref},
        "segments": segments, "elapsed_sec": round(time.monotonic()-started, 3),
        "time_scale": r.scale, "comparison_note": "同设备、同通道描述性log(β/θ)中位数差；质量组成可不同，非疗效，非注意力得分"})
    # No monitor was collected: do not relabel training as assessment EEG.
    monitor = {"steps": [], "quality": r.engine.summarize([]), "summary": {}}
    with r._phase("assessment") as ctx:
        assessment = r._run_assessment(base, {}, {"pvt": pvt}, {}, monitor["quality"])
        r._merge_run_payload(ctx["run_id"], assessment.as_dict())
    with r._phase("report") as ctx:
        paths = r._write_report(base, {}, {}, pvt, monitor, training, assessment)
        r._merge_run_payload(ctx["run_id"], {k: str(v) for k,v in paths.items()})
        r.bus.publish("artifacts", {k: str(v) for k,v in r.artifacts.items()})
    r.summary = {"protocol": "ten_min", "training": training, "quality": monitor["quality"],
                 "assessment": assessment.as_dict(), "artifacts": {k: str(v) for k,v in r.artifacts.items()}}
    r._finish("done", None, summary=r.summary)
