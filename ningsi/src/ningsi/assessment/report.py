"""结构化状态评估报告（文档 7.2.10）。

报告包含状态评分、问题分析、改善建议，并逐条回填证据：每条结论都能指回所用指标、
窗口统计与参数版本，使结果可复算、可追溯。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

from ningsi import config
from ningsi.assessment.joint import JointAssessment


def _finite_number(value):
    return (float(value) if isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) else None)


def _percentage(value):
    number = _finite_number(value)
    return f"{number:.1%}" if number is not None and 0 <= number <= 1 else "—"


def _seconds(value):
    number = _finite_number(value)
    return f"{number:g}秒" if number is not None and number >= 0 else "—"


def _completion_label(value):
    return {"target_reached": "有效时间目标完成", "valid_target_reached": "有效时间目标完成",
            "wall_time_limit": "达到补采上限", "max_time_reached": "达到补采上限", "max_duration": "达到补采上限",
            "baseline_invalid": "基线未合格，保持固定引导", "baseline_unavailable": "基线不可用，保持固定引导",
            "insufficient_data": "有效数据不足", "cancelled": "已取消", "timing_issue": "计时不一致，未记为完成",
            "guided_complete": "固定引导结束，有效反馈目标未完成"}.get(value, "详见阶段采集记录")


def _effective_scale(effective, timing=None):
    for candidate in ((timing or {}).get("time_scale"), effective.get("time_scale")):
        value = _finite_number(candidate)
        if value is not None and value > 0:
            return value
    return 1.0


def _effective_actual_elapsed(timing, effective, protocol_seconds=None):
    actual = _finite_number(timing.get("actual_elapsed_seconds"))
    if actual is not None and actual >= 0:
        return actual
    elapsed = _finite_number(timing.get("elapsed_seconds") if protocol_seconds is None else protocol_seconds)
    return elapsed * _effective_scale(effective, timing) if elapsed is not None and elapsed >= 0 else None


def _effective_summary_lines(effective):
    actual = _finite_number(effective.get("actual_elapsed_seconds"))
    if actual is None:
        actual = _finite_number(effective.get("elapsed_sec"))
    protocol = _finite_number(effective.get("protocol_elapsed_seconds"))
    scale = _effective_scale(effective)
    if protocol is None and actual is not None:
        protocol = actual / scale
    return [
        f"计时单位：有效时间、训练时长和调度上限均为协议秒；时间倍率{scale:g}，实际墙钟用时另列。",
        f"实际采集用时 {_seconds(actual)}；协议采集用时 {_seconds(protocol)}／阶段采集调度上限 {_seconds(effective.get('max_acquisition_seconds'))}（协议秒），连接等待和报告生成另计。",
    ]


def _effective_phase_lines(effective):
    rows = effective.get("phase_results") or {}
    if not isinstance(rows, dict):
        return []
    lines = []
    for key, label in (("baseline_open", "睁眼基线"), ("training", "反馈训练合计"), ("baseline_after", "训练后基线")):
        row = rows.get(key)
        if not isinstance(row, dict):
            continue
        text = (f"{label}：有效时间{_seconds(row.get('valid_seconds'))}／目标{_seconds(row.get('target_seconds'))}（协议秒）；"
                f"协议用时{_seconds(row.get('elapsed_seconds'))}／调度上限{_seconds(row.get('max_seconds'))}；"
                f"实际用时{_seconds(_effective_actual_elapsed(row, effective))}；"
                f"{_completion_label(row.get('completion_reason'))}。")
        if row.get("baseline_valid") is False:
            text += "基线质量条件未通过，不能据此输出个体状态分数。"
        lines.append(text)
    return lines


def _overlapping_windows(extras) -> bool:
    """真实设备 + 快速演示模式：相邻分析窗读到几乎同一段缓冲，窗计数不是独立样本。

    仿真源每窗重新生成数据，不存在这个问题；真机走的是常驻缓冲，快速模式下
    `_sleep()` 把 2 秒步长压到 0.1 秒，于是"35 个窗"实际只覆盖约 3.5 秒的新数据
    （实测质检阶段 5 个窗的指标逐字节相同，见 `_analysis/lsl-hardware/session_tight.json`）。
    """
    extras = extras or {}
    try:
        scale = float(extras.get("time_scale") or 1.0)
    except (TypeError, ValueError):
        scale = 1.0
    return str(extras.get("source_kind") or "") == "lsl" and scale < 0.2


def replay_source_lines(source: dict) -> list[str]:
    channels = " / ".join(source["channel_labels"])
    lines = [
        f"数据来源为公开真人录波。{source['dataset']}，被试 {source['participant']}，"
        f"{channels}，{source['sample_rate_hz']:g} Hz，单位 {source['sample_unit']}。",
        f"出处为 {source['citation']}，使用许可为 {source['license']}。{source['url']}",
        source["timing"], source["scope"], source["missing"],
    ]
    lines.extend(f"{row['filename']} · {row['condition']} · 使用 {row['analyzed_sec']:g} 秒（{row['windows']} 窗）"
                 for row in source["segments"])
    return lines


@dataclass
class SessionReport:
    participant: str
    session: str
    run: str
    indicators: dict = field(default_factory=dict)
    quality: dict = field(default_factory=dict)
    scales: dict = field(default_factory=dict)
    behavior: dict = field(default_factory=dict)
    assessment: JointAssessment = None
    extras: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "spec": config.LABEL_SPEC,
            "validation_note": config.VALIDATION_NOTE,
            "threshold_status": config.VALIDATION_STATUS,
            "participant": self.participant,
            "session": self.session,
            "run": self.run,
            "versions": {
                "product": config.VERSION,
                "spectrum": config.SPECTRUM_SPEC,
                "indicator": config.INDICATOR_SPEC,
                "baseline": config.BASELINE_SPEC,
                "assessment": config.LABEL_SPEC,
                "preprocess": "preprocess-v2",
                "training": config.TRAINING_SPEC,
            },
            "indicators": self.indicators,
            "quality": self.quality,
            "scales": self.scales,
            "behavior": self.behavior,
            "assessment": self.assessment.as_dict() if self.assessment else None,
            "extras": self.extras,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False, indent=indent)

    def to_markdown(self) -> str:
        a = self.assessment
        lines = [
            f"# 凝思状态评估报告（sub-{self.participant} / ses-{self.session} / run-{self.run}）",
            "",
            f"- 指标口径：{config.INDICATOR_SPEC}；基线口径：{config.BASELINE_SPEC}；处理链：{config.SPECTRUM_SPEC}",
            f"- 软件版本：{config.VERSION}；预处理协议：preprocess-v2；训练协议：{config.TRAINING_SPEC}",
            f"- 解释边界：{config.VALIDATION_NOTE}",
            "",
        ]
        replay = self.extras.get("replay")
        if replay:
            lines.extend(["## 脑电来源与回放范围", ""])
            lines.extend(f"- {line}" for line in replay_source_lines(replay))
            lines.append("")
        # 真实设备专用：把采集调理口径写进报告表头（仿真源没有这一步，不出现这一行）
        conditioning = (self.extras or {}).get("signal_conditioning") or {}
        if conditioning:
            stages = (conditioning.get("chain") or {}).get("stages") or []
            chain_text = " → ".join(str(stage.get("stage", "")) for stage in stages) or "—"
            lines.append(
                f"- 采集调理：{conditioning.get('spec', '—')}"
                f"（{conditioning.get('dc_removal', '—')} → {chain_text}，零相位）；"
                f"上下文 {conditioning.get('context_samples', '—')} 点 / 输出 {conditioning.get('window_samples', '—')} 点"
            )
        # 协议档：短协议必须写在报告表头（行为证据减半、未跑训练与模型），否则读者会拿它
        # 与完整协议的结果直接比较。口径由会话行传入 extras.protocol_*。
        protocol_label = (self.extras or {}).get("protocol_label")
        protocol_note = (self.extras or {}).get("protocol_note")
        if protocol_label:
            skipped = (self.extras or {}).get("skipped_phases") or []
            lines.append(f"- 检测协议：{protocol_label}"
                         + (f"（本次未跑：{'、'.join(skipped)}）" if skipped else ""))
            if protocol_note:
                lines.append(f"- ⚠ 协议差异提醒：{protocol_note}")
        lines += [
            "## 一、状态评分",
        ]
        effective = (self.extras or {}).get("effective_time")
        ten = (self.extras or {}).get("ten_min")
        feedback_protocol = effective or ten
        if feedback_protocol:
            lines.extend(["", "## 有效时长反馈训练" if effective else "## 十分钟分级反馈",
                          feedback_protocol.get("note", "描述性训练记录，不代表已验证的训练效果。"),
                          feedback_protocol.get("comparison_note", ""),
                          "- 质量等级窗数：" + json.dumps((self.extras.get("signal_diagnostics") or {}).get("grades"), ensure_ascii=False),
                          "- 同通道中位数变化：" + json.dumps(feedback_protocol.get("changes", {}), ensure_ascii=False),
                          "- 分轮反馈时长：" + json.dumps([x.get("mode_seconds", {}) for x in feedback_protocol.get("segments", [])], ensure_ascii=False)])
            if effective:
                lines.extend([
                    "- 有效时间内达标率描述已有合格反馈内的表现；有效覆盖与实际用时单独保留。缺失不填零，未观察区间不累计奖励。",
                    "- 有效采集目标：" + ("已全部完成" if effective.get("acquisition_complete") is True else "尚未全部完成；请结合未完成原因与覆盖解释"),
                    "- PVT刺激、反应时与练习／正式任务时序独立于脑电质量；补采仅用于静息基线和反馈训练。",
                ])
                lines.extend("- " + item for item in _effective_summary_lines(effective))
                lines.extend("- " + item for item in _effective_phase_lines(effective))
            for segment in feedback_protocol.get("segments", []):
                stats = segment.get("stats") or {}
                if stats.get("timing_spec"):
                    lines.append(f"- 第{segment['index']}轮训练时间统计：有效时间内达标{_percentage(stats.get('on_target_valid_ratio'))}；"
                                 f"有效覆盖{_percentage(stats.get('valid_time_ratio'))}；整段达标{_percentage(stats.get('on_target_ratio'))}；"
                                 f"有效反馈{_seconds(stats.get('valid_seconds'))}／训练时长{_seconds(segment.get('duration_sec'))}（协议秒）。"
                                 + (f"实际时长{_seconds(_effective_actual_elapsed(segment.get('effective_time') or {}, effective, segment.get('duration_sec')))}。" if effective else "")
                                 + segment.get("adaptation_reason", "") if (_finite_number(stats.get("valid_seconds")) or 0) > 0 else
                                 f"- 第{segment['index']}轮无有效脑电反馈；有效覆盖0%，不计算达标率、不调整目标。")
                timing = segment.get("effective_time") or {}
                if effective and timing:
                    lines.append(f"- 第{segment['index']}轮有效时间进度：{_seconds(timing.get('valid_seconds'))}／"
                                 f"目标{_seconds(timing.get('target_seconds'))}（协议秒）；协议用时{_seconds(timing.get('elapsed_seconds'))}／"
                                 f"上限{_seconds(timing.get('max_seconds'))}；实际用时{_seconds(_effective_actual_elapsed(timing, effective))}；"
                                 f"结束原因 {_completion_label(timing.get('completion_reason'))}。")
            overview = feedback_protocol.get("indicator_overview") or {}
            if overview:
                lines.extend(["", "### 训练阶段脑电指数（描述性）", overview.get("note", "")])
                for key, label in (("focus", "专注相关指数"), ("relax", "放松度"), ("load", "认知负荷")):
                    stat = overview.get("summary", {}).get(key)
                    if stat:
                        lines.append(f"- {label}：{stat['mean']:.4f}（波动 {stat['std']:.4f}，n={stat['n']}）")
                quality = overview.get("quality", {})
                lines.append(f"- 训练可评分窗：{quality.get('windows_scored', 0)}/{quality.get('windows_total', 0)}")
                if overview.get("reason"):
                    lines.append("- " + overview["reason"])
        indicator_summary = self.indicators.get("summary", {}) if isinstance(self.indicators, dict) else {}
        labels = {"focus": "专注相关脑电指数", "relax": "放松度", "load": "认知负荷"}
        if indicator_summary:
            for key, label in labels.items():
                stat = indicator_summary.get(key)
                if stat:
                    lines.append(f"- {label}：{stat['mean']:.2f}（波动 {stat['std']:.2f}，n={stat['n']}）")
        else:
            lines.append("- 本协议未采集独立监测阶段指数；训练指数与PVT结果单独展示。" if feedback_protocol
                         else "- 无可用指标（见采集质量）")
        if isinstance(self.indicators, dict) and self.indicators.get("band_z"):
            band_text = "、".join(f"{k} {v:+.2f}" for k, v in sorted(self.indicators["band_z"].items()))
            lines.append(f"- 相对基线的频带 z 值：{band_text}")

        lines += ["", "## 二、采集质量"]
        for key, label in (("baseline_eyes_open", "睁眼"), ("baseline_eyes_closed", "闭眼")):
            baseline = (self.extras or {}).get(key) or {}
            quality = baseline.get("quality") or {}
            if quality:
                lines.append(f"- {label}基线：去重有效时长 {quality.get('usable_seconds')} 秒，"
                             f"可用窗比例 {quality.get('valid_ratio', 0):.0%}；"
                             f"前后半段对数比值均值差 {quality.get('half_mean_difference')}；"
                             f"质检原因 {baseline.get('reasons') or '无'}。稳定性尚未验证。")
        q = self.quality or {}
        if (self.extras or {}).get("protocol") in ("ten_min", "effective_time") and not q.get("windows_total"):
            lines.append("- 本协议未采集独立监测窗口；启动质检、基线与训练质量见分级窗口统计。")
        else:
            lines.append(
                f"- 4 秒窗 {q.get('windows_usable', 0)}/{q.get('windows_total', 0)} 可用"
                f"（可用窗比例 {q.get('valid_ratio', 0):.0%}）"
            )
        if q.get("unusable_reasons"):
            detail = "、".join(f"{k}×{v}" for k, v in sorted(q["unusable_reasons"].items()))
            lines.append(f"- 排除原因：{detail}（含伪迹窗不计入指标）")
        if _overlapping_windows(self.extras):
            lines.append(
                "- 采集口径提醒：真实设备 + 快速演示模式（time_scale < 0.2）下相邻 4 秒窗读到的是"
                "几乎同一段缓冲数据，上面的可用窗计数是**高重叠样本**，只能用于链路自检；"
                "真实节奏采集请用 time_scale = 1.0（此时量表与按键任务由本人操作）。"
            )

        lines += ["", "## 三、量表结果"]
        if self.scales:
            for code, result in sorted(self.scales.items()):
                lines.append(
                    f"- {result['name']}（{code}）：粗分 {result['raw_score']}，标准分 {result['standard_score']}，"
                    f"{result['level']}；反向题 {result['reverse_items']}，版本 {result['version']}"
                )
        else:
            lines.append("- 未采集")

        lines += ["", "## 四、行为任务"]
        if self.behavior:
            for name, result in sorted(self.behavior.items()):
                if not result:
                    lines.append(f"- {name.upper()}：未采集")
                    continue
                if name == "sart":
                    lines.append(
                        f"- SART：{result.get('trials', '—')} 试次（No-Go {result.get('nogo_trials', '—')}），"
                        f"正确率 {result.get('go_accuracy', 0):.0%}，虚报率 {result.get('commission_rate', 0):.1%}，"
                        f"漏报率 {result.get('omission_rate', 0):.1%}，反应时 {result.get('rt_mean', 0):.3f}"
                        f"±{result.get('rt_sd', 0):.3f}s，序列集 {result.get('sequence_set_id', '—')}"
                    )
                elif name == "pvt":
                    if result.get('scoring_spec'):
                        lines.append(f"- PVT 计分：{result['scoring_spec']}；主慢反应 >500 ms；"
                                     f"≥355 ms 对照慢反应率 {result.get('lapse_rate_355ms', 0):.1%}；"
                                     f"漏答率 {result.get('omission_rate', 0):.1%}；"
                                     f"证据{'可用' if result.get('valid', False) else '不可用'}；判定阈值待验证。"
                                     f"{result.get('protocol_note', '')}")
                    lines.append(
                        f"- PVT-B：{result.get('trials', '—')} 试次，中位反应时 {result.get('rt_median', 0):.3f}s，"
                        f"慢反应率 {result.get('lapse_rate', 0):.1%}，抢答 {result.get('false_starts', 0)} 次"
                    )
                else:
                    lines.append(f"- {name}：{json.dumps(result, ensure_ascii=False)}")
        else:
            lines.append("- 未采集")

        lines += ["", "## 五、联合评估结论", f"- 结论：{a.conclusion if a else '未评估'}"]
        if a:
            for dimension, state in sorted(a.dimension_states.items()):
                label = {"attention": "专注维度", "stress": "压力维度"}.get(dimension, dimension)
                lines.append(f"- {label}：{state['state']}（依据 {', '.join(state['evidence_codes']) or '无'}）")
            lines.append(f"- 一致性：脑电 vs 量表 {a.consistency['eeg_vs_scale']}；脑电 vs 行为 {a.consistency['eeg_vs_behavior']}")
            lines += ["", "## 六、改善建议"]
            for item in a.advice:
                lines.append(f"- {item}")
            lines += ["", "## 七、证据回填"]
            for item in a.evidence:
                state = "可用" if item.available else "不可用"
                lines.append(f"- [{item.code}]（{item.source}/{state}）{item.summary}；引用 {json.dumps(item.ref, ensure_ascii=False)}")
        observations = (self.extras or {}).get("state_observations")
        if observations:
            lines.extend(["", "## 当下压力与情绪", ""])
            lines.extend(f"- {item}" for item in observations["summary_lines"])
        return "\n".join(lines)
