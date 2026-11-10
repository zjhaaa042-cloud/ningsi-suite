"""结构化状态评估报告（文档 7.2.10）。

报告包含状态评分、问题分析、改善建议，并逐条回填证据：每条结论都能指回所用指标、
窗口统计与参数版本，使结果可复算、可追溯。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from ningsi import config
from ningsi.assessment.joint import JointAssessment


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
                "training": "neurofeedback-v2",
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
            f"- 软件版本：{config.VERSION}；预处理协议：preprocess-v2；训练协议：neurofeedback-v2",
            f"- 解释边界：{config.VALIDATION_NOTE}",
            "",
        ]
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
        indicator_summary = self.indicators.get("summary", {}) if isinstance(self.indicators, dict) else {}
        labels = {"focus": "专注相关脑电指数", "relax": "放松度", "load": "认知负荷"}
        if indicator_summary:
            for key, label in labels.items():
                stat = indicator_summary.get(key)
                if stat:
                    lines.append(f"- {label}：{stat['mean']:.2f}（波动 {stat['std']:.2f}，n={stat['n']}）")
        else:
            lines.append("- 无可用指标（见采集质量）")
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
        return "\n".join(lines)
