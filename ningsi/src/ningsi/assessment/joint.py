"""三类证据联合评估（文档 8.5）。

流程固定为六步：证据独立计算 → 质量前置判定 → 一致性检查 → 结论输出 → 边界提示 → 证据回填。
量表与脑电指标不一致时明确标注"结果不一致，建议复测"，不强行合并成单一分数。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ningsi import config

BOUNDARY_NOTE = "本结论仅用于研究与自我调节参考，不构成医疗诊断，也不得用于处罚或自动上岗决策。"


@dataclass(frozen=True)
class Evidence:
    code: str
    source: str          # eeg / scales / behavior
    dimension: str       # stress / attention
    summary: str
    value: object
    direction: str       # elevated / low / normal / unknown
    available: bool = True
    ref: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "source": self.source,
            "dimension": self.dimension,
            "summary": self.summary,
            "value": self.value,
            "direction": self.direction,
            "available": self.available,
            "ref": self.ref,
        }


@dataclass(frozen=True)
class JointAssessment:
    conclusion: str
    dimension_states: dict
    consistency: dict
    evidence: tuple
    advice: tuple
    boundary: str = BOUNDARY_NOTE
    spec: str = config.LABEL_SPEC

    def as_dict(self) -> dict:
        return {
            "spec": self.spec,
            "conclusion": self.conclusion,
            "dimension_states": self.dimension_states,
            "consistency": self.consistency,
            "evidence": [e.as_dict() for e in self.evidence],
            "advice": list(self.advice),
            "boundary": self.boundary,
        }


def _scale_evidence(scale_result, dimension: str) -> Evidence:
    code = scale_result.code
    elevated = scale_result.standard_score >= config.ASSESS["scale_elevated"]
    return Evidence(
        code=f"{code.lower()}_standard_score",
        source="scales",
        dimension=dimension,
        summary=f"{scale_result.name}标准分 {scale_result.standard_score}（{scale_result.level}）",
        value={"raw": scale_result.raw_score, "standard": scale_result.standard_score, "level": scale_result.level},
        direction="elevated" if elevated else "normal",
        ref={"scale_version": scale_result.version, "reverse_items": list(scale_result.reverse_items)},
    )


def assess(eeg, scales=None, behavior=None) -> JointAssessment:
    """eeg: 指标汇总字典；scales: {'SAS': ScaleResult, 'SDS': ScaleResult}；behavior: {'sart':..,'pvt':..}。"""
    scales = scales or {}
    behavior = behavior or {}
    thresholds = config.ASSESS
    evidence: list[Evidence] = []

    valid_ratio = float(eeg.get("valid_ratio", 0.0))
    eeg_available = bool(eeg.get("available", True)) and valid_ratio >= thresholds["min_valid_ratio"]
    focus = eeg.get("focus")
    relax = eeg.get("relax")
    load = eeg.get("load")

    if eeg_available and focus is not None and load is not None:
        evidence.append(Evidence(
            code="eeg_focus",
            source="eeg",
            dimension="attention",
            summary=f"专注度评分 {focus:.2f}（可用窗比例 {valid_ratio:.0%}）",
            value=round(float(focus), 4),
            direction="low" if focus < thresholds["focus_low"] else "normal",
            ref={"indicator_spec": eeg.get("spec", config.INDICATOR_SPEC), "baseline_spec": eeg.get("baseline_spec")},
        ))
        evidence.append(Evidence(
            code="eeg_load",
            source="eeg",
            dimension="stress",
            summary=f"认知负荷评分 {load:.2f}",
            value=round(float(load), 4),
            direction="elevated" if load >= thresholds["load_high"] else "normal",
            ref={"indicator_spec": eeg.get("spec", config.INDICATOR_SPEC)},
        ))
        if relax is not None:
            evidence.append(Evidence(
                code="eeg_relax",
                source="eeg",
                dimension="stress",
                summary=f"放松度评分 {relax:.2f}",
                value=round(float(relax), 4),
                direction="low" if relax < thresholds["relax_low"] else "normal",
                ref={"indicator_spec": eeg.get("spec", config.INDICATOR_SPEC)},
            ))
    else:
        reason = "基线不可用" if not eeg.get("available", True) else f"可用窗比例 {valid_ratio:.0%} 低于 {thresholds['min_valid_ratio']:.0%}"
        evidence.append(Evidence(
            code="eeg_unavailable", source="eeg", dimension="attention",
            summary=f"脑电指标不可用：{reason}", value=None, direction="unknown", available=False,
            ref={"valid_ratio": valid_ratio},
        ))
        evidence.append(Evidence(
            code="eeg_unavailable_load", source="eeg", dimension="stress",
            summary=f"脑电指标不可用：{reason}", value=None, direction="unknown", available=False,
            ref={"valid_ratio": valid_ratio},
        ))

    sas = scales.get("SAS")
    sds = scales.get("SDS")
    if sas is not None:
        evidence.append(_scale_evidence(sas, "stress"))
    if sds is not None:
        evidence.append(_scale_evidence(sds, "stress"))

    sart = behavior.get("sart")
    if sart:
        attention_flag = (
            sart.get("commission_rate", 0.0) > thresholds["commission_rate_high"]
            or sart.get("omission_rate", 0.0) > thresholds["omission_rate_high"]
            or sart.get("rt_variability", 0.0) > thresholds["rt_variability_high"]
        )
        evidence.append(Evidence(
            code="sart_performance",
            source="behavior",
            dimension="attention",
            summary=(
                f"SART 正确率 {sart.get('go_accuracy', 0):.0%}、虚报率 {sart.get('commission_rate', 0):.1%}、"
                f"漏报率 {sart.get('omission_rate', 0):.1%}、反应时变异 {sart.get('rt_variability', 0):.2f}"
            ),
            value={k: sart.get(k) for k in ("go_accuracy", "commission_rate", "omission_rate", "rt_mean", "rt_variability")},
            direction="low" if attention_flag else "normal",
            ref={"sequence_set_id": sart.get("sequence_set_id"), "seed": sart.get("seed"), "spec": sart.get("spec")},
        ))
    pvt = behavior.get("pvt")
    if pvt:
        evidence.append(Evidence(
            code="pvt_alertness",
            source="behavior",
            dimension="attention",
            summary=f"PVT-B 中位反应时 {pvt.get('rt_median', 0):.3f}s、慢反应率 {pvt.get('lapse_rate', 0):.1%}",
            value={"rt_median": pvt.get("rt_median"), "lapse_rate": pvt.get("lapse_rate")},
            direction="low" if pvt.get("lapse_rate", 0.0) > thresholds["lapse_rate_high"] else "normal",
            ref={"task": "pvt-b"},
        ))

    dimension_states, consistency = _consistency(evidence, thresholds)
    conclusion = _conclusion(dimension_states, consistency, evidence)
    advice = _advice(dimension_states, consistency)
    return JointAssessment(
        conclusion=conclusion,
        dimension_states=dimension_states,
        consistency=consistency,
        evidence=tuple(evidence),
        advice=tuple(advice),
    )


def _consistency(evidence, thresholds) -> tuple[dict, dict]:
    by_dimension: dict[str, list] = {}
    for item in evidence:
        by_dimension.setdefault(item.dimension, []).append(item)

    states = {}
    for dimension, items in by_dimension.items():
        usable = [i for i in items if i.available]
        flagged = [i for i in usable if i.direction in ("elevated", "low")]
        if not usable:
            state = "证据不足"
        elif not flagged:
            state = "正常范围"
        elif len(flagged) == len(usable):
            state = "一致提示偏离"
        else:
            state = "部分提示偏离"
        states[dimension] = {
            "state": state,
            "evidence_codes": [i.code for i in usable],
            "flagged": [i.code for i in flagged],
            "missing": [i.code for i in items if not i.available],
        }
    consistency = {
        "eeg_vs_scale": _pair_consistency(states.get("stress"), "量表"),
        "eeg_vs_behavior": _pair_consistency(states.get("attention"), "行为任务"),
    }
    return states, consistency


def _pair_consistency(state, other_label: str) -> str:
    """脑电与另一侧证据的一致性判定。

    之前这里直接写 `"不一致" if _mismatch(...) else ("一致" if ... else "无冲突")`，
    而 `_mismatch()` 在任一侧缺证据时返回 False，于是"脑电完全不可用 + 量表提示偏离"
    会落到 `状态 == 一致提示偏离 → "一致"`，凭空断言"脑电与量表一致"——真机会话
    `be4114d37dc94cf393673235be4150e3` 的报告就是这样写的（0/35 可用窗 + 一致）。
    因此缺证据时明确给出"无法比较"，并写清缺的是哪一侧。
    """
    if not state:
        return f"无法比较（{other_label}与脑电均无可用证据）"
    codes = list(state.get("evidence_codes", []))
    has_eeg = any(code.startswith("eeg_") for code in codes)
    has_other = any(not code.startswith("eeg_") for code in codes)
    if not has_eeg and not has_other:
        return f"无法比较（{other_label}与脑电均无可用证据）"
    if not has_eeg:
        return "无法比较（脑电不可用）"
    if not has_other:
        return f"无法比较（{other_label}不可用）"
    if _mismatch(state, "eeg", other_label):
        return "不一致"
    return "一致" if state.get("state") == "一致提示偏离" else "无冲突"


def _mismatch(state, source_a: str, source_b: str) -> bool:
    if not state:
        return False
    codes = state.get("flagged", [])
    has_a = any(c.startswith("eeg_") for c in codes)
    has_b = any(not c.startswith("eeg_") for c in codes)
    available_a = any(c.startswith("eeg_") for c in state.get("evidence_codes", []))
    available_b = any(not c.startswith("eeg_") for c in state.get("evidence_codes", []))
    if not (available_a and available_b):
        return False
    return has_a != has_b


DIMENSION_CONCLUSION = {
    "stress": "压力相关指标偏高，建议休息与放松训练",
    "attention": "专注力相关指标偏低，建议进行专注力强化训练",
}


def _conclusion(states, consistency, evidence) -> str:
    available = [i for i in evidence if i.available]
    if not available:
        return "数据不足，无法给出结论（请检查设备与佩戴）"
    attention = states.get("attention", {})
    stress = states.get("stress", {})
    if attention.get("state") == "证据不足" and stress.get("state") == "证据不足":
        return "数据不足，无法给出结论（请补充量表或重测行为任务）"

    inconsistent = "不一致" in consistency.values()
    consistent_flags = [name for name in ("stress", "attention") if states.get(name, {}).get("state") == "一致提示偏离"]
    partial_flags = [name for name in ("stress", "attention") if states.get(name, {}).get("state") == "部分提示偏离"]
    target = (consistent_flags or partial_flags or [None])[0]

    if target is None:
        if inconsistent:
            return "结果不一致，建议复测（脑电指标与其他证据方向不一致）"
        if any(states.get(name, {}).get("state") in ("证据不足", None) for name in ("stress", "attention")):
            return "证据不足，暂不给出状态结论（请补充缺失的评估环节）"
        return "本次各项证据均落在个体基线与常模范围内"

    conclusion = DIMENSION_CONCLUSION[target]
    if inconsistent:
        conclusion += "；另有维度与其他证据不一致，建议复测"
    return conclusion


def _advice(states, consistency) -> list:
    advice = []
    insufficient = [name for name, value in states.items() if value.get("state") == "证据不足"]
    if insufficient:
        advice.append("证据不足的维度：" + "、".join(insufficient) + "；请先完成量表填写与行为任务，或在安静环境重测脑电。")
    if "不一致" in consistency.values():
        advice.append("在安静环境、固定佩戴位置下复测一次，优先核对伪迹比例与基线质量。")
    if states.get("stress", {}).get("state") in ("一致提示偏离", "部分提示偏离"):
        advice.append("安排 5–10 分钟腹式呼吸或闭眼放松，并观察放松度评分是否回升。")
    if states.get("attention", {}).get("state") in ("一致提示偏离", "部分提示偏离"):
        advice.append("按个体基线中位数设置训练目标，先完成 2 段短时专注训练并记录达标时间占比。")
    if not advice or all(item.startswith("证据不足") for item in advice):
        advice.append("维持当前作息与训练频次，按周复评以观察趋势。")
    advice.append(BOUNDARY_NOTE)
    return advice
