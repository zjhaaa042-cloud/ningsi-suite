"""Presentation layer for the offline report. All quantities come from the saved run."""
from collections import Counter
from datetime import datetime
from html import escape
import math


STYLE = """
*{box-sizing:border-box}html{color-scheme:light}body{margin:0;background:#f1f4f3;color:#163431;
font:16px/1.65 "Segoe UI","Microsoft YaHei",system-ui,sans-serif;font-variant-numeric:tabular-nums}
main{width:min(1080px,100%);margin:auto;padding:22px 32px 56px}
.mast,.chapter,.meta,.measurements,.round-head,.round-line,.baseline-grid{display:flex;align-items:center}
.mast{justify-content:space-between;font-size:13px;color:#47605e;margin-bottom:15px;gap:12px}
.mast b{font-size:16px;color:#183e38;letter-spacing:.04em}
.hero{background:linear-gradient(118deg,#123e42 0%,#164e50 65%,#24615a 100%);color:white;padding:29px 36px;border-radius:14px}
.hero p{margin:0}.hero .protocol{color:#b9ded5;font-weight:600;font-size:14px}
h1{font-size:clamp(25px,4vw,39px);line-height:1.25;margin:7px 0 5px;overflow-wrap:anywhere}
.origin{font-size:17px;color:#e4f0ed}.meta{flex-wrap:wrap;gap:7px 24px;margin-top:18px;
padding-top:14px;border-top:1px solid #4d706a;color:#cde3de;font-size:13px}
section,.appendix,.raw{border:1px solid #d4dfdb;border-radius:12px;background:white;margin:16px 0;padding:23px 28px}
h2{font-size:21px;margin:0 0 9px;line-height:1.35}h3{line-height:1.4;font-size:17px}
p{margin:.5rem 0}.chapter{justify-content:space-between;gap:14px;flex-wrap:wrap}.chapter h2{margin:0}
.muted,.note{color:#47635c}.note{font-size:14px}.measurements{align-items:stretch;margin:17px 0 13px}
.metric{flex:1;border-left:1px solid #d6e2de;padding:0 16px;min-width:0}
.metric:first-child{border:0;padding-left:0}.metric dt{font-size:14px;color:#395951}
.metric dd{margin:4px 0 0;font-size:28px;font-weight:650;line-height:1.3;color:#103f37}
.metric dd small{font-size:14px;font-weight:500}.annotation{padding:10px 13px;border-left:3px solid #57867b;background:#edf5f2}
.baseline-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin:10px 0}
.baseline-grid article{padding:11px 16px;border:1px solid #d5e2dc;border-radius:7px}
.baseline-grid strong{display:block;margin-bottom:2px}.baseline-grid span{color:#47635c;font-size:14px}
.round-head,.round-line{display:grid;grid-template-columns:minmax(45px,.5fr) minmax(55px,1fr) minmax(70px,1.2fr) minmax(120px,2.2fr);gap:12px}
.round-head{color:#526862;font-size:13px;border-bottom:1px solid #cbdad4;margin-top:12px;padding:5px 0}
.round-line{padding:9px 0;border-bottom:1px solid #e3ebe7}.round-line strong{font-size:15px}
.round-meter{display:grid;gap:4px}.meter,.track{height:7px;border-radius:5px;overflow:hidden;background:#e4ebe8}
.meter i,.track>div{display:block;background:#287d69;height:100%}.round-meter span{font-size:14px}
.reasons{list-style:none;margin:7px 0;padding:0}.reasons li{display:flex;gap:12px;justify-content:space-between;padding:6px 0;border-bottom:1px solid #e3ebe7}
.reasons strong{white-space:nowrap;font-size:14px}.bar{display:grid;grid-template-columns:minmax(120px,15rem) 1fr 5rem;gap:10px;align-items:center;margin:7px 0}
.sequence{display:grid;grid-template-columns:4rem 1fr 6rem;align-items:center;gap:10px;margin:6px 0;font-size:13px}
.sequence .cells{display:flex;gap:2px;height:16px}.cells i{flex:1;background:#dccac1;border:1px solid #c6a99a;border-radius:2px}
.cells i.pass{background:#297762;border-color:#297762}.sequence small{text-align:right;color:#47635c}
.timeline{display:flex;flex-wrap:wrap;gap:3px}.window{width:10px;height:22px;background:#ad664b}.good{background:#287d69}
.appendix>summary,.raw>summary{cursor:pointer;font-weight:600;color:#244e46}.appendix section{padding:18px 0;margin:10px 0 0;border-width:1px 0 0;border-radius:0}
pre{font:13px/1.5 ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere}table{width:100%;border-collapse:collapse}
td,th{text-align:left;border-bottom:1px solid #d4dfdb;padding:8px}
.overview{border-top:3px solid #378371}.overview .annotation{margin-bottom:0}
.panel-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin:15px 0}
.panel-grid>section{min-width:0;margin:0}.panel-grid h2,.overview h2{font-size:21px}
.score-line{display:grid;grid-template-columns:7em minmax(25px,1fr) 3em;gap:10px;align-items:center;margin:12px 0}
.score-line strong{text-align:right;font-weight:650}.score-line .track{height:9px}
.score-line .track i{display:block;height:100%;background:#347b71;border-radius:inherit}
.task-block+.task-block,.questionnaire+.questionnaire{border-top:1px solid #dae5e1;margin-top:14px;padding-top:14px}
.task-block h3,.questionnaire h3{margin:0;font-size:16px}.task-block p,.questionnaire p{margin:3px 0}
.questionnaire{display:flex;justify-content:space-between;align-items:baseline;gap:12px}
.questionnaire strong{font-size:18px;white-space:nowrap}.panel-grid .note{margin-top:12px}
.subdetail{border-top:1px solid #dae5e1;margin-top:15px;padding-top:10px}
summary{cursor:pointer;color:#245d54;font-weight:600}
summary:focus-visible{outline:3px solid #c28635;outline-offset:4px}
.subdetail p{font-size:14px}.summary-meter{height:11px;background:#e1ece9;border-radius:4px;overflow:hidden}
.summary-meter span{display:block;height:100%;background:#508b7e}
@media(max-width:650px){main{padding:14px 10px 30px}.hero,section,.appendix,.raw{padding:16px}
.measurements{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:11px}.metric,.metric:first-child{padding:0;border:0}
.metric dd{font-size:23px}.round-head,.round-line{grid-template-columns:36px .8fr 1fr 1.5fr;gap:5px;font-size:13px}
.round-meter span{font-size:12px}.round-line strong{font-size:13px}.baseline-grid{grid-template-columns:1fr;gap:7px}
.bar{grid-template-columns:minmax(92px,1.5fr) 1fr 3rem;font-size:13px}.meta{gap:5px 14px}}
@media(max-width:760px){.panel-grid{grid-template-columns:1fr}.panel-grid .score-line{grid-template-columns:7em 1fr 3em}}
@media print{@page{size:A4;margin:12mm}body{background:white;font-size:11pt}main{width:100%;padding:0}.hero{background:white;color:#143d36;border:1px solid #629082}
.hero .protocol,.origin,.meta{color:#375f56}.mast{margin:0 0 6px}.hero,section,.appendix,.raw{padding:10px 14px;margin:7px 0}
section,.round-line{break-inside:avoid}details:not([open]){display:none}.meter,.track,.meter i,.track>div,.cells i{print-color-adjust:exact}}
"""


def _finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def _number(value, digits=1):
    if not _finite(value):
        return '未记录'
    return f'{value:.{digits}f}'.rstrip('0').rstrip('.') if digits else str(round(value))


def _quantity(value, unit, digits=1):
    return f'{_number(value, digits)} {unit}' if _finite(value) else '未记录'


def _total(values):
    return sum(values) if values and all(_finite(v) for v in values) else None


def report_header(report):
    extras = report.get('extras') or {}
    context = extras.get('report_context') or {}
    protocol = extras.get('protocol_label') or '会话报告'
    titles = {'full': '完整评估报告', 'short': '短程评估报告', 'ten_min': '十分钟评估与训练报告',
              'effective_time': '有效时长训练报告'}
    title = context.get('title') or titles.get(extras.get('protocol'), protocol)
    source = context.get('source_label') or extras.get('source_note') or '来源未记录'
    facts = []
    if context.get('started_at'):
        stamp = str(context['started_at'])
        try:
            date = datetime.fromisoformat(stamp.replace('Z', '+00:00'))
            if date.tzinfo is not None:
                date = date.astimezone()
            stamp = date.isoformat(sep=' ', timespec='seconds')
        except ValueError:
            pass
        facts.append('会话 ' + stamp)
    channels = context.get('channels') or extras.get('channels')
    srate = context.get('srate') or extras.get('srate')
    if _finite(channels):
        facts.append(f'{channels:g} 通道')
    if _finite(srate):
        facts.append(f'{srate:g} Hz')
    if context.get('recording_ref'):
        facts.append('录波 ' + str(context['recording_ref']))
    if report.get('participant') and not context.get('hide_participant'):
        facts.append('记录编号 ' + str(report['participant']))
    if context.get('uuid') and not context.get('title') and not context.get('hide_participant'):
        facts.append('会话编号 ' + str(context['uuid'])[:8])
    facts_html = ''.join('<span>' + escape(item) + '</span>' for item in facts)
    return (f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(str(title))} · 凝思 Studio</title>'
            f'<style>{STYLE}</style></head><body><main><div class="mast"><b>凝思 Studio</b><span>脑电采集与行为分析报告</span></div>'
            f'<header class="hero"><p class="protocol">{escape(str(protocol)) if title != protocol else "脑电会话"}</p>'
            f'<h1>{escape(str(title))}</h1><p class="origin">{escape(str(source))}</p>'
            + (f'<div class="meta">{facts_html}</div>' if facts else '') + '</header>')


def _metric(label, value):
    return f'<div class="metric"><dt>{label}</dt><dd>{value}</dd></div>'


def _count_pair(count, total, unit):
    if not _finite(count) or not _finite(total):
        return _quantity(count, unit, 0)
    return f'{_number(count,0)} <small>/ {_number(total,0)} {unit}</small>'


def assessment_presentation(report):
    extras = report.get('extras') or {}
    context = extras.get('report_context') or {}
    automated = context.get('response_mode') == 'automated'
    generated = context.get('signal_origin') == 'generated' or extras.get('source_kind') in ('sim', 'simulation')
    replay = extras.get('replay')
    quality = report.get('quality') or {}
    available, total = quality.get('windows_usable'), quality.get('windows_total')
    summary = (report.get('indicators') or {}).get('summary') or {}
    scales = report.get('scales') or {}
    behavior = report.get('behavior') or {}
    pvt, sart = behavior.get('pvt') or {}, behavior.get('sart') or {}
    has_pvt = _finite(pvt.get('trials')) and pvt['trials'] > 0
    has_sart = _finite(sart.get('trials')) and sart['trials'] > 0
    training = extras.get('training') or {}
    measures = []
    if _finite(available) and _finite(total):
        measures.append(_metric('脑电监测可用窗口', _count_pair(available, total, '窗')))
    answer_counts = [s.get('answered') for s in scales.values() if isinstance(s, dict)]
    answered = _total(answer_counts)
    if answered is not None:
        measures.append(_metric('问卷已作答', _quantity(answered, '题', 0)))
    if has_pvt:
        rt = pvt.get('rt_median')
        measures.append(_metric('中位反应时', _quantity(rt * 1000 if _finite(rt) else None, '毫秒', 0)))
    if _finite(training.get('valid_seconds')):
        measures.append(_metric('有效反馈时长', _quantity(training['valid_seconds'], '秒')))
    if automated and generated:
        note = '脑电由模拟器生成，任务由程序作答。本页用于核对采集、作答和计分流程。'
    elif automated:
        note = '反应时任务由程序作答。下方的应答速度描述这次程序测试。'
    elif generated:
        note = '本次脑电由模拟器生成，指数用于核对计算流程。'
    elif replay:
        note = '本次评估使用已保存的脑电录波。'
    else:
        note = (report.get('assessment') or {}).get('conclusion') or ''
    parts = ['<section class="overview"><h2>评估概览</h2>']
    if measures:
        parts.append('<dl class="measurements">' + ''.join(measures) + '</dl>')
    if note:
        parts.append(f'<p class="annotation">{escape(str(note))}</p>')
    if not measures and not note:
        parts.append('<p>这份记录未保存可供汇总的评估结果。</p>')
    parts.append('</section><div class="panel-grid">')
    indices = []
    for code, name in [('focus','专注相关指数'), ('relax','放松指数'), ('load','负荷指数')]:
        stat = summary.get(code) or {}
        mean = stat.get('mean')
        if _finite(mean) and 0 <= mean <= 1 and _finite(stat.get('n')) and stat['n'] > 0:
            indices.append(f'<div class="score-line"><span>{name}</span><span class="track" aria-hidden="true"><i style="width:{mean*100:.2f}%"></i></span><strong>{_number(mean,2)}</strong></div>')
    if indices or _finite(total):
        parts.append('<section><h2>脑电监测</h2>')
        monitor = (extras.get('assessment_eeg_collection') or {}).get('monitor') or {}
        if _finite(monitor.get('valid_seconds')):
            parts.append(f'<p>独立监测累计取得 <strong>{_quantity(monitor["valid_seconds"], "秒")}</strong> 可用脑电（协议时间）。</p>')
        if indices:
            parts.extend(indices)
            parts.append('<p class="note">图中是可评分窗口的平均指数。刻度从 0 到 1，表示相对本次基线的变化。</p>')
        else:
            parts.append('<p>这次独立监测没有保存可用的脑电指数。</p>')
        parts.append('</section>')
    if has_sart or has_pvt:
        parts.append('<section><h2>行为任务</h2>')
        if has_sart:
            parts.append(f'<div class="task-block"><h3>持续注意</h3><p>正式试次 <strong>{_quantity(sart["trials"],"次",0)}</strong></p>')
            parts.append(f'<p class="muted">应按时漏按 {_quantity(sart.get("omission_errors"),"次",0)}，应停时误按 {_quantity(sart.get("commission_errors"),"次",0)}。</p>')
            if sart.get('valid') is False:
                parts.append('<p class="note">这次持续注意任务未达到有效性要求。</p>')
            parts.append('</div>')
        if has_pvt:
            parts.append(f'<div class="task-block"><h3>反应时</h3><p>有效应答 <strong>{_count_pair(pvt.get("responded"),pvt["trials"],"次")}</strong></p>')
            parts.append(f'<p class="muted">漏答 {_quantity(pvt.get("missed"),"次",0)}，抢答 {_quantity(pvt.get("false_starts"),"次",0)}。</p>')
            if pvt.get('valid') is False:
                parts.append('<p class="note">这次反应时任务未达到有效性要求。</p>')
            parts.append('</div>')
        parts.append('</section>')
    scale_rows, scoring = [], []
    for code in ('SAS', 'SDS'):
        scale = scales.get(code) or {}
        if not scale:
            continue
        name = escape(str(scale.get('name') or {'SAS':'焦虑自评量表','SDS':'抑郁自评量表'}[code]))
        count = scale.get('answered')
        missing = scale.get('missing_items')
        size = count + len(missing) if _finite(count) and isinstance(missing,list) else None
        scale_rows.append(f'<div class="questionnaire"><h3>{name} <small>({code})</small></h3><strong>{_count_pair(count,size,"题")}</strong></div>')
        level = scale.get('level')
        level_text = f'，记录分级为{escape(str(level))}' if level and not (automated or generated) else ''
        scoring.append(f'<p>{code} 原始分 {_number(scale.get("raw_score"),0)}，标准分 {_number(scale.get("standard_score"),0)}{level_text}。</p>')
    if scale_rows:
        parts.extend(('<section><h2>问卷完成情况</h2>', ''.join(scale_rows)))
        if automated and generated:
            parts.append('<p class="note">本次为模拟流程，量表分数用于核对计分。</p>')
        parts.extend(('<details class="subdetail"><summary>查看问卷计分</summary>', ''.join(scoring), '</details></section>'))
    valid, reached = training.get('valid_seconds'), training.get('on_target_seconds')
    if _finite(valid) and _finite(reached):
        parts.append('<section><h2>训练反馈</h2>')
        parts.append(f'<p>按协议时间累计，有效反馈 <strong>{_quantity(valid,"秒")}</strong>，其中达到训练目标 <strong>{_quantity(reached,"秒")}</strong>。</p>')
        if valid > 0 and 0 <= reached <= valid:
            ratio = 100 * reached / valid
            parts.append(f'<div class="summary-meter" aria-hidden="true"><span style="width:{ratio:.2f}%"></span></div><p class="note">有效反馈时间内的达标比例为 {_number(ratio)}%。</p>')
        parts.append('</section>')
    parts.append('</div>')
    return ''.join(parts)


def _quality_sequences(segments, rows):
    # One diagnostic is logged at each trainer.feed(). The saved accepted and excluded
    # feed counts give the round boundaries only when every diagnostic is present.
    sizes = []
    for segment in segments:
        scored = (segment.get('stats') or {}).get('n')
        excluded = segment.get('excluded_windows')
        if not all(type(n) is int and n >= 0 for n in (scored, excluded)):
            return ''
        sizes.append(scored + excluded)
    if not rows or sum(sizes) != len(rows) or any(type(r.get('usable')) is not bool for r in rows):
        return ''
    parts = ['<h3>逐窗质检记录</h3><p class="note">每格代表一次训练更新，按先后顺序排列。深绿表示通过严格质检，浅棕表示未通过。</p>']
    start = 0
    for index, size in enumerate(sizes,1):
        group = rows[start:start+size]
        start += size
        cells = ''.join('<i class="pass"></i>' if r['usable'] else '<i></i>' for r in group)
        count = sum(r['usable'] for r in group)
        parts.append(f'<div class="sequence"><span>第 {index} 轮</span><div class="cells" aria-hidden="true">{cells}</div>'
                     f'<small>{count} / {size} 窗通过</small></div>')
    return ''.join(parts)


def training_presentation(report, reason_labels):
    extras = report['extras']
    ten = extras['ten_min']
    segments = ten.get('segments') or []
    overview = ten.get('indicator_overview') or {}
    diag = extras.get('signal_diagnostics') or {}
    training_rows = [r for r in diag.get('windows', []) if r.get('phase') == 'training']
    observed = (overview.get('quality') or {}).get('windows_total')
    usable = (overview.get('quality') or {}).get('windows_usable')
    if observed is None and training_rows:
        observed = len(training_rows)
    if usable is None and training_rows:
        usable = sum(r.get('usable') is True for r in training_rows)
    measured = _total([s.get('measured_focus_updates') for s in segments])
    credited = _total([(s.get('stats') or {}).get('valid_seconds') for s in segments])
    supported = _total([(s.get('local_eeg_feedback') or {}).get('support_seconds') for s in segments])
    attempts = (f'{_number(measured)} <small>/ {_number(observed)} 次</small>'
                if _finite(measured) and _finite(observed) else _quantity(measured, '次'))
    duration = ten.get('elapsed_sec')
    actual = ''
    if _finite(duration):
        mins, seconds = divmod(round(duration), 60)
        actual = f'<span class="note">实际用时 {mins} 分 {seconds:02d} 秒</span>'
    parts = [f'<section class="overview" aria-label="训练结果"><div class="chapter"><h2>训练结果</h2>{actual}</div>',
             '<dl class="measurements">',
             _metric('显示专注指数的更新', attempts),
             _metric('通过严格质检的训练窗', _quantity(usable, '窗', 0)),
             _metric('可分析的局部频带', _quantity(supported, '秒')),
             _metric('有效反馈时长', _quantity(credited, '秒')), '</dl>']
    if measured == 0 and supported is not None and supported > 0:
        outcome = '这段训练保留了可分析的局部脑电频带，专注指数尚未达到输出条件。'
    elif measured == 0:
        outcome = '本次训练没有产生可显示的专注指数。'
    elif _finite(measured) and measured > 0:
        outcome = '专注指数反映相对本次基线的变化。持续满足反馈条件的时段计入有效反馈。'
    else:
        outcome = '这份历史报告保留了训练统计，部分逐次显示数据未保存。'
    context = extras.get('report_context') or {}
    if context.get('signal_origin') == 'generated' or extras.get('source_kind') in ('sim', 'simulation'):
        outcome = '本次使用模拟脑电验证反馈流程。' + outcome
    parts.extend((f'<p class="annotation">{outcome}</p>', '</section>'))
    summary = overview.get('summary') or {}
    indices = []
    for key, name in [('focus', '专注相关指数'), ('relax', '放松度'), ('load', '认知负荷')]:
        stat = summary.get(key) or {}
        if _finite(stat.get('mean')) and _finite(stat.get('n')) and stat['n'] > 0:
            indices.append(_metric(name, f'{_number(stat["mean"],3)} <small>（{stat["n"]:g} 窗）</small>'))
    if indices:
        parts.extend(('<details class="appendix"><summary>查看训练期间的脑电指数</summary><dl class="measurements">',
                      ''.join(indices), '</dl><p class="note">数值为可评分训练窗的平均指数，范围 0–1，依据本次训练前基线计算。</p></details>'))

    before = extras.get('baseline_eyes_open') or {}
    after = (extras.get('training') or {}).get('baseline_after') or {}
    if before or after or training_rows:
        parts.append('<section><h2>信号质量与评分条件</h2>')
        if before or after:
            parts.append('<div class="baseline-grid">')
            for label, base in [('训练前基线', before), ('训练后基线', after)]:
                status = {True: '通过', False: '未通过'}.get(base.get('valid'), '未记录')
                quality = base.get('quality') or {}
                details = []
                if _finite(quality.get('usable_seconds')):
                    details.append('可用 ' + _quantity(quality['usable_seconds'], '秒'))
                if _finite(quality.get('valid_ratio')):
                    details.append('合格窗占比 ' + _number(100 * quality['valid_ratio']) + '%')
                parts.append(f'<article><strong>{label} · {status}</strong><span>{"，".join(details)}</span></article>')
            parts.append('</div>')
        if before.get('valid') is False:
            parts.append('<p>训练前基线未达到校准条件，系统缺少计算个人专注指数所需的参照。</p>')
        # Count a reason at most once per window. Each window may carry multiple different reasons.
        reasons = Counter(code for r in training_rows if not r.get('usable')
                          for code in dict.fromkeys(r.get('reasons') or (r.get('quality') or {}).get('reasons') or []))
        if reasons:
            top = sorted(reasons.items(), key=lambda pair: (-pair[1],pair[0]))[:3]
            parts.append('<details class="subdetail"><summary>查看训练窗未通过的原因</summary><ul class="reasons">')
            for code,count in top:
                parts.append(f'<li><span>{escape(str(reason_labels.get(code,code)))}</span><strong>{count} 窗</strong></li>')
            parts.append('</ul><p class="note">同一窗可以有多项标记。相邻窗口有重叠，次数不能直接换算成时长。</p></details>')
        parts.append('</section>')
    if segments:
        parts.append(f'<section><h2>逐轮反馈</h2><div class="round-head"><span>轮次</span><span>指数更新</span>'
                     '<span>局部频带</span><span>有效反馈 / 训练时长</span></div>')
        for idx, seg in enumerate(segments, 1):
            valid = (seg.get('stats') or {}).get('valid_seconds')
            dur = seg.get('duration_sec')
            coverage = ''
            if _finite(valid) and _finite(dur) and dur > 0:
                width = max(0,min(100,100*valid/dur))
                coverage = f'<div class="meter" aria-hidden="true"><i style="width:{width:.2f}%"></i></div>'
            relative = (seg.get('local_eeg_feedback') or {}).get('support_seconds')
            parts.append(f'<div class="round-line"><strong>第 {idx} 轮</strong>'
                         f'<span>{_quantity(seg.get("measured_focus_updates"),"次")}</span><span>{_quantity(relative,"秒")}</span>'
                         f'<div class="round-meter"><span>{_quantity(valid,"秒")} / {_quantity(dur,"秒")}</span>{coverage}</div></div>')
        parts.append('<p class="note">局部频带时长按有数据支持的时间去重。它与反馈时长可能重叠。训练时长采用协议时间。</p>')
        sequence = _quality_sequences(segments, training_rows)
        if sequence:
            parts.extend(('<details class="subdetail"><summary>查看逐次信号质量</summary>', sequence, '</details>'))
        parts.append('</section>')
    pvt = (report.get('behavior') or {}).get('pvt') or {}
    if pvt:
        automated = (extras.get('report_context') or {}).get('response_mode') == 'automated'
        description = ('程序在刺激出现后自动提交应答。这里记录任务链路和程序响应时间，无法用来评价参与者的警觉度。'
                       if automated else '中位反应时根据正式任务保存的有效应答计算。')
        rt = pvt.get('rt_median')
        parts.extend(('<section><h2>反应时任务</h2>', f'<p>{description}</p><dl class="measurements">',
                      _metric('有效应答 / 正式试次', f'{_number(pvt.get("responded"))} <small>/ {_number(pvt.get("trials"))} 次</small>'),
                      _metric('中位反应时', _quantity(1000*rt if _finite(rt) else None, '毫秒', 0)),
                      _metric('漏答', _quantity(pvt.get('missed'), '次')), '</dl></section>'))
    return ''.join(parts)
