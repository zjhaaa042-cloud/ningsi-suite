"""不依赖有效个体基线的采集诊断；被拒绝窗口仅用于诊断，不参与评分。"""
from collections import Counter
from html import escape
import json
import math

from ningsi import config

REASONS = {
    'amplitude': '幅值超限', 'flat_channel': '平直通道', 'channel_span': '通道差异过大',
    'emg': '高频占比偏高（肌电代理）', 'eog': '低频占比偏高（眼电/漂移代理）',
    'no_data': '无样本', 'signal_too_short': '未满4秒', 'non_finite_signal': '非有限值',
}


def summarize_windows(rows):
    rows = list(rows)
    reasons = Counter(reason for row in rows for reason in row.get('reasons', row.get('unusable_reasons', [])))
    grades = Counter(r.get('graded', {}).get('grade') for r in rows if r.get('graded', {}).get('grade'))
    bands = {}
    for label, accepted in [('accepted', True), ('rejected', False)]:
        subset = [r for r in rows if bool(r.get('usable')) == accepted]
        bands[label] = {}
        for band in config.BANDS:
            values = [r.get('relative_band_powers', {}).get(band) for r in subset]
            values = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
            bands[label][band] = {'mean': sum(values)/len(values) if values else None, 'n': len(values)}
    sensitivity = []
    for threshold in (150, 200, 250, 300, 500):
        count = 0
        evaluated = 0
        for row in rows:
            metrics = row.get('quality_metrics') or row.get('quality', {}).get('metrics', {})
            peak = metrics.get('peak_uv')
            if peak is None or not math.isfinite(peak):
                continue
            evaluated += 1
            other = set(row.get('reasons', row.get('unusable_reasons', []))) - {'amplitude'}
            count += not other and peak <= threshold
        sensitivity.append({'amp_max_uv': threshold, 'accepted': count, 'evaluated': evaluated})
    return {
        'spec': 'signal-diagnostics-v1', 'threshold_status': config.VALIDATION_STATUS,
        'thresholds': dict(config.QUALITY), 'total': len(rows),
        'accepted': sum(bool(r.get('usable')) for r in rows),
        'reason_counts': dict(reasons), 'reason_labels': REASONS, 'bands': bands,
        'windows': rows, 'amplitude_sensitivity': sensitivity, 'grades': dict(grades),
        'scope': '本图统计已记录的质检、基线、任务态监测及训练窗口；不是仅任务态评分窗。行为任务期间未同步保存脑电窗。按采集顺序排列，间隔不代表连续时间。',
        'note': '合格窗用于严格评分；被拒绝窗的频带仅描述采集信号，可能包含眼电、肌电或漂移，不能解释为注意力。窗口有重叠，窗数不等于独立样本数。',
    }


def render_html(report):
    """离线可打开、无需外部资源的可解释报告。所有输入文本转义。"""
    d = report.get('extras', {}).get('signal_diagnostics') or {}
    q = report.get('quality') or {}
    total, accepted = d.get('total', q.get('windows_total', 0)), d.get('accepted', q.get('windows_usable', 0))
    def esc(value):
        return escape(str(value))
    def bar(label, value, maximum, color='#246b73'):
        width = max(0, min(100, 100*value/maximum)) if maximum else 0
        return f'<div class="bar"><span>{esc(label)}</span><div class="track"><div style="width:{width:.2f}%;background:{color}"></div></div><b>{value:g}</b></div>'
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>脑电采集与行为分析报告</title>',
             '<style>body{font:16px/1.65 system-ui,sans-serif;background:#f2f5f4;color:#18383c;margin:auto;max-width:1100px;padding:32px}section{background:white;padding:24px;margin:20px 0;border-radius:14px}h1,h2{line-height:1.3}.muted{color:#52686c}.bar{display:grid;grid-template-columns:240px 1fr 60px;gap:12px;align-items:center;margin:10px 0}.track{background:#e8eeee;height:18px}.track>div{height:100%}.timeline{display:flex;flex-wrap:wrap;gap:3px}.window{width:12px;height:28px;background:#b35b45}.good{background:#246b73}pre{white-space:pre-wrap;overflow-wrap:anywhere}table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}@media(max-width:600px){body{padding:12px}.bar{grid-template-columns:140px 1fr 45px}}@media print{section{break-inside:avoid}body{background:white}}</style>',
             '<h1>脑电采集与行为分析报告</h1>',
             '<p>描述性分析与个体评分分开呈现 · 工程阈值待验证 · 不作诊断</p>',
             f'<section><h2>1. 数据可用性</h2><p>严格合格窗 <b>{accepted}/{total}</b>。' + ('本次没有可用个体脑电评分；仍可查看采集质量和行为表现。' if not report.get('indicators', {}).get('summary') else '个体指标见完整证据。') + '</p>',
             f'<p>{esc(d.get("scope", ""))}</p><p class="muted">{esc(d.get("note", "旧版未保存完整逐窗数据，以下仅展示已有汇总，不能重建频谱或重新判定质量。"))}</p>',
             bar('严格合格', accepted, total), bar('排除', max(0,total-accepted),total,'#b35b45')]
    ten = report.get('extras', {}).get('ten_min')
    if ten:
        parts.append('<section><h2>十分钟测试＋训练</h2><p>' + esc(ten.get('note')) + '</p>')
        for grade, label in [('A', '严格可用'), ('B', '受限片段可用'), ('C', '不可解释')]:
            parts.append(bar(label, d.get('grades', {}).get(grade, 0), total))
        parts.append('<p>等级按窗口计数，重叠窗不是独立样本。B级仅作描述与质量反馈。</p>')
        parts.append('<h3>同通道训练前后变化</h3><p>' + esc(ten.get('comparison_note')) + '</p>')
        for channel, change in ten.get('changes', {}).items():
            parts.append('<p>通道 ' + esc(channel) + '：log(β/θ)中位数差 ' + esc(round(change, 4)) + '</p>')
        if not ten.get('changes'):
            parts.append('<p>没有同通道可比较片段；行为报告仍可使用。</p>')
        for key, label in [('baseline_before', '训练前'), ('baseline_after', '训练后')]:
            for ch, ref in ten.get(key, {}).items():
                parts.append('<p>' + esc(label) + '通道' + esc(ch) + '：去重覆盖 ' + esc(ref.get('usable_seconds', '未知')) + ' 秒；MAD ' + esc(ref.get('mad')) + '</p>')
        for segment in ten.get('segments', []):
            parts.append('<p>训练轮 ' + esc(segment['index']) + '（秒）：' + esc(segment.get('mode_seconds', {})) + '；未反馈秒数 ' + esc(segment.get('unobserved_seconds', '未知')) + '</p>')
        parts.append('<p>实际采集耗时 ' + esc(ten.get('elapsed_sec')) + ' 秒；倍率 ' + esc(ten.get('time_scale')) + '</p></section>')
    for key,label in [('baseline_eyes_open' ,'睁眼'),('baseline_eyes_closed','闭眼')]:
        base = report.get('extras',{}).get(key) or {}
        quality = base.get('quality',{})
        parts.append(f'<p>{label}基线：{"合格" if base.get("valid") else "不合格/未采集"}；有效覆盖 {esc(quality.get("usable_seconds", "未知"))} 秒；有效窗 {esc(base.get("n_windows",0))}。阈值：至少30秒、5窗、60%有效比例。</p>')
    parts.append('</section><section><h2>2. 窗口与排除原因</h2><p>绿色合格，橙色排除；悬停查看阶段及原因。各原因可重叠，计数之和可能超过窗数。</p><div class="timeline">')
    for row in d.get('windows',[]):
        label = f"{row.get('phase','')} / {row.get('index','')} / " + ', '.join(REASONS.get(r,r) for r in row.get('reasons',row.get('unusable_reasons',[])))
        parts.append(f'<span class="window {"good" if row.get("usable") else ""}" title="{esc(label)}"></span>')
    parts.append('</div>')
    for reason,count in (d.get('reason_counts', {}) if d else q.get('unusable_reasons', {})).items():
        parts.append(bar(REASONS.get(reason,reason),count,total,'#b35b45'))
    parts.append('<p class="muted">幅值150 µV、通道差100 µV、低频占比55%、高频占比35%为当前工程筛查门槛，尚未按设备验证。低频占比高不能单独证明眼电。</p>')
    if d.get('amplitude_sensitivity'):
        parts.append('<h3>阈值敏感性：只改变幅值门槛会怎样？</h3><p>其余条件保持不变。此表用于解释拒绝原因，不改变原判定、不证明更宽门槛科学有效。</p>')
        for item in d['amplitude_sensitivity']:
            parts.append(bar(f"幅值≤{item['amp_max_uv']}µV时合格窗",item['accepted'],total))
    parts.append('</section><section><h2>3. 频带组成（描述性）</h2>')
    if not d.get('bands'):
        parts.append('<p>此旧版会话未保存逐窗频带数据，无法事后重建。请使用新版采集。</p>')
    for group,label in [('accepted','严格合格窗'),('rejected','排除窗：仅用于污染诊断')]:
        parts.append(f'<h3>{label}</h3>')
        for band,stat in d.get('bands',{}).get(group,{}).items():
            if stat.get('mean') is not None:
                parts.append(bar(f'{band} %（n={stat["n"]}）',round(stat['mean']*100,1),100,'#246b73' if group=='accepted' else '#b35b45'))
    parts.append('</section><section><h2>4. 行为表现（独立于脑电）</h2>')
    pvt = report.get('behavior',{}).get('pvt') or {}
    if pvt:
        for name,key in [('有效反应','responded'),('抢答（排除RT）','false_starts'),('漏答','missed'),('慢反应 >500ms','lapses')]:
            parts.append(bar(name,pvt.get(key,0),pvt.get('trials',0)))
        parts.append(f'<p>中位反应时 {esc(pvt.get("rt_median"))} 秒；平均 {esc(pvt.get("rt_mean"))} 秒；慢反应率分母 {esc(pvt.get("responded"))}。抢答不进入反应时统计。阈值待验证。</p>')
    sart = report.get('behavior',{}).get('sart') or {}
    if sart:
        parts.append(f'<p>SART：正式 {esc(sart.get("trials"))} 试次；Go正确率 {esc(sart.get("go_accuracy"))}（不是全任务正确率）；No-Go误按 {esc(sart.get("commission_errors"))}/{esc(sart.get("nogo_trials"))}；漏按 {esc(sart.get("omission_errors"))}。</p>')
    parts.append('</section><section><h2>5. 结论与下一步</h2>')
    parts.append(f'<p>{esc((report.get("assessment") or {}).get("conclusion", "尚无联合结论"))}</p>')
    for advice in (report.get('assessment') or {}).get('advice',[]):
        parts.append(f'<p>{esc(advice)}</p>')
    parts.append('<p>有波形不等于有可用脑电，有描述性频带结果不等于注意力评分有效。量表与行为任务不能代替脑电基线。</p></section><details><summary>完整结构化证据</summary><pre>'+esc(json.dumps(report,ensure_ascii=False,indent=2))+'</pre></details></html>')
    return ''.join(parts)
