import { card, el, fmtNum, fmtPercent } from '../util.js';

const labels = { amplitude: '幅值超限', eog: '低频占比偏高（眼电/漂移代理）', emg: '高频占比偏高（肌电代理）', channel_span: '通道差异过大', flat_channel: '平直通道', signal_too_short: '未满4秒', no_data: '无样本', non_finite_signal: '非有限值', electrode_step: '电极突跳或移动', ocular_transient: '显著慢瞬态', muscle_burst: '短时高频肌电代理', emg_reference_burst: '同步肌电参考活动', motion_reference_excursion: '同步加速度参考活动', motion_reference_unavailable: '动作参考缺失或未同步' };

Object.assign(labels, { blink_uncalibrated: '疑似眨眼（未校准）', blink_boundary: '边界眨眼，支持不足',
  blink_dense: '密集或重叠眨眼', blink_calibration_mismatch: '个人眨眼配置已变化',
  blink_shape_mismatch: '眨眼形状不匹配', blink_channel_mismatch: '双通道眨眼关系不匹配',
  blink_residual: '眨眼校正残留', blink_signal_fault: '眨眼路径检测到信号异常', contact_dropout: '接触丢失或贴轨' });

function bar(label, value, max, rejected = false) {
  const number = Number.isFinite(value) ? value : 0;
  return el('div', { class: 'diagnostic-bar' }, [
    el('span', { text: label }),
    el('progress', { max: Number.isFinite(max) ? Math.max(1, max) : 1,
      value: Math.max(0, number), class: rejected ? 'diagnostic-rejected' : '', 'aria-label': label }),
    el('strong', { text: fmtNum(value, 1) }),
  ]);
}

export function renderDiagnostics(report) {
  const d = report.extras?.signal_diagnostics;
  const quality = report.quality || {};
  const total = d?.total ?? quality.windows_total;
  const accepted = d?.accepted ?? quality.windows_usable;
  const hasCounts = Number.isFinite(total) && Number.isFinite(accepted);
  const root = el('div', { class: 'stack' });
  if (d?.scope) root.append(el('p', { text: d.scope }));
  const countNote = hasCounts ? `严格合格 ${accepted}/${total} 窗。` : '未保存完整窗口质量计数。';
  const qualityNote = d ? (d.note || '质量统计按已保存结果展示；缺失的诊断项目不补算。')
    : '旧版本未保存完整逐窗信号；可查看已有质量与行为汇总，不能事后重建频谱。';
  root.append(el('p', { text: countNote + qualityNote }));
  if (hasCounts && total > 0) root.append(bar('严格合格窗', accepted, total), bar('排除窗', total - accepted, total, true));
  const ten = report.extras?.ten_min;
  if (ten) {
    root.append(el('h3', { text: '十分钟测试＋训练：分级利用' }), el('p', { text: ten.note }));
    const overview = ten.indicator_overview;
    if (overview) {
      root.append(el('h3', { text: '训练阶段脑电指数（描述性）' }), el('p', { text: overview.note }));
      for (const [key, label] of [['focus', '专注相关指数'], ['relax', '放松度'], ['load', '认知负荷']]) {
        const stat = overview.summary?.[key];
        root.append(el('p', { text: `${label}：${fmtNum(stat?.mean, 3)}（n=${stat?.n ?? 0}）` }));
      }
      if (overview.reason) root.append(el('p', { text: overview.reason }));
    }
    for (const [grade, label] of [['A', '严格可用'], ['B', '受限片段可用'], ['C', '不可解释']]) root.append(bar(label, d?.grades?.[grade] || 0, total));
    root.append(el('p', { text: ten.comparison_note }));
    for (const [ch, change] of Object.entries(ten.changes || {})) root.append(el('p', { text: `通道${ch}：log(β/θ)中位数变化 ${fmtNum(change, 4)}` }));
    if (!Object.keys(ten.changes || {}).length) root.append(el('p', { text: '无同通道可比较片段，保留行为与质量报告。' }));
    for (const [key, label] of [['baseline_before','训练前'], ['baseline_after','训练后']]) {
      for (const [ch, ref] of Object.entries(ten[key] || {})) root.append(el('p', { text: `${label}通道${ch}：保留片段去重覆盖 ${fmtNum(ref.usable_seconds, 1)}秒，窗数${ref.n}（非独立样本），中位数${fmtNum(ref.median, 4)}，MAD ${fmtNum(ref.mad, 4)}` }));
    }
    const modes = { eeg: '探索性脑电反馈', quality: '质量反馈', guided: '固定引导' };
    for (const seg of ten.segments || []) {
      root.append(el('h3', { text: `训练第${seg.index}轮（另含10秒休息）` }));
      if (seg.stats?.timing_spec) root.append(el('p', { text:
        `有效时间覆盖 ${fmtPercent(seg.stats.valid_time_ratio)}；整段达标 ${fmtPercent(seg.stats.on_target_ratio)}；有效时间内达标 ${fmtPercent(seg.stats.on_target_valid_ratio)}。${seg.adaptation_reason || ''}` }));
      root.append(bar('未反馈（采样/计算）秒', seg.unobserved_seconds ?? null, 50));
      for (const [mode, seconds] of Object.entries(seg.mode_seconds || {})) root.append(bar(`${modes[mode] || mode} 秒`, seconds, 50));
    }
  }
  const timeline = el('div', { class: 'diagnostic-timeline', role: 'img', 'aria-label': '窗口质量时间轴，绿色合格，橙色排除' });
  for (const row of d?.windows || []) {
    const reasons = (row.reasons || row.unusable_reasons || []).map(r => labels[r] || r).join('、');
    timeline.append(el('span', { class: row.usable ? 'diagnostic-window good' : 'diagnostic-window', title: `${row.phase || ''} · 等级${row.graded?.grade || '未分级'} · 窗${row.index ?? ''} · ${row.usable ? '合格' : reasons}` }));
  }
  if (timeline.childNodes.length) root.append(el('p', { text: '逐窗质量：绿色合格，橙色排除；悬停查看原因。' }), timeline);
  for (const [reason, count] of Object.entries(d?.reason_counts || quality.unusable_reasons || {})) root.append(bar(labels[reason] || reason, count, total, true));
  root.append(el('p', { class: 'muted', text: '原因可重叠。现行门槛：幅值150µV、通道差100µV、低频占比55%、高频占比35%；均待验证。低频偏高不等于已证实眼电污染。' }));
  if (d?.amplitude_sensitivity?.length) {
    root.append(el('h3', { text: '幅值阈值敏感性（其他条件不变）' }), el('p', { text: '仅解释放宽幅值门槛的影响，不更改严格判定，也不证明更宽阈值有效。' }));
    for (const item of d.amplitude_sensitivity) root.append(bar(`≤${item.amp_max_uv}µV的合格窗`, item.accepted, total));
  }
  for (const [key, label] of [['baseline_eyes_open', '睁眼'], ['baseline_eyes_closed', '闭眼']]) {
    const b = report.extras?.[key];
    if (b) root.append(el('p', { text: `${label}基线：${b.valid ? '合格' : '不合格'} · 有效覆盖 ${fmtNum(b.quality?.usable_seconds, 1)} 秒 / 要求30秒 · 有效比例 ${fmtPercent(b.quality?.valid_ratio, 1)} / 要求60%。` }));
  }
  for (const [group, label] of [['accepted', '合格窗频带组成'], ['rejected', '排除窗频带组成（仅用于污染诊断）']]) {
    const bands = d?.bands?.[group];
    if (!bands) continue;
    root.append(el('h3', { text: label }));
    let n = 0;
    for (const [name, stat] of Object.entries(bands)) {
      if (stat.mean === null) continue;
      root.append(bar(`${name} %（n=${stat.n}）`, stat.mean * 100, 100, group === 'rejected')); n++;
    }
    if (!n) root.append(el('p', { text: '没有符合此组条件的频带数据。' }));
  }
  const pvt = report.behavior?.pvt;
  if (pvt && Object.keys(pvt).length) {
    root.append(el('h3', { text: 'PVT 行为分析（不依赖脑电基线）' }));
    for (const [label, key] of [['有效反应', 'responded'], ['抢答（排除RT）', 'false_starts'], ['漏答', 'missed']]) {
      if (Number.isFinite(pvt[key]) && pvt.trials > 0) root.append(bar(label, pvt[key], pvt.trials));
    }
    const lapseDenominator = pvt.rate_denominators?.lapse_rate ?? pvt.responded;
    if (Number.isFinite(pvt.lapses) && lapseDenominator > 0) root.append(bar('慢反应 >500ms', pvt.lapses, lapseDenominator));
    const hasResponses = lapseDenominator !== 0;
    const median = hasResponses && Number.isFinite(pvt.rt_median) ? pvt.rt_median * 1000 : null;
    const mean = hasResponses && Number.isFinite(pvt.rt_mean) ? pvt.rt_mean * 1000 : null;
    const lapseBasis = !Number.isFinite(lapseDenominator) ? '慢反应率分母未记录。'
      : lapseDenominator > 0 ? `慢反应率按 ${lapseDenominator} 个有效反应计算。` : '有效反应为 0，慢反应率不可计算。';
    root.append(el('p', { text: `中位反应时 ${fmtNum(median, 0)} ms；平均 ${fmtNum(mean, 0)} ms。${lapseBasis}阈值待验证。` }));
  }
  return card('采集质量与行为可视化', root);
}
