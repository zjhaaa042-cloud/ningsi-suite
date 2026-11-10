import { card, el, fmtNum, fmtPercent } from '../util.js';

const labels = { amplitude: '幅值超限', eog: '低频占比偏高（眼电/漂移代理）', emg: '高频占比偏高（肌电代理）', channel_span: '通道差异过大', flat_channel: '平直通道', signal_too_short: '未满4秒', no_data: '无样本', non_finite_signal: '非有限值' };

function bar(label, value, max, rejected = false) {
  const number = Number.isFinite(value) ? value : 0;
  return el('div', { class: 'diagnostic-bar' }, [
    el('span', { text: label }),
    el('progress', { max: Math.max(1, max), value: Math.max(0, number), class: rejected ? 'diagnostic-rejected' : '', 'aria-label': label }),
    el('strong', { text: fmtNum(value, 1) }),
  ]);
}

export function renderDiagnostics(report) {
  const d = report.extras?.signal_diagnostics;
  const quality = report.quality || {};
  const total = d?.total ?? quality.windows_total ?? 0;
  const accepted = d?.accepted ?? quality.windows_usable ?? 0;
  const root = el('div', { class: 'stack' });
  if (d?.scope) root.append(el('p', { text: d.scope }));
  root.append(el('p', { text: `严格合格 ${accepted}/${total} 窗。${d ? d.note : '旧版本未保存完整逐窗信号；可查看已有质量与行为汇总，不能事后重建频谱。'}` }));
  root.append(bar('严格合格窗', accepted, total), bar('排除窗', total - accepted, total, true));
  const timeline = el('div', { class: 'diagnostic-timeline', role: 'img', 'aria-label': '窗口质量时间轴，绿色合格，橙色排除' });
  for (const row of d?.windows || []) {
    const reasons = (row.reasons || row.unusable_reasons || []).map(r => labels[r] || r).join('、');
    timeline.append(el('span', { class: row.usable ? 'diagnostic-window good' : 'diagnostic-window', title: `${row.phase || ''} · 窗${row.index ?? ''} · ${row.usable ? '合格' : reasons}` }));
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
  if (pvt) {
    root.append(el('h3', { text: 'PVT 行为分析（不依赖脑电基线）' }));
    for (const [label, key] of [['有效反应', 'responded'], ['抢答（排除RT）', 'false_starts'], ['漏答', 'missed'], ['慢反应 >500ms', 'lapses']]) root.append(bar(label, pvt[key], pvt.trials));
    root.append(el('p', { text: `中位反应时 ${fmtNum(Number.isFinite(pvt.rt_median) ? pvt.rt_median * 1000 : null, 0)} ms；平均 ${fmtNum(Number.isFinite(pvt.rt_mean) ? pvt.rt_mean * 1000 : null, 0)} ms。慢反应率分母为${pvt.responded ?? '未知'}个有效反应；阈值待验证。` }));
  }
  return card('采集质量与行为可视化', root);
}
