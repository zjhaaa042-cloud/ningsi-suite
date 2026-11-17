import { DASH, el, fmtDuration, fmtPercent } from './util.js';

const finite = value => typeof value === 'number' && Number.isFinite(value) ? value : null;

/** Keep performance and availability separate; historic sample ratios are never relabelled. */
export function trainingMetrics(stats = {}, duration = null) {
  const modern = Object.hasOwn(stats, 'on_target_valid_ratio') && Object.hasOwn(stats, 'valid_time_ratio');
  const empty = stats.n === 0 || stats.valid_seconds === 0;
  const coverage = modern ? finite(stats.valid_time_ratio) : null;
  return {
    modern,
    validRatio: modern && !empty ? finite(stats.on_target_valid_ratio) : null,
    wallRatio: !empty ? finite(stats.on_target_ratio) : null,
    coverage,
    validSeconds: modern ? finite(stats.valid_seconds) : null,
    duration: finite(duration) ?? finite(stats.duration_sec),
    sufficient: modern && (typeof stats.coverage_sufficient === 'boolean'
      ? stats.coverage_sufficient : coverage === null ? null : coverage >= .6),
  };
}

export function trainingCoverageNote(metrics) {
  if (!metrics.modern) return '旧版记录缺少有效时间字段；样本达标占比保留原口径，不能与有效时间内达标率直接比较。';
  if (metrics.validSeconds === 0 || metrics.coverage === 0) return '暂无有效评分；缺失不记零分，请先改善信号质量。';
  if (metrics.sufficient === false) return '有效覆盖不足，样本不足；有效时间内达标率仅描述已有数据，整体训练表现暂不判定，目标保持。';
  return '有效时间内达标率反映已有有效反馈的表现；请结合覆盖率与实际时长阅读。';
}

/** A single server-owned timer renders identically for SSE and saved results. */
export function effectiveTimeState(snapshot, phase = null) {
  if (!snapshot || typeof snapshot !== 'object') return null;
  if (snapshot.effective_time) {
    if (phase && (snapshot.key || snapshot.phase) !== phase) return null;
    snapshot = snapshot.effective_time;
  }
  return finite(snapshot.target_seconds) !== null ? snapshot : null;
}

export function effectiveTimeOutcome(progress) {
  if (!progress) return DASH;
  if (progress.completed) return '有效目标已完成';
  return ({ cancelled: '已取消', baseline_invalid: '基线未通过 · 固定引导',
    timing_issue: '计时不一致 · 未记为完成' })[progress.completion_reason] || '数据不足';
}

export function effectiveTimeContent(progress, { training = true, aggregate = false } = {}) {
  const p = progress || {};
  const valid = finite(p.valid_seconds);
  const target = finite(p.target_seconds);
  const reason = p.completion_reason;
  const insufficient = ['max_duration', 'baseline_invalid', 'timing_issue'].includes(reason) || (reason === 'target_reached' && p.completed === false) || (training && p.baseline_valid === false);
  const accelerated = finite(p.time_scale) !== null && p.time_scale !== 1;
  const status = reason === 'cancelled' ? '采集已取消，保留已累计的有效时间。'
    : reason === 'timing_issue' ? '计时记录不一致，本次未记为完成；请查看采集记录。'
    : (reason === 'baseline_invalid' || p.baseline_valid === false && training) ? '基线未通过：已切换固定引导，本轮不累计有效反馈；数据不足。'
    : insufficient ? '已达到补采上限或基线门槛未通过，数据不足。'
    : reason === 'target_reached' && p.completed !== false ? '有效时间目标已完成。'
    : p.quality_paused ? (training
      ? '质量暂停：当前窗不计分，累计有效时间保留；质量反馈或固定引导继续，连续保持重新开始。'
      : '质量暂停：当前不合格区间不计入有效基线时间，已累计时间保留。')
      : '正在累计有效时间；达到目标或时长上限后自动继续。';
  const metric = (label, value) => el('div', {}, [el('span', { class: 'muted', text: label }), el('strong', { class: 'mono', text: value })]);
  return el('div', { class: 'effective-time', dataset: { state: insufficient ? 'insufficient' : reason || (p.quality_paused ? 'paused' : 'running') } }, [
    el('div', { class: 'effective-time__metrics' }, [
      metric(training ? aggregate ? '训练有效反馈 / 目标' : '本轮有效反馈 / 目标' : '有效基线 / 目标', `${fmtDuration(valid)} / ${fmtDuration(target)}`),
      metric(accelerated ? '协议用时 / 补采上限' : '实际用时 / 补采上限', `${fmtDuration(p.elapsed_seconds)} / ${fmtDuration(p.max_seconds)}`),
      metric('延长用时', fmtDuration(p.extension_seconds)),
      accelerated ? metric('实际墙钟用时', fmtDuration(p.actual_elapsed_seconds)) : null,
    ]),
    el('progress', { max: 1, value: target > 0 && valid !== null ? Math.min(1, Math.max(0, valid / target)) : 0, 'aria-label': '有效时间完成进度' }),
    el('p', { class: 'effective-time__status', role: 'status', text: status }),
    el('p', { class: 'muted', text: training ? '只累计实际可用反馈秒数，坏窗不填分；累计进度与连续保持分别计算。' : '重叠分析窗按合格时间区间并集累计，基线仍须通过原有质量门槛。' }),
    accelerated ? el('p', { class: 'muted', text: '加速模拟：有效时间、目标与上限按协议秒显示，实际墙钟用时单独记录。' }) : null,
  ]);
}

export function formatWallRatio(metrics) {
  return metrics.wallRatio === null ? DASH : `${fmtPercent(metrics.wallRatio)}${metrics.modern ? '（实际时间）' : '（旧版有效样本）'}`;
}
