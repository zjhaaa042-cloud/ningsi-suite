/** Charts and import validation for the competition's real analysis workbench. */
import { el, empty, fmtInt, fmtNum, fmtPercent, list, table } from '../util.js';
import { drawLineChart, svgEl } from '../charts.js';

const REASONS = {
  no_data: '没有信号', signal_too_short: '窗口样本不足', non_finite_signal: '信号含无效值',
  unit_unconfirmed: '电压换算未确认', clock_unsynchronized: '时标尚未确认',
  nonmonotonic_timestamps: '时间戳倒序', timestamp_gap_over_100ms: '时间间隔过大',
  sample_rate_mismatch_over_5pct: '实际采样率与声明不符', preprocessing_warmup: '滤波历史不足',
  sample_gap_in_filter_history: '滤波历史内缺样', baseline_invalid: '独立基线不可用',
  window_unusable: '当前窗未通过质控', baseline_configuration_mismatch: '基线配置不匹配',
  amplitude: '幅度超限', flat_channel: '通道平线', channel_span: '通道间幅差超限',
  emg: '高频污染线索', eog: '低频污染线索', ocular_transient: '眼动样瞬态',
  muscle_burst: '肌电样爆发', motion_transient: '运动样突变', contact_dropout: '接触或信号中断',
  motion_reference_excursion: '运动参考超限', emg_reference_burst: '肌电参考爆发',
  no_usable_window: '基线没有合格窗', insufficient_windows: '合格窗数不足',
  insufficient_usable_seconds: '有效覆盖时间不足', low_valid_ratio: '合格比例不足',
};

export const reasonText = value => REASONS[value] || String(value || '未提供原因');
export const finite = value => typeof value === 'number' && Number.isFinite(value);

export async function readRawFile(file) {
  if (!file) throw new Error('请先选择原始 EEG JSON 文件。');
  if (file.size > 1024 * 1024) throw new Error('JSON 超过 1 MiB，请截取较短片段。');
  let data;
  try { data = JSON.parse(await file.text()); }
  catch (_) { throw new Error('文件不是有效的 JSON。'); }
  if (!data || !Array.isArray(data.samples) || !data.samples.length
      || !data.samples.every(row => Array.isArray(row))) {
    throw new Error('需要 samples 二维矩阵，按“通道 × 采样点”排列。');
  }
  if (!finite(data.rate) || !finite(data.scale_to_uv) || data.scale_to_uv <= 0) {
    throw new Error('必须提供数值 rate 和已确认的正数 scale_to_uv。');
  }
  // Only the selected file's explicit signal fields enter analysis. No local
  // filesystem paths are uploaded and a filename never proves human origin.
  const body = { mode: 'raw', samples: data.samples, rate: data.rate, scale_to_uv: data.scale_to_uv };
  for (const name of ['timestamps', 'channel_labels', 'baseline_samples', 'baseline_timestamps']) {
    if (data[name] !== undefined) body[name] = data[name];
  }
  return body;
}

export function metric(label, value, note = '') {
  return el('div', { class: 'showcase-metric' }, [
    el('span', { class: 'showcase-metric__label', text: label }),
    el('strong', { text: value }),
    note ? el('span', { class: 'showcase-metric__note', text: note }) : null,
  ]);
}

function wave(values, times, intervals, label, color, showMasks) {
  const good = list(values).filter(finite);
  const root = svgEl('svg', { viewBox: '0 0 900 132', role: 'img', 'aria-label': label,
    class: 'showcase-wave', preserveAspectRatio: 'none' });
  if (!good.length) {
    root.append(svgEl('text', { x: 450, y: 65, 'text-anchor': 'middle', class: 'showcase-svg-label' },
      [document.createTextNode('该窗口不可处理，缺失值保持留空')]));
    return root;
  }
  const max = Math.max(...good.map(Math.abs), 1);
  const duration = Math.max(...list(times).filter(finite), 4);
  const x = t => 64 + 808 * t / duration;
  const y = v => 58 - 43 * v / max;
  for (const fraction of [-1, 0, 1]) {
    root.append(svgEl('line', { x1: 64, x2: 872, y1: y(max * fraction), y2: y(max * fraction),
      class: 'showcase-wave__grid' }));
    root.append(svgEl('text', { x: 56, y: y(max * fraction) + 4, 'text-anchor': 'end', class: 'showcase-svg-label' },
      [document.createTextNode(fmtNum(max * fraction, 1))]));
  }
  if (showMasks) for (const span of list(intervals)) {
    if (!finite(span.start_sec) || !finite(span.end_sec)) continue;
    const left = Math.max(0, Math.min(duration, span.start_sec));
    const right = Math.max(left, Math.min(duration, span.end_sec));
    root.append(svgEl('rect', { x: x(left), y: 10, width: x(right) - x(left), height: 96,
      class: 'showcase-wave__mask' }));
  }
  let path = '', pen = false;
  list(values).forEach((value, index) => {
    if (!finite(value)) { pen = false; return; }
    const time = finite(times?.[index]) ? times[index] : duration * index / Math.max(values.length - 1, 1);
    path += `${pen ? 'L' : 'M'}${x(time).toFixed(2)},${y(value).toFixed(2)} `;
    pen = true;
  });
  root.append(svgEl('path', { d: path.trim(), fill: 'none', stroke: color, 'stroke-width': 1.35,
    'vector-effect': 'non-scaling-stroke' }));
  for (let i = 0; i <= 4; i++) root.append(svgEl('text', { x: x(duration * i / 4), y: 125,
    'text-anchor': 'middle', class: 'showcase-svg-label' }, [document.createTextNode(`${fmtNum(duration * i / 4, 1)}s`)]));
  return root;
}

export function analysisSummary(data) {
  const q = data?.quality || {}, timing = data?.timing || {};
  return el('div', { class: 'showcase-metrics' }, [
    metric('分析窗口', `${fmtInt(q.windows_usable)} / ${fmtInt(q.windows_total)}`, '通过当前质量门控'),
    metric('有效窗比例', fmtPercent(q.valid_ratio, 1), '不代表识别准确率'),
    metric('有效覆盖', `${fmtNum(q.usable_seconds, 1)} s`, '按重叠窗口并集计时'),
    metric('本次计算', `${fmtNum(timing.compute_ms, 1)} ms`, '输入跨度与更新周期另计'),
  ]);
}

export function renderWaveforms(data, channelIndex = 0) {
  const traces = data?.traces || {};
  const channel = list(traces.channels)[channelIndex];
  if (!channel) return empty('暂无完整窗口可供绘图。');
  return el('div', { class: 'showcase-wave-stack' }, [
    ...[
      ['原始 EEG', 'raw_uv', '#54708a', false],
      ['滤波后', 'filtered_uv', '#167c91', true],
      ['污染片段屏蔽后', 'cleaned_uv', '#275cb1', true],
    ].map(([label, key, color, masked]) => el('div', { class: 'showcase-wave-row' }, [
      el('div', { class: 'showcase-wave-row__title' }, [
        el('strong', { text: label }), el('span', { text: `${channel.label || '通道'} · µV` }),
      ]),
      wave(channel[key], traces.time_sec, traces.masked_intervals, `${label}，${channel.label || '通道'}`, color, masked),
    ])),
    el('p', { class: 'showcase-caption', text: '各图使用独立纵轴，幅度单位均为 µV。阴影为疑似污染时段；留空表示排除，不用插值补造脑电。' }),
  ]);
}

export function renderSpectrum(data) {
  const chart = el('div', { class: 'showcase-plot' });
  const spectrum = data?.spectrum || {};
  const points = list(spectrum.freqs_hz).map((hz, i) => ({
    label: `${fmtNum(hz, 0)} Hz`, values: {
      raw: finite(spectrum.raw_psd_uv2_hz?.[i]) && spectrum.raw_psd_uv2_hz[i] > 0
        ? 10 * Math.log10(spectrum.raw_psd_uv2_hz[i]) : null,
      filtered: finite(spectrum.filtered_psd_uv2_hz?.[i]) && spectrum.filtered_psd_uv2_hz[i] > 0
        ? 10 * Math.log10(spectrum.filtered_psd_uv2_hz[i]) : null,
    },
  }));
  drawLineChart(chart, points, { series: ['raw', 'filtered'], colors: ['#92a2b4', '#167c91'],
    points: false, height: 230, label: '滤波前后功率谱密度，分贝微伏平方每赫兹', emptyText: '尚无可计算频谱' });
  return el('div', {}, [el('div', { class: 'showcase-legend' }, [
    el('span', { class: 'showcase-legend__raw', text: '原始 PSD' }),
    el('span', { class: 'showcase-legend__filtered', text: '滤波后 PSD' }),
    el('span', { text: '纵轴 dB(µV²/Hz)' }),
  ]), chart]);
}

export function renderBandFeatures(data) {
  const names = { theta: 'θ', alpha: 'α', beta: 'β', low: '低频', gamma: 'γ' };
  const bands = list(data?.bands);
  const rows = bands.filter(row => ['theta', 'alpha', 'beta'].includes(row.name));
  const cards = el('div', { class: 'showcase-band-grid' }, rows.map(row => el('div', {
    class: `showcase-band showcase-band--${row.name}`,
  }, [
    el('div', { class: 'showcase-band__head' }, [
      el('strong', { text: names[row.name] }), el('span', { text: `${row.low_hz}–${row.high_hz} Hz` }),
    ]),
    el('strong', { class: 'showcase-band__value', text: fmtPercent(row.relative_power, 1) }),
    el('span', { class: 'showcase-caption', text: '滤波后相对功率' }),
    el('div', { class: 'showcase-powerbar', role: 'img', 'aria-label': `${row.name} 相对功率 ${fmtPercent(row.relative_power, 1)}` }, [
      el('span', { style: `width:${finite(row.relative_power) ? Math.min(100, Math.max(0, row.relative_power * 100)) : 0}%` }),
    ]),
  ])));
  const values = data?.latest?.time_domain || {};
  return el('div', {}, [cards, table([
    { title: '频带', render: r => `${names[r.name] || r.name} · ${r.low_hz}–${r.high_hz} Hz` },
    { title: '原始功率 µV²', render: r => fmtNum(r.raw_power_uv2, 3), align: 'right' },
    { title: '滤波后功率 µV²', render: r => fmtNum(r.filtered_power_uv2, 3), align: 'right' },
    { title: '相对功率', render: r => fmtPercent(r.relative_power, 1), align: 'right' },
  ], rows), el('div', { class: 'showcase-metrics showcase-metrics--compact' }, [
    metric('RMS', `${fmtNum(values.rms_uv, 2)} µV`),
    metric('峰峰值', `${fmtNum(values.ptp_uv, 2)} µV`),
    metric('最小通道相关', fmtNum(values.channel_corr_min, 3)),
  ]), el('p', { class: 'showcase-caption', text: data?.latest?.usable
    ? '本窗口通过当前质量规则。频带变化提供可解释特征，不能单独证明专注或情绪状态。'
    : '本窗口未通过质量门控，表内频谱仅用于诊断，不作为状态识别证据。' })]);
}

export function renderQuality(data) {
  const latest = data?.latest || {};
  const entries = Object.entries(data?.quality?.reason_counts || {});
  return el('div', {}, [
    el('div', { class: `showcase-quality ${latest.usable ? 'showcase-quality--usable' : ''}` }, [
      el('span', { class: 'showcase-dot' }),
      el('strong', { text: latest.usable ? '当前窗通过质量门控' : '当前窗暂停状态估计' }),
    ]),
    el('p', { text: latest.usable ? '继续检查独立基线与模型适用条件。' : list(latest.reasons).map(reasonText).join('、') || '数据不足，等待处理结果。' }),
    entries.length ? table([
      { title: '整段拒评原因', render: ([key]) => reasonText(key) },
      { title: '涉及窗数', render: ([, n]) => fmtInt(n), align: 'right' },
    ], entries) : el('p', { class: 'showcase-caption', text: '本段未触发当前质控规则；不等于所有伪迹已被排除。' }),
    el('div', { class: 'showcase-baseline' }, [
      el('strong', { text: data?.baseline?.valid ? '独立基线可用' : '独立基线不可用' }),
      el('p', { text: data?.source?.simulated
        ? '模拟数据采用单独生成的模拟基线，仅用于演示计算。'
        : data?.baseline?.valid ? '使用此次输入中单独提供并通过检查的基线。' : '导入数据未提供合格独立基线时，指数保持缺失。' }),
    ]),
  ]);
}
