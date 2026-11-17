/** Existing session evidence, reports and trends shown without synthetic fill-ins. */
import { card, el, empty, fmtInt, fmtNum, fmtPercent, list, pick, table } from '../util.js';
import { drawHeatmap, drawLineChart } from '../charts.js';
import { metric } from './showcase-analysis.js';

const sourceNames = { eeg: '脑电', scales: '心理量表', behavior: '行为任务' };
const indicatorNames = { focus: '专注相关指数', relax: '放松指数', load: '认知负荷指数' };

export function sessionSource(session, report) {
  const sim = report?.extras?.source_kind === 'sim'
    || String(session?.device || '').startsWith('sim');
  return sim ? '模拟会话 · 软件流程演示' : '已有会话 · 按记录来源解释';
}

export function sessionLinks(api, uuid, ready = false) {
  if (!uuid) return null;
  return el('div', { class: 'showcase-actions' }, [
    el('a', { class: 'btn', href: `#/report?session=${encodeURIComponent(uuid)}`, text: '打开完整报告' }),
    ready ? el('a', { class: 'btn', href: api.artifactUrl(uuid, 'report_html'), target: '_blank', rel: 'noopener noreferrer', text: '离线 HTML 报告' }) : null,
    ready ? el('a', { class: 'btn', href: api.exportZipUrl(uuid), download: '', text: '导出会话 ZIP' }) : null,
  ]);
}

export function renderJointResults(report, session, api) {
  if (!report) return empty('请先选择已有会话，或运行完整模拟流程。完成后会显示真实保存的量表、行为、脑电和报告内容。');
  const summary = report.indicators?.summary || {};
  const scales = report.scales || {};
  const behavior = report.behavior || {};
  const assessment = report.assessment || {};
  const evidence = list(assessment.evidence);
  const source = sessionSource(session, report);
  const panel = el('div');
  panel.append(el('p', { class: 'showcase-source', text: source }));
  panel.append(el('div', { class: 'showcase-evidence-grid' }, [
    card('脑电证据', [el('p', { class: 'showcase-caption', text: '相对个人基线的指数' }),
      ...Object.entries(indicatorNames).map(([key, label]) => el('div', { class: 'showcase-value-row' }, [
        el('span', { text: label }), el('strong', { text: fmtNum(summary[key]?.mean, 3) }),
      ])),
      el('p', { class: 'showcase-caption', text: `合格窗 ${fmtInt(report.quality?.windows_usable)} / ${fmtInt(report.quality?.windows_total)}` }),
    ]),
    card('量表证据', Object.keys(scales).length ? Object.entries(scales).map(([key, value]) => el('div', {
      class: 'showcase-scale-result',
    }, [el('strong', { text: value.name || key }),
      el('span', { text: `标准分 ${fmtNum(value.standard_score, 0)} · ${value.level || '按量表记录解释'}` }),
    ])) : empty('本协议没有保存量表结果。')),
    card('行为证据', [
      el('div', { class: 'showcase-value-row' }, [el('span', { text: 'SART 虚报率' }), el('strong', { text: fmtPercent(behavior.sart?.commission_rate, 1) })]),
      el('div', { class: 'showcase-value-row' }, [el('span', { text: 'SART 漏报率' }), el('strong', { text: fmtPercent(behavior.sart?.omission_rate, 1) })]),
      el('div', { class: 'showcase-value-row' }, [el('span', { text: 'PVT 中位反应时' }), el('strong', { text: `${fmtNum(behavior.pvt?.rt_median, 3)} s` })]),
      el('p', { class: 'showcase-caption', text: '正式成绩与练习分开，未完成或无效结果不作为评估证据。' }),
    ]),
  ]));
  panel.append(card('联合评估与结构化报告', [
    el('div', { class: 'showcase-conclusion' }, [el('span', { text: '评估结论' }),
      el('h3', { text: assessment.conclusion || '报告尚未形成联合结论' })]),
    el('p', { class: 'showcase-caption', text: '当前采用可解释规则整合三类证据，保留缺失与冲突，不把相对指数解释为个人能力或诊断。' }),
    evidence.length ? table([
      { title: '来源', render: row => sourceNames[row.source] || row.source || '—' },
      { title: '评估依据', render: row => row.summary || '—' },
      { title: '可用', render: row => row.available === false ? '不可用' : '已记录' },
    ], evidence) : empty('报告尚无可展示的证据条目。'),
    el('h4', { text: '改善建议' }),
    list(assessment.advice).length
      ? el('ul', { class: 'showcase-advice' }, assessment.advice.map(item => el('li', { text: item })))
      : empty('暂无建议，等待会话完成或补充有效证据。'),
    sessionLinks(api, session?.uuid, session?.status === 'done' && report.partial !== true),
  ]));
  return panel;
}

export function renderSavedReportPreview(report, session) {
  if (!report) return empty('报告生成后，这里展示保存的评分、问题分析与改善建议。');
  const summary = report.indicators?.summary || {};
  return el('div', { class: 'showcase-report-preview' }, [
    el('span', { class: 'showcase-eyebrow', text: 'NINGSI · 状态评估报告' }),
    el('h3', { text: `参与者 ${report.participant || session?.participant || '—'}` }),
    el('p', { class: 'showcase-caption', text: sessionSource(session, report) }),
    el('div', { class: 'showcase-metrics showcase-metrics--compact' },
      Object.entries(indicatorNames).map(([key, label]) => metric(label, fmtNum(summary[key]?.mean, 3)))),
    el('strong', { text: report.assessment?.conclusion || '证据不足，等待完成评估' }),
    el('p', { class: 'showcase-caption', text: '评分 · 质量 · 量表 · 行为 · 证据 · 建议均可导出。' }),
  ]);
}

export function renderHeatmapData(heatmap) {
  const chart = el('div', { class: 'showcase-heatmap' });
  if (!heatmap || !list(heatmap.cells).length) return empty('暂无已保存的会话状态热力图。');
  drawHeatmap(chart, heatmap.cells, { columns: heatmap.columns || 30,
    missingColor: heatmap.missing_color || '#d5d8dc', label: '已保存会话状态热力图' });
  return el('div', {}, [chart, el('p', { class: 'showcase-caption', text: heatmap.note
    || '缺失窗单独显示，不进入低专注计时。颜色表达相对指数分档。' })]);
}

export function renderTrendData(trend, field, period) {
  const chart = el('div', { class: 'showcase-plot' });
  const points = list(trend?.points).map(point => ({
    label: String(point.period ?? point.label ?? point.bucket ?? point.t ?? '—'),
    values: { value: point.mean ?? point.value ?? point.focus ?? point.avg ?? null },
  }));
  drawLineChart(chart, points, { series: ['value'], colors: ['#167c91'], height: 220,
    min: 0, max: field === 'stress' ? 10 : 1, points: true, percent: field !== 'stress',
    showValues: true, label: `${field === 'stress' ? '压力自评' : '专注相关指数'}${period === 'month' ? '月' : '周'}趋势`,
    emptyText: field === 'stress' ? '暂无已完成现场会话的压力自评' : '暂无可比较的已完成会话' });
  return el('div', {}, [
    el('p', { class: 'showcase-source', text: field === 'stress' ? '数据来源：本人压力自评 · 0–10 分' : '数据来源：已完成会话的相对脑电指数' }),
    chart,
    el('p', { class: 'showcase-caption', text: trend?.note || '历史只显示实际保存的记录，缺少长期资料时保持空态。' }),
    el('p', { class: 'showcase-caption', text: `参与聚合 ${fmtInt(trend?.sessions)} 次会话；按${period === 'month' ? '月' : '周'}汇总。` }),
  ]);
}

export function renderModelResult(result) {
  if (!result) return empty('点击“运行模拟训练”后，由后端现算并展示训练、验证、测试结果。');
  return el('div', {}, [
    el('p', { class: 'showcase-source', text: '模拟被试数据 · 软件训练链路自检 · 非真人准确率' }),
    el('div', { class: 'showcase-metrics showcase-metrics--compact' }, [
      metric('可解释分类器', '逻辑回归', '传统机器学习'),
      metric('仿真被试', fmtInt(list(result.participants).length)),
      metric('训练样本', fmtInt(result.samples)),
    ]),
    table([
      { title: '按人隔离的划分', render: row => row.label },
      { title: '样本数', render: row => fmtInt(row.metrics?.n), align: 'right' },
      { title: '准确率', render: row => fmtPercent(row.metrics?.accuracy, 1), align: 'right' },
      { title: 'AUROC', render: row => fmtNum(row.metrics?.auc, 3), align: 'right' },
      { title: '参与者', render: row => list(result.subject_split?.[row.key]).join('、') || '—' },
    ], [['train', '训练'], ['validation', '验证'], ['test', '测试']].map(([key, label]) => ({key, label, metrics: result[key]}))),
    el('p', { class: 'showcase-caption', text: '真人研究使用独立采集导出、开发组交叉验证和封存测试，模型核对设备、单位、时间策略与处理版本后启用。' }),
  ]);
}
