/**
 * 工作概览：会话记录、最近完成结果、个人周趋势。
 *
 * 数据来源（严格按 docs/API.md）：
 * - GET /api/health、GET /api/overview
 * - GET /api/sessions/{uuid}（取最近会话的 indicator_summary、alert_count、runs 里的质检结果）
 * - GET /api/reports/trend?participant=&field=&period=
 */

import { api, historyArtifactUrl } from '../api.js';
import { sessionPresentationUrl, sessionSourceLabel } from '../session-history.js';
import { describeError, toast } from '../store.js';
import {
  DASH, button, card, el, empty, fmtInt, fmtNum, fmtPercent, fmtTime,
  list, phaseText, pick, statusText, table,
} from '../util.js';
import { SERIES_COLORS, drawGauge, drawLineChart } from '../charts.js';

/**
 * 从 runs 里取质检事件（phase=qc 的 payload 由后端 _decode_payload 落成对象）。
 * 保险起见仍兼容"payload 是 JSON 字符串"的旧数据，避免整卡退回"尚无质检记录"。
 */
function qualityFromRuns(runs) {
  for (const run of list(runs)) {
    let payload = run && run.payload;
    if (typeof payload === 'string') {
      try {
        payload = JSON.parse(payload);
      } catch (error) {
        payload = null;
      }
    }
    if (payload && typeof payload === 'object' && (payload.passed !== undefined || payload.windows !== undefined)) {
      return payload;
    }
  }
  return null;
}

/** 趋势点字段兼容：后端聚合结构可能是 {period, mean, n} 或 {label, value}。 */
function normalizeTrendPoint(point) {
  if (!point || typeof point !== 'object') return null;
  const label = point.period ?? point.label ?? point.bucket ?? point.t ?? point.key;
  const value = point.mean ?? point.value ?? point.focus ?? point.avg;
  const n = Number(point.n ?? point.count);
  const rejected = Number(point.rejected);
  return {
    label: label === undefined || label === null ? DASH : String(label),
    value: typeof value === 'number' && Number.isFinite(value) ? value : null,
    n: Number.isFinite(n) ? n : null,
    rejected: Number.isFinite(rejected) ? rejected : null,
  };
}

function renderKpi(label, value, hint) {
  return el('div', { class: 'overview-stat' }, [
    el('span', { class: 'overview-stat__label', text: label }),
    el('strong', { class: 'overview-stat__value', text: value }),
    hint ? el('span', { class: 'overview-stat__hint', text: hint }) : null,
  ]);
}

function disclosure(label, children, className = '') {
  return el('details', { class: `overview-details ${className}`.trim() }, [
    el('summary', { text: label }), el('div', { class: 'overview-details__body' }, children),
  ]);
}

export async function render(container, ctx) {
  container.textContent = '';
  container.append(el('p', { class: 'muted', text: '正在加载仪表盘…' }));
  let health;
  let overview;
  let configResponse = null;
  let recent = [];
  let historyTotal = null;
  let historyError = null;
  try {
    // config 与仪表盘同时取：质检门槛（config.quality.valid_ratio_min）必须来自接口，
    // 不能写死；config 失败不拖垮整个仪表盘
    [health, overview, configResponse] = await Promise.all([
      api.health(), api.overview(), api.config().catch(() => null),
    ]);
  } catch (error) {
    container.textContent = '';
    container.append(card('仪表盘', [empty(`加载失败：${describeError(error)}`)]));
    toast(describeError(error));
    return;
  }
  if (ctx.signal.aborted) return;
  if (configResponse) ctx.store.setState({ config: configResponse.data });

  try {
    const response = await api.listTestHistory({ limit: 6, page: 1 });
    recent = list(pick(response.data, 'items', []));
    historyTotal = response.data.total;
  } catch (error) {
    historyError = describeError(error);
  }
  if (ctx.signal.aborted) return;

  ctx.store.setState({ health: health.data, overview: overview.data });
  const stats = overview.data || {};
  const latest = list(stats.latest_detail);
  const newest = latest[0] || null;
  // KPI / 质检 / 预警的口径必须取"最近一条【已完成】会话"：latest[0] 可能正在运行，
  // 运行中的会话还没有 indicator_summary、qc 也未必落库，整排 KPI 会一起变成 —
  // （Lead 实测：最新会话 fe1817 status=running 时仪表盘 KPI 整排空）
  const newestDone = latest.find((row) => row.status === 'done') || null;
  const metricRow = newestDone || newest;

  // 最近会话的细节：KPI 里的均值/预警数都来自这里，避免用列表接口猜数据
  let detail = null;
  if (metricRow && metricRow.uuid) {
    try {
      const response = await api.getSession(metricRow.uuid);
      detail = response.data;
    } catch (error) {
      detail = null;
    }
  }
  if (ctx.signal.aborted) return;

  const summary = pick(detail, 'indicator_summary', {}) || {};
  const quality = qualityFromRuns(pick(detail, 'runs', []));
  // 预警数口径：明细存在（哪怕是 0 条）就用明细条数；只有"没有明细"才回落到列表的 alert_count。
  // 原来的 `list(...).length || alert_count` 会把"明细 0 条"串到另一个口径上。
  const alertItems = detail ? list(pick(detail, 'alerts', [])) : null;
  const alertCount = alertItems !== null
    ? alertItems.length
    : (metricRow ? pick(metricRow, 'alert_count', null) : null);
  const totalAlerts = pick(stats, 'alerts', null);
  const qualityConfig = pick(ctx.store.state.config, 'quality', {}) || {};
  const validRatioMin = pick(qualityConfig, 'valid_ratio_min', null);
  const subjectCount = stats.subjects;
  const doneCount = stats.sessions_done;
  const totalCount = stats.sessions;
  const shortUuid = (row) => (row && row.uuid ? `${String(row.uuid).slice(0, 8)}…` : DASH);
  const metricNote = metricRow
    ? `${newestDone ? '最近已完成会话' : '最近会话'} ${shortUuid(metricRow)} · ${fmtTime(metricRow.started_at, false)}`
      + (newest && newest.uuid !== metricRow.uuid ? `；最新会话${statusText(newest.status)}，未计入本次结果` : '')
    : '完成一次评估后，这里会显示本次结果。';
  const page = el('div', { class: 'overview-page' });
  page.append(el('header', { class: 'overview-heading' }, [
    el('div', {}, [
      el('p', { class: 'overview-eyebrow', text: '工作概览' }),
      el('h1', { text: '每一次练习，都有记录' }),
      el('p', { class: 'overview-intro', text: '回顾评估结果，关注长期变化，安排下一次专注练习。' }),
    ]),
    button('开始新的评估', () => ctx.navigate('#/start'), { primary: true }),
  ]));
  page.append(el('section', { class: 'overview-stats', 'aria-label': '已保存会话统计' }, [
    renderKpi('参与者', fmtInt(subjectCount), '当前数据目录'),
    renderKpi('已完成评估', fmtInt(doneCount), `共 ${fmtInt(totalCount)} 次会话`),
    renderKpi('正在进行', fmtInt(list(pick(health.data, 'active_sessions', [])).length), '后台运行的会话'),
  ]));

  /* ------------------------------------------------------ 最近会话表格 */
  const TERMINAL = new Set(['done', 'failed', 'cancelled', 'canceled', 'error', 'unfinished']);
  const isTerminal = (row) => TERMINAL.has(String(row.status || '').toLowerCase());
  const sessionRows = historyError ? latest.slice(0, 6) : recent;
  const sessionsCard = card('最近会话', [
    historyError ? el('p', { class: 'muted', text: `历史测试加载失败：${historyError}` }) : null,
    sessionRows.length ? table([
      {
        title: '会话 / 来源',
        render: (row) => {
          const documentUrl = sessionPresentationUrl(row) || (row.archived && historyArtifactUrl(row.report_url || row.evidence_url));
          const href = documentUrl || `#/${row.status === 'done' ? 'report' : 'flow'}?session=${row.uuid}`;
          return el('a', {
            class: 'overview-session-link', href,
            target: documentUrl ? '_blank' : null, rel: documentUrl ? 'noopener noreferrer' : null,
            title: row.hide_participant ? row.label : row.uuid,
          }, [
            el('strong', { text: row.label || row.participant || shortUuid(row) }),
            el('small', { text: row.hide_participant ? sessionSourceLabel(row)
              : [row.label && row.participant, shortUuid(row), sessionSourceLabel(row)].filter(Boolean).join(' · ') }),
          ]);
        },
      },
      {
        title: '状态',
        render: (row) => el('span', {
          class: 'overview-state', dataset: { state: row.status || 'unknown' },
          text: statusText(row.status), title: row.error ? `结束原因：${row.error}` : '',
        }),
      },
      {
        title: '进度 / 结束原因',
        render: (row) => {
          const presentation = sessionPresentationUrl(row);
          if (presentation && row.status === 'done') return el('a', {
            href: presentation, text: '展示报告', target: '_blank', rel: 'noopener noreferrer',
          });
          if (row.archived) return row.error || phaseText(row.phase, row.phase_label) || DASH;
          const duplicated = String(row.phase || '') === String(row.status || '');
          if (isTerminal(row) && duplicated) {
            return row.error
              ? el('span', { class: 'cell-reason', text: String(row.error), title: String(row.error) })
              : el('span', { class: 'muted', text: DASH });
          }
          return el('span', {}, [
            el('span', { text: phaseText(row.phase, row.phase_label) }),
            !isTerminal(row) ? el('small', { class: 'overview-progress', text: fmtPercent(row.progress) }) : null,
          ]);
        },
      },
      { title: '记录时间', render: (row) => fmtTime(row.created_at || row.started_at, false) },
    ], sessionRows) : el('div', { class: 'overview-empty' }, [
      el('span', { class: 'overview-empty__symbol', 'aria-hidden': 'true', text: '◎' }),
      el('strong', { text: '从第一次评估开始' }),
      el('p', { text: '连接设备并完成一次评估，结果会自动保存在这里。' }),
      button('进入评估', () => ctx.navigate('#/start')),
    ]),
  ], { class: 'overview-sessions', sub: historyTotal == null ? '' : `共 ${fmtInt(historyTotal)} 条记录，按记录时间排列`,
    actions: button('全部记录 →', () => ctx.navigate('#/history'), { small: true }) });

  /* ---------------------------------------------------------- 最近结果 */
  const snapshotBody = el('div', { class: 'overview-snapshot' });
  const metricSource = sessionSourceLabel({ ...metricRow, runtime: detail?.runtime });
  if (metricRow) {
    snapshotBody.append(el('div', { class: 'overview-snapshot__meta' }, [
      el('strong', { text: metricRow.participant || DASH }),
      el('span', { class: 'overview-source', text: metricSource,
        dataset: { simulated: metricSource === '模拟数据' ? 'true' : 'false' } }),
    ]));
  }
  snapshotBody.append(el('div', { class: 'overview-focus' }, [
    el('span', { text: '专注相关指数' }),
    el('strong', { text: fmtNum(pick(summary, 'focus.mean')) }),
    el('small', { text: '相对个人基线 · 0–1' }),
  ]));
  snapshotBody.append(el('div', { class: 'overview-secondary' }, [
    el('div', {}, [el('span', { text: '放松指数' }), el('strong', { text: fmtNum(pick(summary, 'relax.mean')) })]),
    el('div', {}, [el('span', { text: '认知负荷指数' }), el('strong', { text: fmtNum(pick(summary, 'load.mean')) })]),
  ]));
  const rawRatio = pick(quality, 'valid_ratio', null);
  const ratio = typeof rawRatio === 'number' && Number.isFinite(rawRatio) ? rawRatio : null;
  const threshold = typeof validRatioMin === 'number' && Number.isFinite(validRatioMin) ? validRatioMin : null;
  snapshotBody.append(el('dl', { class: 'overview-facts' }, [
    el('div', {}, [el('dt', { text: '质检可用窗比例' }), el('dd', { text: quality ? fmtPercent(ratio) : '无质检记录' })]),
    el('div', {}, [el('dt', { text: '本次预警' }), el('dd', { text: fmtInt(alertCount) })]),
  ]));
  if (quality) {
    const gaugeHost = el('div', { class: 'overview-quality-gauge' });
    const qualityDetails = disclosure('查看采集质量明细', [
      gaugeHost,
      el('dl', { class: 'overview-facts' }, [
        el('div', {}, [el('dt', { text: '可用窗 / 总窗' }),
          el('dd', { text: `${fmtInt(pick(quality, 'usable'))} / ${fmtInt(pick(quality, 'windows'))}` })]),
        el('div', {}, [el('dt', { text: '通过门槛' }), el('dd', { text: threshold === null ? '未配置' : fmtPercent(threshold) })]),
        el('div', {}, [el('dt', { text: '质检结果' }), el('dd', { text: pick(quality, 'passed') ? '质检通过' : '未达门槛' })]),
      ]),
    ]);
    qualityDetails.title = '门槛来源：config.quality.valid_ratio_min';
    let drawn = false;
    qualityDetails.addEventListener('toggle', () => {
      if (!qualityDetails.open || drawn || ctx.signal.aborted) return;
      drawGauge(gaugeHost, ratio, { color: SERIES_COLORS[2], target: threshold, label: '质检可用窗比例仪表' });
      drawn = true;
    });
    snapshotBody.append(qualityDetails);
  }
  snapshotBody.append(el('p', { class: 'overview-note', text: '指数描述相对基线的变化，不能解释为专注概率。缺失值不表示低专注。' }));
  if (metricRow?.status === 'done') {
    const presentation = sessionPresentationUrl(metricRow);
    snapshotBody.append(presentation ? el('a', {
      href: presentation, text: '展示报告', target: '_blank', rel: 'noopener noreferrer',
    }) : button('查看阶段记录', () => ctx.navigate(`#/report?session=${metricRow.uuid}`)));
  }
  const snapshotCard = card('最近一次结果', snapshotBody, { class: 'overview-result', sub: metricNote });
  page.append(el('div', { class: 'overview-main' }, [sessionsCard, snapshotCard]));

  /* -------------------------------------------------------------- 趋势图 */
  const trendBody = el('div');
  const trendCard = card('专注相关指数 · 周趋势', trendBody, { class: 'overview-trend',
    sub: '按同一参与者的可比会话汇总',
    actions: button('查看历史趋势 →', () => ctx.navigate('#/history'), { small: true }) });
  page.append(trendCard);

  /* ------------------------------------------------------------ 运行环境 */
  const engine = pick(health.data, 'engine', {}) || {};
  const environmentDetails = disclosure('运行环境与数据存储', [
    el('dl', { class: 'overview-environment' }, [
      el('div', {}, [el('dt', { text: '后台会话' }),
        el('dd', { text: `最多 ${fmtInt(pick(health.data, 'max_active_sessions'))} 个；当前运行 ${fmtInt(list(pick(health.data, 'active_sessions', [])).length)} 个` })]),
      el('div', {}, [el('dt', { text: '数据目录' }),
        el('dd', { class: 'mono', text: pick(health.data, 'data_dir', DASH) })]),
      el('div', {}, [el('dt', { text: '全库累计预警' }), el('dd', { text: fmtInt(totalAlerts) })]),
    ]),
  ], 'overview-environment-details');
  environmentDetails.title = `引擎口径：频谱 ${engine.spectrum || DASH}｜指标 ${engine.indicator || DASH}`
    + `｜基线 ${engine.baseline || DASH}｜评估 ${engine.assessment || DASH}`;
  page.append(environmentDetails);
  container.textContent = '';
  container.append(page);

  // 先挂载布局再画图，确保图表使用实际可用宽度。
  const trendCandidates = [];
  for (const row of latest) {
    const name = row && row.participant;
    if (name && !trendCandidates.includes(name)) trendCandidates.push(name);
    if (trendCandidates.length >= 5) break;
  }
  if (!trendCandidates.length) {
    trendBody.append(empty('完成评估后查看每周变化。系统只汇总可比较的已完成会话。'));
    return;
  }
  trendBody.append(el('p', { class: 'muted', text: '正在读取周趋势…' }));
  let shown = null;
  let lastData = null;
  let failed = 0;
  for (const participant of trendCandidates) {
    try {
      const response = await api.reportsTrend({ participant, field: 'focus', period: 'week' });
      if (ctx.signal.aborted) return;
      const data = response.data || {};
      lastData = data;
      const points = list(data.points).map(normalizeTrendPoint).filter(Boolean);
      if (points.length) {
        shown = { participant, data, points };
        break;
      }
    } catch (error) {
      if (ctx.signal.aborted) return;
      failed += 1;
    }
  }
  if (ctx.signal.aborted) return;
  trendBody.textContent = '';
  if (!shown) {
    const rejected = Number(pick(lastData, 'rejected', 0)) || 0;
    trendBody.append(empty(newestDone
      ? '最近参与者尚无可聚合的趋势点。'
        + (rejected ? ` ${fmtInt(rejected)} 条会话因设备、采样率或通道变化不可比较。` : '')
      : '完成一次评估后查看周趋势。'));
    if (failed) trendBody.append(el('p', { class: 'overview-note', text: `${fmtInt(failed)} 次趋势查询失败，请稍后重新打开本页。` }));
    return;
  }
  const { participant, data, points } = shown;
  const current = points[points.length - 1];
  trendBody.append(el('div', { class: 'overview-trend-meta' }, [
    el('span', { text: `${participant} · ${fmtInt(data.sessions)} 次会话`
      + (Number(data.rejected) ? ` · 排除 ${fmtInt(data.rejected)} 条不可比记录` : '') }),
    el('span', { text: `${current.label} · 指数 ${fmtNum(current.value)}` }),
  ]));
  const canvasHost = el('div', { class: 'chart-host' });
  trendBody.append(canvasHost);
  drawLineChart(canvasHost, points.map((point) => ({ label: point.label, values: { focus: point.value } })), {
    series: ['focus'], colors: [SERIES_COLORS[0]], max: 1, min: 0,
    showValues: true, height: points.length <= 2 ? 170 : 220,
    label: '专注相关指数周趋势折线图',
  });
  trendBody.append(disclosure('查看趋势数据与可比性说明', [
    table([
      { title: '周期', render: (row) => row.label },
      { title: '指数均值', align: 'right', render: (row) => fmtNum(row.value) },
      { title: '有效窗 n', align: 'right', render: (row) => fmtInt(row.n) },
    ], points),
    data.note ? el('p', { class: 'overview-note', text: data.note }) : null,
  ]));
}

export default render;
