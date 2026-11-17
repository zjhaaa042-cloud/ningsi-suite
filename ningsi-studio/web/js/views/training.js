/**
 * 训练视图：大号专注度仪表 + 目标线、连续保持状态、分段表格、训练前后基线对比。
 *
 * 数据来源：
 * - SSE：`training_start`（模式/段数/初始目标）、`feedback`（逐窗目标与达标）、`segment`（分段统计）
 * - GET /api/sessions/{uuid}/training → segments[] 与 summary（含 baseline_before / baseline_after）
 */

import { api } from '../api.js';
import { describeError, toast } from '../store.js';
import { connectSessionEvents } from '../sse.js';
import {
  DASH, button, card, el, empty, fmtDuration, fmtInt, fmtNum, fmtPercent,
  list, pageFrame, pick, statusText, table,
} from '../util.js';
import { SERIES_COLORS, drawLineChart, drawGauge } from '../charts.js';
import { effectiveTimeContent, effectiveTimeOutcome, effectiveTimeState, formatWallRatio, trainingCoverageNote, trainingMetrics } from '../training-metrics.js';

/**
 * 训练前后基线：结构来自引擎 Baseline.as_dict()
 * （spec/srate/channels/device/n_windows/valid/reasons/bands/indices），
 * indices 为 focus/relax/load 的 {mean,std,median,n}。
 */
function baselineRow(label, baseline) {
  const indices = pick(baseline, 'indices', {}) || {};
  const fmt = (name) => {
    const stat = indices[name];
    if (!stat) return DASH;
    return `${fmtNum(pick(stat, 'mean'), 3)}（n=${fmtInt(pick(stat, 'n'))}）`;
  };
  return [
    label,
    fmtInt(pick(baseline, 'n_windows')),
    `${pick(baseline, 'device', DASH)} · ${fmtNum(pick(baseline, 'srate'), 0)} Hz`,
    pick(baseline, 'valid') ? '有效' : '无效',
    `${fmt('focus')} / ${fmt('relax')} / ${fmt('load')}`,
  ];
}

function baselineEntries(summary) {
  const before = pick(summary, 'baseline_before');
  const after = pick(summary, 'baseline_after');
  const rows = [];
  if (before) rows.push(baselineRow('训练前基线', before));
  if (after) rows.push(baselineRow('训练后基线', after));
  return rows;
}

export function render(container, ctx) {
  const uuid = ctx.params.get('session') || pick(ctx.store.state.currentSession, 'uuid', null);
  const actions = uuid ? [
    button('流程视图', () => ctx.navigate(`#/flow?session=${uuid}`)),
    button('实时监测', () => ctx.navigate(`#/live?session=${uuid}`)),
    button('报告', () => ctx.navigate(`#/report?session=${uuid}`)),
  ] : [];
  const host = pageFrame(container, '神经反馈训练', actions);
  host.classList.add('training-page');

  if (!uuid) {
    host.append(card('尚未选择会话', el('div', {}, [
      empty('请先选择一次包含训练阶段的会话。'),
      el('div', { class: 'row', style: 'margin-top:12px' }, [
        button('前往历史会话', () => ctx.navigate('#/history'), { primary: true }),
        button('前往被试管理', () => ctx.navigate('#/subjects')),
      ]),
    ])));
    return;
  }

  // 配置读取必须是"惰性"的：直接进本页时 /api/config 还没回来（boot 里是异步加载），
  // 一次性取值会把 target_min / target_max / hold_sec 全部固化成 "—"（实测渲染成 "—–—"）
  const trainingConfig = () => pick(ctx.store.state.config, 'training', {}) || {};
  const local = {
    uuid,
    target: null,
    score: null,
    onTarget: null,
    currentHold: null,
    maxHold: null,
    segment: null,
    mode: null,
    holdSec: pick(trainingConfig(), 'hold_sec', null),
    segments: [],
    summary: null,
    trainingEnded: false,
    segmentStats: null,
    effectiveTime: null,
    timeScale: pick(ctx.store.state.currentSession, 'time_scale', 1),
    title: '训练进行中',
  };

  /**
   * 会话结束后 `feedback` 不再推送，使用已保存的最终段 + summary 展示结果。
   * 训练结束后才显示落库结果；实时缺测必须保持缺测，不能借用上一段成绩。
   */
  const finalView = () => {
    const summarySegments = list(pick(local.summary, 'segments', []));
    const summarySeg = summarySegments.length ? summarySegments[summarySegments.length - 1] : null;
    const seg = local.segments.length ? local.segments[local.segments.length - 1] : summarySeg;
    const stats = (seg && seg.stats) || {};
    const firstDefined = (...values) => values.find((value) => value !== null && value !== undefined);
    const saved = local.trainingEnded;
    const emptySegment = stats.n === 0;
    return {
      usingFallback: saved,
      hasData: Boolean(seg),
      score: saved
        ? (seg ? (emptySegment ? null : pick(stats, 'mean')) : pick(local.summary, 'mean_focus'))
        : local.score,
      target: saved
        ? firstDefined(seg && seg.target, pick(local.summary, 'final_target'), pick(local.summary, 'initial_target'))
        : local.target,
      achieved: saved
        ? (!emptySegment && typeof stats.achieved === 'boolean' ? stats.achieved : null)
        : local.onTarget,
      seq: saved ? (seg ? (typeof seg.seq === 'number' ? seg.seq + 1 : seg.index) : null) : local.segment,
      currentHold: saved ? pick(stats, 'current_hold_sec') : local.currentHold,
      maxHold: saved ? pick(stats, 'max_hold_sec') : local.maxHold,
      holdSec: !saved && local.holdSec !== null && local.holdSec !== undefined
        ? local.holdSec
        : firstDefined(pick(seg, 'hold_sec'), pick(seg, 'hold_after'), pick(trainingConfig(), 'hold_sec')),
      mode: local.mode || pick(local.summary, 'mode'),
      metrics: saved ? trainingMetrics(seg ? stats : local.summary || {}, pick(seg, 'duration_sec'))
        : local.segmentStats ? trainingMetrics(local.segmentStats) : null,
      segmentCount: local.segments.length || summarySegments.length || null,
    };
  };

  const gaugeHost = el('div');
  const statusHost = el('div');
  const segmentHost = el('div');
  const baselineHost = el('div');
  const noteHost = el('div');
  const effectiveHost = el('div');
  host.append(effectiveHost);
  host.append(el('div', { class: 'training-overview' }, [gaugeHost, statusHost]));
  host.append(segmentHost);
  host.append(baselineHost);
  host.append(noteHost);

  /**
   * noteHost 是"追加型"容器：所有写入都先清空再挂一个节点，
   * 避免同一张说明卡被 SSE 重放 / 轮询刷新追加多次（幂等渲染）。
   */
  const renderNote = (node) => {
    noteHost.textContent = '';
    if (node) noteHost.append(node);
  };

  const renderEffective = () => {
    effectiveHost.textContent = '';
    const aggregate = Boolean(local.trainingEnded && local.summary?.effective_time);
    const progress = effectiveTimeState(aggregate ? local.summary.effective_time : local.effectiveTime, 'training');
    if (progress) effectiveHost.append(card('有效时长训练进度', effectiveTimeContent(progress, { aggregate })));
  };

  const renderGauge = () => {
    gaugeHost.textContent = '';
    const view = finalView();
    if (local.feedbackMode && local.feedbackMode !== 'eeg' && !local.trainingEnded) {
      gaugeHost.append(card('当前反馈模式', el('p', { text: local.modeLabel })));
      return;
    }
    const body = el('div');
    gaugeHost.append(card('专注度仪表', body, {
      sub: `目标随训练表现调整；允许目标范围 ${fmtNum(pick(trainingConfig(), 'target_min'), 2)}–${fmtNum(pick(trainingConfig(), 'target_max'), 2)}`,
    }));
    drawGauge(body, view.score, {
      target: view.target,
      color: 'var(--accent)',
      note: view.target === null || view.target === undefined ? '' : `目标 ${fmtNum(view.target, 3)}`,
      label: '专注度仪表',
    });
    if (view.usingFallback && view.hasData) {
      body.append(el('p', { class: 'muted', text: '仪表显示已保存的最后一段平均专注度和训练目标。' }));
    }
  };

  const renderStatus = () => {
    statusHost.textContent = '';
    const view = finalView();
    const achieved = view.achieved === null || view.achieved === undefined
      ? DASH
      : (view.achieved ? '达标' : '未达标');
    const metrics = view.metrics;
    const body = el('div');
    body.append(el('div', { class: 'training-stats' }, [
      el('div', { class: 'training-primary' }, [el('p', { class: 'muted', text: '有效时间内达标率' }), el('p', { class: 'mono kpi__value', text: fmtPercent(metrics?.validRatio) })]),
      el('div', { class: 'training-primary' }, [el('p', { class: 'muted', text: '有效时间覆盖率' }), el('p', { class: 'mono kpi__value', text: fmtPercent(metrics?.coverage) })]),
      el('div', {}, [el('p', { class: 'muted', text: local.timeScale !== 1 ? '有效反馈秒数 / 协议时长' : '有效反馈秒数 / 实际时长' }), el('p', { class: 'mono', text: `${fmtDuration(metrics?.validSeconds)} / ${fmtDuration(metrics?.duration)}` })]),
      el('div', {}, [el('p', { class: 'muted', text: '本段达标时间占比' }), el('p', { class: 'mono', text: metrics ? formatWallRatio(metrics) : DASH })]),
      el('div', {}, [el('p', { class: 'muted', text: view.usingFallback ? '末段平均专注度' : '当前专注度' }), el('p', { class: 'mono kpi__value', text: fmtNum(view.score, 3) })]),
      el('div', {}, [el('p', { class: 'muted', text: '目标线' }), el('p', { class: 'mono kpi__value', text: fmtNum(view.target, 3) })]),
      el('div', {}, [el('p', { class: 'muted', text: '连续保持状态' }), el('p', { class: 'kpi__value', text: achieved })]),
      el('div', {}, [el('p', { class: 'muted', text: '当前段' }), el('p', { class: 'mono', text: fmtInt(view.seq) })]),
      el('div', {}, [el('p', { class: 'muted', text: '训练模式' }), el('p', { text: ({ quick: '简短训练', full: '标准训练', ten_min: '十分钟分级训练', effective_time: '有效时长训练' })[view.mode] || DASH })]),
      el('div', {}, [el('p', { class: 'muted', text: '保持时长要求' }), el('p', { class: 'mono', text: fmtDuration(view.holdSec) })]),
      el('div', {}, [el('p', { class: 'muted', text: '当前连续保持 / 本段最长' }), el('p', { class: 'mono', text: `${fmtDuration(view.currentHold)} / ${fmtDuration(view.maxHold)}` })]),
      el('div', { class: 'training-hold' }, [
        el('span', { class: 'muted', text: view.usingFallback ? '末段最长保持 / 目标要求' : '连续保持进度' }),
        el('progress', { max: 1, value: typeof view.holdSec === 'number' && view.holdSec > 0 ? Math.min(1, Math.max(0, Number(view.usingFallback ? view.maxHold : view.currentHold) || 0) / view.holdSec) : 0, 'aria-label': '连续保持进度' }),
      ]),
    ]));
    if (metrics) body.append(el('p', { class: 'training-coverage-note', role: 'status', text: trainingCoverageNote(metrics) }));
    if (local.timeScale !== 1) body.append(el('p', { class: 'muted', text: '加速模拟：此处训练统计按协议秒记录。' }));
    statusHost.append(card('连续保持状态', body, {
      sub: `达标需连续保持 ${fmtDuration(view.holdSec)}；目标按表现自适应（步长 ${fmtNum(pick(trainingConfig(), 'target_step'), 2)}，区间 ${fmtNum(pick(trainingConfig(), 'target_min'), 2)}–${fmtNum(pick(trainingConfig(), 'target_max'), 2)}）`,
    }));
  };

  const renderSegments = () => {
    segmentHost.textContent = '';
    if (!local.segments.length) {
      segmentHost.append(card('训练分段', [empty('训练阶段开始后，每段结束会显示统计结果。')]));
      return;
    }
    const rows = local.segments.map((segment) => ({
      ...segment,
      __stats: segment.stats || {},
      __metrics: trainingMetrics(segment.stats || {}, segment.duration_sec),
    }));
    const body = el('div');
    const chartHost = el('div');
    body.append(table([
      { title: '段', render: (row) => fmtInt(Number(pick(row, 'seq', 0)) + 1) },
      { title: '有效时间内达标率', align: 'right', render: (row) => fmtPercent(row.__metrics.validRatio) },
      { title: '有效覆盖', align: 'right', render: (row) => fmtPercent(row.__metrics.coverage) },
      { title: local.timeScale !== 1 ? '有效秒数 / 协议时长' : '有效秒数 / 实际时长', render: (row) => `${fmtDuration(row.__metrics.validSeconds)} / ${fmtDuration(row.__metrics.duration)}` },
      { title: '可解释性', render: (row) => !row.__metrics.modern ? '旧版口径' : row.__metrics.sufficient ? '覆盖充足' : '样本不足' },
      { title: '目标（前→后）', render: (row) => `${fmtNum(row.target, 3)} → ${fmtNum(row.target_after, 3)}` },
      { title: '有效时长结果', render: row => effectiveTimeOutcome(row.effective_time) },
    ], rows));
    body.append(el('p', { class: 'muted', text: '有效时间内达标率与覆盖率并列阅读。覆盖不足60%时样本不足、保持目标；缺失不填零，不用上一分数代替。' }));
    if (rows.some(row => !row.__metrics.modern)) body.append(el('p', { class: 'muted', text: trainingCoverageNote(trainingMetrics()) }));
    body.append(el('h4', { style: 'margin:16px 0 6px', text: '各段有效时间内达标率与覆盖率' }));
    body.append(el('p', { class: 'muted', text: '蓝线：有效时间内达标率；绿线：有效覆盖率。无有效反馈的段留空，不跨缺失段连线。' }));
    body.append(chartHost);
    segmentHost.append(card('训练分段', body));
    drawLineChart(chartHost, rows.map((row, index) => ({
      label: `第${fmtInt(Number(row.seq ?? index) + 1)}段`,
      values: { success: row.__metrics.validRatio, coverage: row.__metrics.coverage },
    })), { series: ['success', 'coverage'], colors: [SERIES_COLORS[0], SERIES_COLORS[2]], max: 1, percent: true, height: 180, label: '各段有效时间内达标率与有效覆盖率', showValues: true });
    body.append(el('details', { class: 'result-details' }, [
      el('summary', { text: '查看实际时间占比、连续保持与调整明细' }),
      table([
        { title: '段', render: row => fmtInt(Number(pick(row, 'seq', 0)) + 1) },
        { title: '实际时间达标占比 / 旧版样本占比', render: row => formatWallRatio(row.__metrics) },
        { title: '均值', render: row => fmtNum(row.__stats.mean, 3) },
        { title: '保持要求（前→后）', render: row => `${fmtDuration(row.hold_sec)} → ${fmtDuration(row.hold_after)}` },
        { title: '最长保持', render: row => fmtDuration(row.__stats.max_hold_sec) },
        { title: '连续达标', render: row => row.__stats.n === 0 ? DASH : typeof row.__stats.achieved === 'boolean' ? (row.__stats.achieved ? '达标' : '未达标') : DASH },
        { title: '波动', render: row => fmtNum(row.__stats.volatility, 3) },
        { title: '有效点 / 排除窗', render: row => `${fmtInt(row.__stats.n)} / ${fmtInt(row.excluded_windows)}` },
        { title: '调整依据', render: row => row.adaptation_reason || DASH },
      ], rows),
    ]));
  };

  const renderBaseline = (summary) => {
    baselineHost.textContent = '';
    const rows = baselineEntries(summary);
    const comparable = pick(summary, 'baseline_comparable');
    const durations = list(summary?.segments).map(segment => segment.duration_sec);
    const duration = durations.length && durations.every(Number.isFinite) ? durations.reduce((sum, seconds) => sum + seconds, 0) : null;
    const metrics = trainingMetrics(summary || {}, duration);
    const body = el('div');
    if (rows.length) {
      body.append(table([
        { title: '基线', render: (row) => row[0] },
        { title: '可用窗', align: 'right', render: (row) => row[1] },
        { title: '设备 / 采样率', render: (row) => row[2] },
        { title: '有效性', render: (row) => row[3] },
        { title: '基线对数特征（专注 / 放松 / 负荷）', render: (row) => row[4] },
      ], rows));
      body.append(el('p', { class: 'muted', text: '上表为对数频带比值的基线参照，允许负数或大于1，不是0–1评分。' }));
    } else {
      body.append(empty('尚无训练前后基线记录。'));
    }
    body.append(el('div', { class: 'grid grid--3', style: 'margin-top:12px' }, [
      el('div', {}, [el('p', { class: 'muted', text: '有效时间内达标率' }), el('p', { class: 'mono kpi__value', text: fmtPercent(metrics.validRatio) })]),
      el('div', {}, [el('p', { class: 'muted', text: '有效时间覆盖率' }), el('p', { class: 'mono kpi__value', text: fmtPercent(metrics.coverage) })]),
      el('div', {}, [el('p', { class: 'muted', text: local.timeScale !== 1 ? '有效反馈秒数 / 协议时长' : '有效反馈秒数 / 实际时长' }), el('p', { class: 'mono', text: `${fmtDuration(metrics.validSeconds)} / ${fmtDuration(metrics.duration)}` })]),
      el('div', {}, [el('p', { class: 'muted', text: '训练平均专注度' }), el('p', { class: 'mono', text: fmtNum(pick(summary, 'mean_focus'), 3) })]),
      el('div', {}, [el('p', { class: 'muted', text: '实际时间达标占比 / 旧版样本占比' }), el('p', { class: 'mono', text: formatWallRatio(metrics) })]),
      el('div', {}, [el('p', { class: 'muted', text: '首末变化' }), el('p', { class: 'mono', text: fmtNum(pick(summary, 'first_to_last_change'), 4) })]),
    ]));
    if (Object.keys(summary || {}).length) body.append(el('p', { class: 'training-coverage-note', role: 'status', text: trainingCoverageNote(metrics) }));
    if (comparable === false) {
      body.append(el('p', { class: 'muted', text: '训练前后基线不可比（设备 / 采样率 / 通道或可用窗不足），不作为效果结论。' }));
    }
    baselineHost.append(card('训练前后基线对比', body, {
      sub: pick(summary, 'target_rationale') && typeof pick(summary, 'target_rationale') === 'string'
        ? `目标依据：${pick(summary, 'target_rationale')}`
        : '训练目标初始值取自当次基线中位数对应的评分',
    }));
  };

  renderGauge();
  renderStatus();
  renderSegments();

  const applyFeedback = (payload, fromSnapshot = false) => {
    if (!payload || local.trainingEnded) return;
    if (fromSnapshot && local.hasLiveFeedback) {
      // A request made before the latest SSE frame may resolve later; don't rewind it.
      const nextSegment = Number(payload.segment);
      const currentSegment = Number(local.segment);
      const nextElapsed = payload.segment_stats?.duration_sec;
      const currentElapsed = local.segmentStats?.duration_sec;
      if (nextSegment < currentSegment || (nextSegment === currentSegment
        && (!Number.isFinite(nextElapsed) || !Number.isFinite(currentElapsed) || nextElapsed <= currentElapsed))) return;
    }
    if (!fromSnapshot) local.hasLiveFeedback = true;
    const usable = payload.usable !== false && Number.isFinite(payload.score);
    local.feedbackMode = payload.feedback_mode;
    local.modeLabel = payload.mode_label;
    local.segment = payload.segment;
    local.currentHold = usable ? payload.current_hold_sec ?? null : 0;
    local.maxHold = payload.max_hold_sec ?? null;
    local.holdSec = payload.hold_sec ?? local.holdSec;
    local.target = Number.isFinite(payload.target) ? payload.target : null;
    local.score = usable ? payload.score : null;
    local.onTarget = usable && typeof payload.achieved === 'boolean' ? payload.achieved : null;
    local.segmentStats = payload.segment_stats || null;
    if (effectiveTimeState(payload.effective_time)) { local.effectiveTime = payload.effective_time; local.mode = 'effective_time'; }
    if (Number.isFinite(payload.effective_time?.time_scale)) local.timeScale = payload.effective_time.time_scale;
    renderEffective();
    renderGauge();
    renderStatus();
  };

  /* -------------------------------------------------------- 事件接线 */
  const handleEvent = (type, payload) => {
    switch (type) {
      case 'training_start': {
        local.mode = payload.mode;
        local.target = typeof payload.target === 'number' ? payload.target : null;
        local.holdSec = payload.hold_sec ?? local.holdSec;
        if (effectiveTimeState(payload.effective_time)) local.effectiveTime = payload.effective_time;
        renderNote(card('训练设置', el('div', {}, [
          el('p', { text: `模式 ${payload.mode || DASH}｜${fmtInt(payload.segments)} 段 × ${fmtDuration(payload.segment_sec)}｜初始目标 ${fmtNum(payload.target, 3)}｜保持 ${fmtDuration(payload.hold_sec)}` }),
          payload.rationale ? el('p', { class: 'muted', text: `目标依据：${typeof payload.rationale === 'string' ? payload.rationale : JSON.stringify(payload.rationale)}` }) : null,
        ])));
        renderGauge();
        renderStatus();
        renderEffective();
        break;
      }
      case 'feedback': {
        applyFeedback(payload);
        break;
      }
      case 'effective_progress': {
        if (payload.key === 'training' && payload.effective_time) {
          local.effectiveTime = payload.effective_time;
          renderEffective();
        }
        break;
      }
      case 'segment': {
        // 幂等入列：SSE 断线重连会带 Last-Event-ID 重放，按 seq 覆盖而不是重复 push，
    // 否则分段表与各段趋势图会出现重复条目
        const seq = Number(payload.seq);
        const existing = Number.isFinite(seq)
          ? local.segments.findIndex((item) => Number(item.seq) === seq)
          : -1;
        if (existing >= 0) local.segments[existing] = payload;
        else local.segments.push(payload);
        renderSegments();
        // 段结束后目标可能已自适应，及时同步
        if (typeof payload.target_after === 'number') local.target = payload.target_after;
        if (payload.effective_time) local.effectiveTime = payload.effective_time;
        renderEffective();
        renderGauge();
        renderStatus();
        break;
      }
      case 'phase': {
        if (payload.key === 'training' && payload.state === 'done') {
          local.trainingEnded = true;
          local.title = '训练阶段已结束';
          renderGauge();
          renderStatus();
          renderEffective();
        }
        break;
      }
      case 'finished': {
        local.trainingEnded = true;
        if (local.finishedNoted) break;
        local.finishedNoted = true;
        stopUpdates();
        renderGauge();
        renderStatus();
        renderNote(card('会话结束', el('div', {}, [
          el('p', { text: `最终状态：${statusText(payload.status)}` }),
          button(payload.status === 'done' ? '查看报告与产物' : '查看阶段记录',
            () => ctx.navigate(`#/report?session=${uuid}`)),
        ])));
        loadTraining();
        break;
      }
      default:
        break;
    }
  };

  /* ------------------------------------------------------ 初次加载 */
  let requestVersion = 0;
  const loadTraining = async () => {
    const version = ++requestVersion;
    try {
      const response = await api.training(uuid);
      if (ctx.signal.aborted || version !== requestVersion) return;
      const data = response.data || {};
      const segments = list(data.segments).map((segment) => ({
        ...segment,
        stats: segment.stats && typeof segment.stats === 'object' ? segment.stats : {
          mean: segment.mean_score,
          on_target_ratio: segment.on_target_ratio,
          n: segment.n,
        },
      }));
      if (segments.length) local.segments = segments;
      local.summary = data.summary && Object.keys(data.summary).length ? data.summary : null;
      if (local.summary) {
        local.trainingEnded = true;
      }
      if (data.current_feedback) applyFeedback(data.current_feedback, true);
      if (!local.hasLiveFeedback && effectiveTimeState(data.effective_time, 'training')) local.effectiveTime = data.effective_time;
      renderSegments();
      renderBaseline(data.summary || {});
      // 会话结束后 feedback 停推，仪表/连续保持状态必须由落库数据兜底刷新（否则整卡都是 —）
      renderGauge();
      renderStatus();
      renderEffective();
    } catch (error) {
      if (ctx.signal.aborted || version !== requestVersion) return;
      if (error && error.name === 'ApiError' && (error.status === 404 || error.status === 409)) {
        if (local.finishedNoted && local.segments.length === 0) {
          for (const node of [gaugeHost, statusHost, segmentHost, effectiveHost]) node.textContent = '';
          baselineHost.textContent = '';
          baselineHost.append(card('训练记录', [empty('会话已结束，没有已保存的训练分段。')]));
          return;
        }
        baselineHost.textContent = '';
        baselineHost.append(card('训练前后基线对比', [empty('该会话暂无训练记录（训练阶段尚未结束或未执行）。')]));
        return;
      }
      baselineHost.textContent = '';
      baselineHost.append(card('训练前后基线对比', [empty(`加载失败：${describeError(error)}`)]));
      toast(describeError(error));
    }
  };

  let events, timer;
  function stopUpdates() {
    events?.close();
    window.clearInterval(timer);
  }
  ctx.onCleanup(stopUpdates);

  api.getSession(uuid).then((response) => {
    if (ctx.signal.aborted) return;
    const data = response.data || {};
    const session = data.session ? { ...data.session } : { ...data };
    session.uuid = uuid;
    ctx.store.setState({ currentSession: session });
    if (session.protocol === 'short') {
      host.textContent = '';
      host.append(card('训练记录', [empty('短协议不包含反馈训练阶段。')]));
      return;
    }
    if (Number.isFinite(session.time_scale)) local.timeScale = session.time_scale;
    const runtime = data.runtime || {};
    if (runtime.latest_feedback) applyFeedback(runtime.latest_feedback, true);
    if (!local.hasLiveFeedback && session.phase === 'training' && effectiveTimeState(runtime.effective_time, 'training')) local.effectiveTime = runtime.effective_time;
    if (['done', 'failed', 'cancelled'].includes(session.status)) {
      handleEvent('finished', {status: session.status});
      return;
    }
    if (list(session.runs).some(run => run.phase === 'training' && run.status === 'done')) {
      local.trainingEnded = true;
      renderGauge();
      renderStatus();
    }
    renderEffective();
    renderStatus();
    renderSegments();
    if (local.summary) renderBaseline(local.summary);
    loadTraining();
    events = connectSessionEvents(uuid, {
      onEvent: (type, frame) => {
        if (!ctx.signal.aborted) handleEvent(type, frame && frame.data);
      },
    });
    timer = window.setInterval(() => {
      if (!document.hidden) loadTraining();
    }, 10000);
  }).catch(error => {
    if (ctx.signal.aborted) return;
    host.textContent = '';
    host.append(card('会话加载失败', [empty(describeError(error)), button('重试', () => ctx.reload())]));
  });
}

export default render;
