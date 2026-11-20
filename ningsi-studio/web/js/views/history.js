/**
 * 历史视图：会话表格（分页 + 按被试/状态筛选）、周/月趋势、不可比提示、单会话导出。
 *
 * 数据来源：
 * - GET /api/test-history?participant=&status=&page=&limit=
 * - GET /api/reports/trend?participant=&field=&period=（含 note 与 ledger_points）
 * - GET /api/sessions/{uuid}/export.zip（导出链接，直接指向接口）
 */

import { api, historyArtifactUrl } from '../api.js';
import { sessionPresentationUrl, sessionSourceLabel } from '../session-history.js';
import { describeError, toast } from '../store.js';
import {
  DASH, button, card, el, empty, field, fmtInt, fmtNum, fmtPercent,
  fmtTime, list, pageFrame, phaseText, pick, statusText, table,
} from '../util.js';
import { SERIES_COLORS, drawLineChart } from '../charts.js';

const STATUS_OPTIONS = [
  ['', '全部状态'],
  ['running', '进行中'],
  ['done', '已完成'],
  ['failed', '失败'],
  ['cancelled', '已取消'],
  ['unfinished', '未记录结束'],
];

/** 视图内筛选条件：切换页签/返回时保留。 */
const view = { participant: '', status: '', page: 1, field: 'focus', period: 'week' };

function normalizePoint(point) {
  if (!point || typeof point !== 'object') return null;
  const label = point.period ?? point.label ?? point.bucket ?? point.t ?? point.key;
  const value = point.mean ?? point.value ?? point.focus ?? point.avg;
  return {
    label: label === undefined || label === null ? DASH : String(label),
    value: typeof value === 'number' && Number.isFinite(value) ? value : null,
  };
}

export async function render(container, ctx) {
  const host = pageFrame(container, '历史与趋势', [
    button('刷新', () => ctx.reload()),
  ]);

  /* ------------------------------------------------------------ 筛选项 */
  let sessionsData;
  try {
    const response = await api.listTestHistory({
      participant: view.participant || undefined,
      status: view.status || undefined,
      page: view.page,
      limit: 20,
    });
    sessionsData = response.data || {};
  } catch (error) {
    if (!ctx.signal.aborted) {
      host.append(card('会话列表', [empty(`加载失败：${describeError(error)}`)]));
      toast(describeError(error));
    }
    return;
  }
  if (ctx.signal.aborted) return;
  const participants = list(sessionsData.participants);

  const participantSelect = el('select', { id: 'history-participant' });
  participantSelect.append(el('option', { value: '', text: '全部被试' }));
  for (const publicId of participants) participantSelect.append(el('option', { value: publicId, text: `sub-${publicId}` }));
  participantSelect.value = view.participant;

  const statusSelect = el('select', { id: 'history-status' });
  for (const [value, text] of STATUS_OPTIONS) statusSelect.append(el('option', { value, text }));
  statusSelect.value = view.status;

  const fieldSelect = el('select', { id: 'history-field' });
  for (const [value, text] of [['focus', '专注度'], ['relax', '放松度'], ['load', '认知负荷'], ['stress', '当前压力自评（0–10 分）']]) {
    fieldSelect.append(el('option', { value, text }));
  }
  fieldSelect.value = view.field;

  const periodSelect = el('select', { id: 'history-period' });
  for (const [value, text] of [['week', '按周'], ['month', '按月']]) {
    periodSelect.append(el('option', { value, text }));
  }
  periodSelect.value = view.period;

  const applyFilters = () => ctx.reload();
  for (const node of [participantSelect, statusSelect]) {
    node.addEventListener('change', () => {
      view.participant = participantSelect.value;
      view.status = statusSelect.value;
      view.page = 1;
      applyFilters();
    });
  }

  const filterCard = card('筛选', el('div', { class: 'filters' }, [
    field('被试', participantSelect),
    field('会话状态', statusSelect),
    field('趋势指标', fieldSelect),
    field('趋势周期', periodSelect),
    button('重算趋势', () => {
      view.field = fieldSelect.value;
      view.period = periodSelect.value;
      applyFilters();
    }),
  ]), { sub: '脑电趋势仅聚合同一协议、训练模式、来源、倍率与设备配置；压力自评取本人填写的分数。' });
  host.append(filterCard);

  /* -------------------------------------------------------- 会话表格 */
  const items = list(pick(sessionsData, 'items', []));
  const total = Number(pick(sessionsData, 'total', 0)) || 0;
  const limit = Number(pick(sessionsData, 'limit', 20)) || 20;
  const page = Number(pick(sessionsData, 'page', view.page)) || 1;
  const pageCount = Math.max(1, Math.ceil(total / limit));
  const participantColumns = items.some(row => !row.hide_participant && row.participant)
    ? [{ title: '被试', render: row => !row.hide_participant && row.participant ? `sub-${row.participant}` : DASH }]
    : [];

  const pagination = el('div', { class: 'row row--between', style: 'margin-top:12px' }, [
    el('span', { class: 'muted', text: `第 ${fmtInt(page)} / ${fmtInt(pageCount)} 页，共 ${fmtInt(total)} 条` }),
    el('div', { class: 'row' }, [
      button('上一页', () => {
        view.page = Math.max(1, page - 1);
        ctx.reload();
      }, { small: true, disabled: page <= 1 }),
      button('下一页', () => {
        view.page = Math.min(pageCount, page + 1);
        ctx.reload();
      }, { small: true, disabled: page >= pageCount }),
    ]),
  ]);

  host.append(card('会话列表', el('div', {}, [
    items.length ? table([
      {
        title: '会话',
        render: (row) => {
          const report = sessionPresentationUrl(row);
          const documentUrl = report || (row.archived && historyArtifactUrl(row.report_url || row.evidence_url));
          return el('a', {
            class: 'overview-session-link',
            href: documentUrl || `#/live?session=${row.uuid}`,
            target: documentUrl ? '_blank' : null, rel: documentUrl ? 'noopener noreferrer' : null,
            title: row.hide_participant ? row.label : row.uuid,
          }, [
            row.label && el('strong', { text: row.label }),
            (!row.hide_participant || !row.label) && el(row.label ? 'small' : 'span', { text: `${String(row.uuid).slice(0, 8)}…` }),
          ]);
        },
      },
      ...participantColumns,
      { title: '状态', render: (row) => statusText(row.status) },
      { title: '方案', render: (row) => row.protocol_label || DASH },
      { title: '阶段', render: (row) => phaseText(row.phase, row.phase_label) },
      { title: '进度', align: 'right', render: (row) => fmtPercent(row.progress) },
      { title: '数据来源', render: (row) => el('span', { text: sessionSourceLabel(row), title: row.source || row.device }) },
      { title: '时间倍率', align: 'right', render: (row) => (row.source || '').startsWith('replay:') ? DASH : fmtNum(row.time_scale, 2) },
      { title: '预警', align: 'right', render: (row) => fmtInt(row.alert_count) },
      { title: '记录时间', render: (row) => fmtTime(row.created_at || row.started_at) },
      {
        title: '操作',
        render: (row) => row.archived ? el('div', { class: 'row' }, [
          row.presentation_url && el('a', { href: historyArtifactUrl(row.presentation_url), text: '展示报告', target: '_blank', rel: 'noopener noreferrer' }),
          row.report_url && el('a', { href: historyArtifactUrl(row.report_url), text: '原始报告', target: '_blank', rel: 'noopener noreferrer' }),
          row.evidence_url && el('a', { href: historyArtifactUrl(row.evidence_url), text: '记录详情', target: '_blank', rel: 'noopener noreferrer' }),
        ]) : el('div', { class: 'row' }, [
          row.has_report && el('a', {
            href: sessionPresentationUrl(row), text: '展示报告', target: '_blank', rel: 'noopener noreferrer',
            attrs: { 'aria-label': `查看会话 ${row.uuid} 的展示报告` },
          }),
          row.has_artifacts && el('a', {
            href: api.exportZipUrl(row.uuid),
            text: '导出 zip',
            attrs: {
              title: '打包下载该会话全部产物',
              'aria-label': `导出会话 ${row.uuid} 的全部产物（zip 打包下载）`,
            },
          }),
          el('a', {
            href: `#/report?session=${row.uuid}`,
            text: row.has_report ? '报告详情' : '阶段记录',
            attrs: { 'aria-label': `查看会话 ${row.uuid} 的${row.has_report ? '报告详情' : '阶段记录'}` },
          }),
          row.has_training && el('a', {
            href: `#/training?session=${row.uuid}`,
            text: '训练',
            attrs: { 'aria-label': `查看会话 ${row.uuid} 的神经反馈训练视图` },
          }),
        ]),
      },
    ], items) : empty('没有符合筛选条件的会话。'),
    pagination,
  ]), { class: 'card--scroll' }));

  /* ---------------------------------------------------------- 趋势图 */
  const trendHost = el('div');
  host.append(card('周 / 月趋势', trendHost, {
    sub: '仅汇总当前 Studio 数据库中可比较的会话。',
  }));
  if (view.participant && !list(sessionsData.live_participants).includes(view.participant)) {
    trendHost.append(empty('该编号仅有归档记录，可从会话列表查看各次测试结果。'));
    return;
  }

  try {
    const response = await api.reportsTrend({
      participant: view.participant || undefined,
      field: view.field,
      period: view.period,
    });
    if (ctx.signal.aborted) return;
    const data = response.data || {};
    const points = list(data.points).map(normalizePoint).filter(Boolean);
    const ledgerPoints = list(data.ledger_points).map(normalizePoint).filter(Boolean);

    trendHost.append(el('p', {
      class: 'muted',
      text: `指标 ${pick(data, 'field', view.field)}｜周期 ${pick(data, 'period', view.period)}｜参与聚合 ${fmtInt(data.comparable ?? data.sessions)} 次 / 已完成 ${fmtInt(data.sessions)} 次`,
    }));
    const group = data.comparison_group;
    if (group) trendHost.append(el('p', { class: 'muted', text:
      `当前分组：${group.protocol_label || group.protocol || DASH} · ${{ quick: '简短训练', full: '标准训练' }[group.training_mode] || group.training_mode || DASH} · ${group.source_kind === 'sim' ? '模拟数据' : group.source_kind === 'lsl' ? '实时设备' : '来源待确认'} · 时间倍率 ${fmtNum(group.time_scale, 2)}` }));
    if (points.length) {
      const chartHost = el('div');
      drawLineChart(chartHost, points.map((point) => ({ label: point.label, values: { value: point.value } })), {
        series: ['value'],
        colors: [SERIES_COLORS[0]],
        min: 0,
        max: view.field === 'stress' ? 10 : 1,
        percent: view.field !== 'stress',
        label: '跨会话趋势折线图',
      });
      trendHost.append(chartHost);
      trendHost.append(table([
        { title: '周期', render: (row) => row.label },
        { title: '均值', align: 'right', render: (row) => fmtNum(row.value, 3) },
      ], points));
    } else {
      trendHost.append(empty(view.field === 'stress'
        ? '暂无已完成现场会话的压力自评。' : '暂无可比趋势点（缺少该指标，或协议、来源与采集配置不一致）。'));
    }
    if (data.note) {
      trendHost.append(el('p', { class: 'muted', text: `不可比说明：${data.note}` }));
    }
    if (ledgerPoints.length) {
      trendHost.append(el('p', { class: 'muted', text: `台账点（JSONL）示例：${ledgerPoints.slice(-3).map((point) => `${point.label}=${fmtNum(point.value, 3)}`).join('，')}` }));
    }
  } catch (error) {
    trendHost.append(empty(`趋势加载失败：${describeError(error)}`));
    toast(describeError(error));
  }
}

export default render;
