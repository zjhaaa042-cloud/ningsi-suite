/** Session-wide coverage stays visible when the timeline is filtered or paged. */
import { button, el, fmtInt, fmtPercent, fmtSeconds, list } from '../util.js';
import { drawHeatmap } from '../charts.js';

const PAGE_SIZE = 120;
const STATES = {
  scored: '可评分', baseline_pending: '信号合格 · 待基线', signal_limited: '信号受限',
  no_data: '等待数据', timing_issue: '时标受限', unscored: '原因未记录',
};

export const isScoredCell = cell => typeof cell?.score === 'number'
  && Number.isFinite(cell.score) && cell.score >= 0 && cell.score <= 1 && cell.index >= 0
  && cell.signal_usable !== false && (!cell.status || cell.status === 'scored');

export function heatmapCellState(cell) {
  if (isScoredCell(cell)) return 'scored';
  return cell?.status && cell.status in STATES && cell.status !== 'scored' ? cell.status : 'unscored';
}

/** Replayed SSE windows and a concurrent REST snapshot must not be double-counted. */
export function mergeHeatmapCells(existing, incoming) {
  const rows = list(existing).slice();
  const ids = new Map();
  const times = new Map();
  const finiteTime = cell => typeof cell?.t === 'number' && Number.isFinite(cell.t) ? cell.t.toFixed(6) : null;
  const record = (cell, index) => {
    if (cell?.window_id !== undefined && cell.window_id !== null) ids.set(String(cell.window_id), index);
    const t = finiteTime(cell); if (t !== null) times.set(t, index);
  };
  rows.forEach(record);
  for (const cell of list(incoming)) {
    const id = cell?.window_id === undefined || cell.window_id === null ? null : String(cell.window_id);
    const t = finiteTime(cell);
    let found = id !== null ? ids.get(id) : undefined;
    if (found === undefined && t !== null) {
      const candidate = times.get(t);
      // Time aliases allow old snapshots without ids; distinct explicit ids stay distinct.
      if (candidate !== undefined && (id === null || rows[candidate]?.window_id == null)) found = candidate;
    }
    if (found === undefined) { rows.push(cell); record(cell, rows.length - 1); }
    else { rows[found] = cell; record(cell, found); }
  }
  return rows.sort((a, b) => (typeof a?.t === 'number' ? a.t : Infinity) - (typeof b?.t === 'number' ? b.t : Infinity));
}

export function summarizeHeatmap(cells) {
  const rows = list(cells);
  const counts = Object.fromEntries(Object.keys(STATES).map(key => [key, 0]));
  const reasons = new Map();
  let signal = 0, unknown = 0;
  rows.forEach(cell => {
    const state = heatmapCellState(cell); counts[state] += 1;
    if (cell?.signal_usable === true || state === 'scored' || state === 'baseline_pending') signal += 1;
    else if (cell?.signal_usable == null) unknown += 1;
    const codes = state === 'scored' ? [] : list(cell?.reason_codes);
    const labels = list(cell?.reason_labels);
    [...new Set(codes)].forEach(code => {
      if (!reasons.has(code)) reasons.set(code, { code, label: labels[codes.indexOf(code)] || code, count: 0 });
      reasons.get(code).count += 1;
    });
  });
  return { total: rows.length, scored: counts.scored, signal, unknown, counts,
    reasons: [...reasons.values()].sort((a, b) => b.count - a.count) };
}

export function createStateHeatmap() {
  let data = {}, mode = 'all', page = null, detailsTouched = false;
  const node = el('div', { class: 'state-map' });
  const summaryHost = el('div', { class: 'state-map__summary' });
  const coverageHost = el('div');
  const messageHost = el('div', { class: 'state-map__message', attrs: { role: 'status' } });
  const reasonHost = el('div', { class: 'state-map__reasons' });
  const filter = el('select', { attrs: { 'aria-label': '热力图显示范围' }, onChange: () => { mode = filter.value; page = null; draw(); } }, [
    el('option', { value: 'all', text: '全部窗口' }),
    el('option', { value: 'scored', text: '仅可评分片段' }),
  ]);
  const details = el('details', { class: 'state-map__details', onToggle: () => { detailsTouched = true; } });
  const detailsLabel = el('summary', { text: '查看逐窗时间轴' });
  const chart = el('div');
  const range = el('p', { class: 'state-map__range' });
  const previous = button('上一页', () => { page = Math.max(0, activePage() - 1); draw(); }, { small: true });
  const next = button('下一页', () => { page = activePage() + 1; draw(); }, { small: true });
  const latest = button('最近片段', () => { page = null; draw(); }, { small: true });
  const pager = el('div', { class: 'state-map__pager' }, [range, el('div', { class: 'row' }, [previous, next, latest])]);
  details.append(detailsLabel, el('div', { class: 'state-map__tools' }, [
    el('label', {}, [document.createTextNode('显示 '), filter]),
    el('span', { text: '筛选不改变上方全时段覆盖率' }),
  ]), chart, pager);
  node.append(summaryHost, coverageHost, messageHost, reasonHost, details,
    el('p', { class: 'state-map__footnote', text: '覆盖按已记录窗口统计；窗口可重叠，未采集的行为阶段不补窗，不代表连续有效时长。窗口时间来自分析步进。未评分不等于低专注，信号合格也不等于已建立个人基线。' }));

  const selected = () => list(data.cells).filter(cell => mode !== 'scored' || isScoredCell(cell));
  const activePage = () => Math.min(page ?? Math.max(0, Math.ceil(selected().length / PAGE_SIZE) - 1), Math.max(0, Math.ceil(selected().length / PAGE_SIZE) - 1));
  const stat = (label, value, note) => el('div', {}, [el('span', { text: label }), el('strong', { text: value }), el('small', { text: note })]);
  const draw = () => {
    const rows = list(data.cells), summary = summarizeHeatmap(rows);
    summaryHost.textContent = '';
    summaryHost.append(
      stat('可评分覆盖', summary.total ? fmtPercent(summary.scored / summary.total) : '—', `${fmtInt(summary.scored)} / ${fmtInt(summary.total)} 窗`),
      stat('信号合格窗口', !summary.total ? '—' : summary.unknown ? `${fmtInt(summary.signal)} 窗已确认` : `${fmtInt(summary.signal)} / ${fmtInt(summary.total)}`, summary.unknown ? `${summary.unknown} 窗历史质量未记录` : summary.total ? '通过信号质量检查' : '等待分析窗口'),
      stat('已记录窗口', fmtInt(summary.total), summary.counts.baseline_pending ? `${summary.counts.baseline_pending} 窗信号合格，等待基线` : '包含所有受限与未评分窗口'),
    );
    coverageHost.textContent = '';
    const coverage = el('div', { class: 'state-map__coverage', attrs: { role: 'img', 'aria-label': `全时段：${summary.scored}/${summary.total} 窗可评分` } });
    const keys = Object.keys(STATES).filter(key => summary.counts[key]);
    const availability = list(data.availability_legend);
    keys.forEach(key => {
      const segment = el('span', { class: `state-map__segment state-map__segment--${key}`, title: `${STATES[key]}：${summary.counts[key]} 窗`,
        style: `flex:${summary.counts[key]}` });
      const color = key === 'scored' ? null : availability.find(item => (item.status || item.key) === key)?.color;
      if (color) segment.style.backgroundColor = color;
      coverage.append(segment);
    });
    if (summary.total) coverageHost.append(coverage, el('div', { class: 'state-map__coverage-labels' }, keys.map(key => el('span', { text: `${STATES[key]} ${summary.counts[key]} 窗` }))));
    messageHost.textContent = '';
    let heading, message;
    if (!summary.total) {
      heading = '等待首个分析窗口'; message = '收到足够样本后显示采集覆盖；建立合格个人基线后呈现状态分档。';
    } else if (!summary.scored && summary.counts.baseline_pending) {
      heading = '已收到合格信号，状态评分等待个人基线';
      message = '这些窗口保留为“信号合格 · 待基线”，不归为低质量。请按会话流程完成基线；若本次基线未通过，需要调整后重新建立。';
    } else if (!summary.scored && (summary.counts.signal_limited || summary.counts.timing_issue || summary.counts.no_data)) {
      heading = '已记录采集情况，暂未形成可评分片段';
      message = '先查看下方主要受限原因，核对设备连接、电极接触与参考配置。质量通过并建立个人基线后，状态颜色会自动出现。';
    } else if (!summary.scored) {
      heading = '本次暂无可评分片段'; message = '历史记录未保存完整的未评分原因，无法判定为低质量。可在报告查看已保存的采集与基线证据。';
    } else {
      heading = `已识别 ${summary.scored} 个可评分窗口`;
      message = '全部窗口保留未评分记录。可切换“仅可评分片段”集中回看；各窗仍标注对应的分析窗口时间。';
    }
    messageHost.append(el('strong', { text: heading }), el('p', { text: message }));
    reasonHost.textContent = '';
    if (summary.reasons.length) {
      reasonHost.append(el('span', { class: 'state-map__reason-title', text: '主要未评分原因' }),
        ...summary.reasons.slice(0, 3).map(reason => el('span', { class: 'state-map__reason', text: `${reason.label} · ${reason.count} 窗` })),
        el('small', { text: '同窗可有多个原因' }));
    }
    if (!detailsTouched) details.open = summary.scored > 0;
    detailsLabel.textContent = summary.scored ? '状态时间轴与逐窗记录' : '查看逐窗记录';
    const filtered = selected(), current = activePage(), start = current * PAGE_SIZE, shown = filtered.slice(start, start + PAGE_SIZE);
    if (mode === 'scored' && !shown.length) {
      chart.textContent = ''; chart.append(el('p', { class: 'empty', text: '此会话没有可评分片段；切回“全部窗口”查看采集情况。' }));
    } else {
      drawHeatmap(chart, shown, { legend: summary.scored ? data.legend : [], availabilityLegend: data.availability_legend,
        missingColor: data.missing_color, label: mode === 'scored' ? '仅可评分窗口，标注分析窗口时间' : '按采集顺序排列的状态窗口' });
    }
    range.textContent = shown.length ? `窗口时间（分析步进） ${fmtSeconds(shown[0].t)} — ${fmtSeconds(shown[shown.length - 1].t)} · 第 ${start + 1}–${start + shown.length} / ${filtered.length} 窗`
      : '暂无窗口';
    previous.disabled = current === 0; next.disabled = (current + 1) * PAGE_SIZE >= filtered.length;
    latest.disabled = current === Math.max(0, Math.ceil(filtered.length / PAGE_SIZE) - 1);
    pager.hidden = !filtered.length;
  };
  draw();
  return { node, update(nextData = {}) { data = nextData; draw(); } };
}
