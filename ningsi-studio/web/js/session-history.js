import { api, historyArtifactUrl } from './api.js';

export function sessionPresentationUrl(row) {
  if (row.archived) return historyArtifactUrl(row.presentation_url);
  return row.has_report ? api.presentationUrl(row.uuid) : null;
}

export function sessionSourceLabel(row) {
  if (row?.origin_label) return row.origin_label;
  const key = String(row?.source || row?.device || '').toLowerCase();
  const kind = row?.source_kind || row?.runtime?.source_kind;
  const note = row?.source_note || row?.runtime?.source_note || '';
  if (key.startsWith('replay:') || kind === 'replay') return '公开录波回放';
  if (row?.simulated === true || kind === 'sim' || kind === 'simulation'
      || /(^|[^a-z])(sim|simulation|synthetic|demo)([^a-z]|$)/i.test(key)
      || /模拟|仿真/.test(note)) return '模拟数据';
  if (key.startsWith('lsl:')) return 'LSL 数据';
  return key ? (row.source || row.device) : '来源未记录';
}
