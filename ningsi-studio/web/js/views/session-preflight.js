import { api } from '../api.js';
import { describeError } from '../store.js';
import { button, el, list } from '../util.js';

const MODEL_LABELS = {
  none: '未启用真人模型', compatible: '配置匹配 · 采集后检查信号',
  incompatible: '模型暂停 · 配置不匹配', unavailable: '模型暂不可用',
  simulation: '模拟来源 · 仅用于演示',
};
const CHECK_LABELS = { pass: '通过', warning: '需留意', fail: '待处理', pending: '采集后检查' };

/** Read-only configuration checks; quality and baseline remain acquisition-time gates. */
export function createSessionPreflight(ctx, { onChange = () => {}, onUseProfile = () => {} } = {}) {
  const body = el('div');
  const summary = el('span', { class: 'preflight-state', role: 'status', text: '等待选择设备' });
  const node = el('section', { class: 'session-preflight', 'aria-label': '本次评分与模型适配' }, [
    el('div', { class: 'preflight-heading' }, [el('h3', { text: '本次评分与模型适配' }), summary]), body,
  ]);
  let data = null, error = '', pending = false, lastBody = null, signature = '', ticket = 0, timer = null, disposed = false;
  const alive = () => !disposed && !ctx.signal.aborted;
  ctx.onCleanup(() => { disposed = true; ticket++; if (timer !== null) window.clearTimeout(timer); });

  function draw() {
    body.textContent = '';
    const model = data?.scoring?.attention_model;
    const relative = data?.scoring?.relative_indices;
    summary.textContent = pending ? '正在检查' : error ? '检查未完成' : !lastBody?.device ? '等待选择设备'
      : data?.ready ? '配置可用' : '需要处理配置';
    summary.dataset.state = pending ? 'pending' : error ? 'fail' : data?.ready ? 'pass' : data ? 'warning' : 'pending';
    body.append(el('div', { class: 'preflight-methods' }, [
      el('div', {}, [el('span', { text: '基础指标与反馈' }), el('strong', { text: '相对个人基线指数' }),
        el('small', { text: relative?.message || '有效信号与合格基线建立后，计算专注、放松与负荷指数。' })]),
      el('div', {}, [el('span', { text: '专注状态模型' }),
        el('strong', { text: model ? (MODEL_LABELS[model.status] || '适配状态待确认') : '选择设备后检查' }),
        model?.model_id ? el('small', { class: 'preflight-model-id', title: String(model.model_id), text: `模型版本 ${String(model.model_id).slice(0, 16)}` }) : null,
        el('small', { text: model?.message || '真人模型的估计单独保存，不替代基础指数和反馈目标。' })]),
    ]));
    body.append(el('p', { class: 'preflight-note', text: '本检查只核对配置。信号质量、基线及实时数据是否有效，需要开始采集后确认；训练反馈沿用相对基线专注指数。' }));
    if (error) body.append(el('p', { class: 'preflight-error', role: 'alert', text: `开始前检查失败：${error}` }));
    const checks = list(data?.checks);
    if (checks.length) {
      const unresolved = checks.filter(check => check.status === 'fail' || check.status === 'warning').length;
      body.append(el('details', { class: 'preflight-checks', open: checks.some(check => check.status === 'fail') }, [
        el('summary', { text: unresolved ? `查看配置检查 · ${unresolved} 项需处理或留意` : '查看配置与采集检查' }),
        el('ul', {}, checks.map(check => el('li', { dataset: { status: check.status || 'pending' } }, [
          el('span', { class: 'preflight-check-state', text: CHECK_LABELS[check.status] || '待确认' }),
          el('div', {}, [el('strong', { text: check.label || '配置检查' }), el('p', { text: check.message || '请核对设备设置。' })]),
        ]))),
      ]));
    }
    const actions = el('div', { class: 'preflight-actions' }, [
      el('a', { href: '#/devices' + (lastBody?.device ? `?device=${encodeURIComponent(lastBody.device)}` : ''), text: '设备连接与档案' }),
      el('a', { href: '#/models', text: '管理真人模型' }),
    ]);
    if (lastBody?.device) actions.append(button('重新检查', () => refresh(lastBody, { force: true, immediate: true }), { small: true, disabled: pending }));
    body.append(actions);
    const profile = data?.device_profile?.profile;
    if (profile?.source_unit && !pending) {
      body.append(el('div', { class: 'preflight-profile-unit' }, [
        el('p', { text: `该设备档案记录单位 ${profile.source_unit}${profile.scale_to_uv != null ? `，换算倍率 ${profile.scale_to_uv} µV/输入单位` : ''}。请核对本次实际配置后使用。` }),
        button('采用档案单位', () => onUseProfile({ source_unit: profile.source_unit, scale_to_uv: profile.scale_to_uv }), { small: true }),
      ]));
    }
  }

  async function run(bodyToCheck, currentTicket) {
    try {
      const response = await api.sessionPreflight(bodyToCheck);
      if (!alive() || currentTicket !== ticket) return;
      data = response.data || {};
    } catch (err) {
      if (!alive() || currentTicket !== ticket) return;
      error = describeError(err);
    } finally {
      if (alive() && currentTicket === ticket) { pending = false; draw(); onChange(); }
    }
  }

  function refresh(payload, { force = false, immediate = false } = {}) {
    const next = { device: payload?.device || '', protocol: payload?.protocol || 'ten_min',
      ...(payload?.participant ? { participant: payload.participant } : {}),
      eeg_calibration: payload?.eeg_calibration || null };
    const key = JSON.stringify(next);
    if (!force && key === signature) return;
    signature = key; lastBody = next; const currentTicket = ++ticket;
    if (timer !== null) { window.clearTimeout(timer); timer = null; }
    data = null; error = ''; pending = Boolean(next.device);
    draw(); onChange();
    if (!next.device) return;
    if (immediate) return run(next, currentTicket);
    timer = window.setTimeout(() => { timer = null; run(next, currentTicket); }, 250);
  }
  draw();
  return { node, refresh, get ready() { return data?.ready === true && !pending && !error; },
    get pending() { return pending; }, get data() { return data; }, get error() { return error; } };
}
