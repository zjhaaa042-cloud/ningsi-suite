import { api } from '../api.js';
import { describeError } from '../store.js';
import { button, el } from '../util.js';

export function createBlinkCalibration(ctx, { getSettings, ensureParticipant }) {
  let token = null;
  let ready = false;
  let count = 0;
  let minimum = 8;
  let feedback = '';
  let signature = '';
  let revision = 0;
  let busy = false;
  let poll;
  const status = el('p', { role: 'status', text: '可选：为本次参与者建立个人眨眼模板。' });
  const begin = button('开始个人眨眼校准', () => action('start'));
  const mark = button('已眨眼，记录这一次', () => action('mark'), { disabled: true });
  const finish = button('结束并使用校准', () => action('finish'), { disabled: true });
  const clear = button('取消校准', () => action('clear'), { small: true, disabled: true });
  const node = el('details', { class: 'setup-details' }, [
    el('summary', { text: '减少眨眼干扰（双通道个人校准）' }),
    el('p', { text: '保持佩戴和接触稳定。自然短眨后等约 0.8 秒再点记录，每次间隔至少 3 秒。建议记录 15–20 次。' }),
    el('p', { class: 'muted', text: '模板仅用于当前参与者和设备。换人、重新佩戴或设备重连后请重新校准。校准不会训练专注分类器；残留干扰或接触异常时暂停更新分数。' }),
    el('div', { class: 'actions' }, [begin, mark, finish, clear]), status,
  ]);
  function update() {
    const settings = getSettings();
    const current = JSON.stringify(settings);
    if (signature !== current) {
      if (signature) {
        reset();
        status.textContent = '参与者、设备或单位已改变，请重新校准。';
      }
      revision += 1;
      signature = current;
    }
    node.hidden = !settings.device?.startsWith('lsl:');
    begin.disabled = busy || !settings.device;
    mark.disabled = busy || !token || ready;
    finish.disabled = busy || !token || ready || count < minimum;
    clear.disabled = busy || !token;
  }
  function reset() {
    token = null; ready = false;
    count = 0; feedback = '';
    window.clearInterval(poll); poll = null;
  }
  function isCurrent(requestRevision) {
    if (ctx.signal.aborted) return false;
    update();
    return revision === requestRevision;
  }
  function showStatus(data) {
    count = data.count;
    minimum = data.minimum_events ?? minimum;
    status.textContent = ready ? `个人眨眼校准已就绪，模板使用 ${data.template_events ?? data.count} 次一致的短眨眼。`
      : data.state === 'cleared' ? '已取消；未校正的可疑眨眼仍会暂停分数更新。'
      : feedback || `已通过单次筛查 ${count} 次。至少 ${minimum} 次形状一致的短眨眼才能生成模板。`;
  }
  function showRetry(error) {
    if (error.detail?.retryable !== true) return false;
    count = error.detail.accepted_count;
    feedback = `${error.message} 已保留 ${count} 次通过单次筛查的标记。`;
    status.textContent = feedback;
    return true;
  }
  async function checkStatus() {
    if (busy || ctx.signal.aborted) return;
    update();
    if (!token) return;
    const requestRevision = revision;
    const requestToken = token;
    const settings = getSettings();
    busy = true; update();
    try {
      const { data } = await api.blinkCalibration({ ...settings, action: 'status', token: requestToken });
      if (!isCurrent(requestRevision) || token !== requestToken) return;
      ready = data.state === 'ready';
      showStatus(data);
    } catch (error) {
      if (!isCurrent(requestRevision) || token !== requestToken) return;
      if (!showRetry(error)) {
        if ([404, 410, 422].includes(error.status)) reset();
        status.textContent = describeError(error);
      }
    } finally { busy = false; if (!ctx.signal.aborted) update(); }
  }
  async function action(kind) {
    if (busy) return;
    busy = true; update();
    let requestRevision;
    try {
      if (kind === 'start' && await ensureParticipant() === false) return;
      update();
      if (ctx.signal.aborted) return;
      requestRevision = revision;
      const settings = getSettings();
      if (kind === 'start') reset();
      const { data } = await api.blinkCalibration({ ...settings, action: kind, token });
      if (!isCurrent(requestRevision)) return;
      token = data.token || null;
      ready = data.state === 'ready';
      feedback = '';
      if (token && !poll) poll = window.setInterval(checkStatus, 2000);
      if (!token) { window.clearInterval(poll); poll = null; }
      showStatus(data);
    } catch (error) {
      if (!ctx.signal.aborted && (requestRevision === undefined || isCurrent(requestRevision))) {
        if (!showRetry(error)) status.textContent = describeError(error);
      }
    } finally { busy = false; if (!ctx.signal.aborted) update(); }
  }
  ctx.signal.addEventListener('abort', reset, { once: true });
  return { node, update, get token() { return ready && signature === JSON.stringify(getSettings()) ? token : null; } };
}
