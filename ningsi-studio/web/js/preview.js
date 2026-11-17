/**
 * 「设备实时预览」共用组件：**不建会话**也能看真实脑电。
 *
 * 为什么单独抽出来：设备状态页（#/devices）和"未选会话"的实时监测页（#/live）都要这块能力，
 * 逻辑必须一致，否则两处会各写一套轮询与文案。
 *
 * 产品规则（2026-10-07，用户明确）：
 * - 没有脑机信号时**不使用仿真**：这里只认检测到的实时源，检测不到就显示"无信号"并给接入指引；
 * - 仿真源可以被预览，但必须由人**显式**点「预览仿真源（仅演示）」；
 * - 预览只读实时流（调理 + 质检判定），**不落库、不出指标**——它不代表一次会话。
 */
import { api } from './api.js';
import { button, card, el, list, pick } from './util.js';
import { drawMultiChannelEeg } from './charts.js';

const POLL_MS = 1200;

/**
 * 渲染预览卡。
 * @param {object} ctx 视图上下文（用 ctx.signal.aborted 判断卸载）
 * @param {HTMLElement} host 容器
 */
export function renderPreviewCard(ctx, host, options = {}) {
  const statusLine = el('p', { class: 'muted', text: '正在探测设备…' });
  const deviceLine = el('p', { class: 'mono', text: '' });
  const qualityLine = el('p', { class: 'muted', text: '' });
  const waveHost = el('div');
  const actionHost = el('div', { class: 'row', style: 'margin:10px 0' });
  let timer = null;
  let running = false;
  let busy = false;
  let generation = 0;
  let pendingRefresh = null;
  const selectedSource = () => options.getSource ? options.getSource() : null;
  const relevant = payload => !options.getSource || payload?.source === selectedSource();
  const current = version => !ctx.signal.aborted && version === generation;

  const stopPolling = () => {
    if (timer !== null) {
      window.clearInterval(timer);
      timer = null;
    }
  };
  ctx.onCleanup(() => { generation++; stopPolling(); });

  const startPolling = () => {
    if (timer === null) timer = window.setInterval(refresh, POLL_MS);
  };

  const paint = (payload) => {
    const selected = selectedSource();
    if (options.getSource && payload && payload.active && payload.source !== selected) {
      statusLine.textContent = '当前预览与所选设备不同，请点击“预览所选设备”重新确认。';
      waveHost.textContent = ''; qualityLine.textContent = ''; deviceLine.textContent = '';
      return;
    }
    const active = payload && payload.active === true;
    if (!active) {
      statusLine.textContent = payload && payload.note
        ? payload.note
        : '未开始预览：点击下方按钮读取当前检测到的脑电信号。';
      deviceLine.textContent = '';
      qualityLine.textContent = '';
      // 卡片副标题已经写明"只读、不创建会话、不写入数据库"，这里不再重复第三遍
      waveHost.textContent = '';
      return;
    }
    const live = payload.live !== false && payload.no_signal !== true;
    statusLine.textContent = live
      ? `正在接收信号：${payload.device || payload.source}（${payload.simulated || payload.kind === 'sim' ? '仿真源，仅演示' : '实时设备'}）`
      : `无信号：数据源 ${payload.source} 已连接但最近没有样本`
        + (payload.seconds_since_last === null || payload.seconds_since_last === undefined
          ? '' : `（最近一个样本 ${payload.seconds_since_last}s 前）`);
    // 缓冲点数 / 调理口径属于排障细节：正文只留采样率与通道，细节进 title（鼠标悬停可见）
    deviceLine.textContent = `采样率 ${payload.srate || '—'} Hz｜通道 ${payload.channels || '—'}`;
    deviceLine.title = [
      payload.buffered_samples === null || payload.buffered_samples === undefined
        ? null : `缓冲 ${payload.buffered_samples} 点`,
      payload.conditioning && payload.conditioning.spec ? `调理 ${payload.conditioning.spec}` : null,
    ].filter(Boolean).join('｜');

    if (payload.window_error) {
      statusLine.textContent = payload.window_error;
      qualityLine.textContent = ''; waveHost.textContent = '';
      return;
    }
    const samples = list(payload.samples);
    if (!samples.length) {
      waveHost.textContent = '';
      if (Object.hasOwn(payload, 'samples')) {
        waveHost.append(el('p', { class: 'muted', text: '这一窗没有取到样本。' }));
      }
    } else {
      // 先清掉上一次的"没取到样本"提示或旧画布，否则会残留在新波形上方
      waveHost.textContent = '';
      const frame = {
        srate: payload.srate,
        window_sec: payload.seconds,
        channels: samples.map((channel, index) => ({
          label: list(payload.channel_labels)[index] || `CH${index + 1}`,
          samples: channel,
          peak: payload.peak_uv,
        })),
      };
      drawMultiChannelEeg(waveHost, frame, { srate: payload.srate, windowSec: payload.seconds, channelHeight: 110 });
    }
    const verdict = pick(payload, 'quality', null);
    const note = pick(payload, 'quality_note', null);
    if (verdict) {
      const ok = verdict.ok;
      const label = ok === true ? '可用'
        : (ok === false ? `不可用（${list(verdict.reasons).join('、')}）` : '样本不足，暂不判定');
      qualityLine.textContent = `本窗质检：${label}`
        + `｜峰峰值 ${payload.peak_uv} µV｜均值标准差 ${payload.std_uv} µV`
        + (note ? `｜${note}` : '');
      const blink = verdict.metrics?.artifact_control?.blink_control;
      if (blink?.corrected_events > 0) qualityLine.textContent += `｜已局部校正 ${blink.corrected_events} 次眨眼`;
    } else {
      qualityLine.textContent = '';
    }
  };

  const showState = payload => {
    paint(payload);
    running = payload?.active === true && relevant(payload);
    if (running) startPolling(); else stopPolling();
  };
  const showError = message => {
    paint({ active: false, note: message });
  };
  const refresh = async () => {
    if (busy || pendingRefresh?.generation === generation) return;
    const request = { generation };
    pendingRefresh = request;
    try {
      const response = running ? await api.devicePreviewWindow() : await api.devicePreview();
      if (!current(request.generation)) return;
      const wasRunning = running;
      showState(response.data);
      if (running !== wasRunning) renderActions();
    } catch (error) {
      if (!current(request.generation)) return;
      showError(`读取预览失败：${error.message || error}`);
      startPolling();
    } finally {
      if (pendingRefresh === request) pendingRefresh = null;
    }
  };

  // A lost POST/DELETE reply leaves the server state unknown. Read it back
  // before changing the controls, and keep polling if that read also fails.
  const reconcileFailure = async (version, message) => {
    if (!current(version)) return;
    showError(message);
    try {
      const { data } = await api.devicePreview();
      if (!current(version)) return;
      showState(data);
      statusLine.textContent = message + ' ' + statusLine.textContent;
    } catch {
      if (current(version)) {
        showError(message + ' 暂时无法确认后台状态，正在重试。');
        startPolling();
      }
    }
  };
  const finishAction = version => {
    busy = false;
    if (ctx.signal.aborted) return;
    renderActions();
    if (version !== generation) refresh();
  };
  const start = async (source) => {
    if (busy) return;
    const version = ++generation;
    stopPolling();
    busy = true; renderActions();
    showError('正在启动预览…');
    try {
      const response = await api.startDevicePreview(source);
      if (!current(version)) return;
      showState(response.data);
    } catch (error) {
      await reconcileFailure(version, `启动请求未确认：${error.message || error}`);
    } finally {
      finishAction(version);
    }
  };

  const stop = async () => {
    if (busy) return;
    const version = ++generation;
    stopPolling();
    busy = true;
    renderActions();
    showError('正在停止预览…');
    try {
      await api.stopDevicePreview();
      if (current(version)) showState({ active: false, note: '预览已停止。' });
    } catch (error) {
      await reconcileFailure(version, `停止请求未确认：${error.message || error}`);
    } finally {
      finishAction(version);
    }
  };

  const renderActions = () => {
    actionHost.textContent = '';
    if (options.getSource) {
      actionHost.append(button(busy ? '正在连接…' : '预览所选设备', () => start(selectedSource()),
        { primary: true, small: true, disabled: busy || !selectedSource() }));
      if (running) actionHost.append(button('停止预览', stop, { small: true, disabled: busy }));
      return;
    }
    if (running) {
      actionHost.append(
        button('停止预览', stop, { small: true, disabled: busy }),
      );
      return;
    }
    actionHost.append(
      button('开始预览（自动选检测到的设备）', () => start(null), { primary: true, small: true, disabled: busy }),
      button('预览仿真源（仅演示）', () => start('sim-bsense'), { small: true, disabled: busy }),
    );
  };

  renderActions();
  host.append(card('设备实时预览', el('div', {}, [
    el('p', { class: 'muted', text: '不开始会话也能看当前脑电信号：读取检测到的设备，做与报告一致的真机调理，'
      + '并给出这一窗的质检判定。' }),
    actionHost,
    deviceLine,
    statusLine,
    waveHost,
    qualityLine,
  ]), { sub: '只读、不创建会话、不写入数据库' }));
  refresh();
  return { stopPolling, selectionChanged() {
    generation++; stopPolling(); running = false;
    paint({ active: false, note: selectedSource() ? '设备已选择，请预览确认波形与信号质量。' : '请先选择设备。' });
    renderActions();
  } };
}
