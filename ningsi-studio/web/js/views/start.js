import { api } from '../api.js';
import { describeError } from '../store.js';
import { button, el, field } from '../util.js';
import { renderPreviewCard } from '../preview.js';
import { createSessionPreflight } from './session-preflight.js';
import { createBlinkCalibration } from './blink-calibration.js';

const option = (value, text) => el('option', { value, text });

export async function render(container, ctx) {
  const purpose = el('select', { id: 'purpose' }, [
    option('ten_min', '十分钟测试＋训练'), option('effective_time', '有效时长训练'),
    option('short', '评估专注力'), option('full', '评估并进行反馈训练'),
  ]);
  const participant = el('select', { id: 'participant' }, [option('', '新参与者（自动分配编号）')]);
  const alias = el('input', { id: 'alias', type: 'text', placeholder: '可留空', maxLength: 80 });
  const aliasField = field('参与者别名', alias);
  let participantRevision = 0;
  participant.onchange = () => { participantRevision += 1; aliasField.hidden = Boolean(participant.value); update(); };
  const device = el('select', { id: 'device' });
  const unit = el('select', { id: 'eeg-source-unit' }, [
    option('', '使用设备声明'), ...['uV','mV','V','nV','ADC'].map(u => option(u, u === 'ADC' ? 'ADC 计数' : u)),
  ]);
  const factor = el('input', { id: 'eeg-scale', type: 'number', min: 0, step: 'any' });
  const unitField = field('确认 EEG 源单位', unit, '设备未声明单位时，请按厂商配置或标定结果选择。');
  const factorField = field('校准倍率（µV/计数）', factor);
  const mode = el('select', { id: 'training-length' }, [
    option('quick', '2 段，约 4 分钟'), option('full', '4 段，约 8 分钟'),
  ]);
  const modeField = field('反馈训练', mode);
  const plan = el('p', { class: 'muted' });
  const message = el('p', { role: 'alert' });
  const deviceNote = el('p', { class: 'muted', role: 'status' });
  const readiness = el('span', { class: 'setup-readiness', role: 'status', text: '正在检查设备' });
  const previewHost = el('div');
  const preview = el('details', {}, [el('summary', { text: '预览脑电波形' }), previewHost]);
  let previewController;
  preview.ontoggle = () => {
    if (preview.open && !previewController) {
      previewController = renderPreviewCard(ctx, previewHost, { getSource: () => device.value });
    }
  };
  let profiles = [];
  let scanning = false;
  let submitting = false;
  let demoId;
  let subjectsReady = false;
  const scanButton = button('重新扫描', scan, { small: true });
  const start = button('开始评估', () => launch(false), { primary: true });
  const demo = button('运行模拟演示', () => launch(true));
  const preflight = createSessionPreflight(ctx, { onChange: () => update(false) });
  const choices = [
    ['ten_min', '轻量体验', '十分钟测试与训练', '快速了解本次状态，体验反馈训练。', '约 10 分钟'],
    ['effective_time', '质量中断时补足', '有效时长训练', '4 轮各累计 50 秒有效反馈；坏窗暂停累计，达到上限后如实报告数据不足。', '约 10 分钟起 · 补采上限约 17 分 20 秒'],
    ['short', '标准评估', '专注力综合评估', '结合脑电、量表与行为任务建立评估。', '按协议计算'],
    ['full', '完整流程', '评估与反馈训练', '完成综合评估后，继续进行专注训练。', '按协议计算'],
  ];
  const planCards = choices.map(([value, tag, title, detail, duration]) => {
    const input = el('input', { type: 'radio', name: 'assessment-plan', value, checked: value === purpose.value });
    input.onchange = () => { purpose.value = value; update(); };
    const time = el('span', { class: 'plan-choice__duration', text: duration });
    const tile = el('label', { class: 'plan-choice' }, [
      el('div', { class: 'plan-choice__top' }, [el('span', { text: tag }), input]),
      el('strong', { text: title }), el('p', { text: detail }), time,
    ]);
    return { value, input, time, tile };
  });
  const calibration = el('details', { class: 'setup-details' }, [
    el('summary', { text: '单位与标定设置' }), unitField, factorField,
  ]);
  const blinkCalibration = createBlinkCalibration(ctx, {
    getSettings: () => ({ participant: participant.value, device: device.value, eeg_calibration: calibrationSelection() }),
    ensureParticipant: async () => {
      if (participant.value) return;
      const requestedRevision = participantRevision;
      const requestedLabel = alias.value.trim();
      const { data } = await api.createSubject({ auto_id: true, label: requestedLabel || null });
      if (ctx.signal.aborted) return false;
      participant.add(option(data.public_id, data.public_id + (requestedLabel ? ' · ' + requestedLabel : '')));
      if (participantRevision !== requestedRevision || participant.value) return false;
      participant.value = data.public_id;
      participant.onchange();
    },
  });
  const protocolDetails = el('details', { class: 'setup-details' }, [
    el('summary', { text: '查看本方案的流程与时长' }), plan, modeField,
  ]);
  const form = el('fieldset', { class: 'assessment-form' }, [
    el('div', { class: 'setup-step-heading' }, [el('span', { text: '01' }), el('h2', { text: '选择本次方案' })]),
    el('div', { class: 'plan-choices' }, planCards.map(item => item.tile)), protocolDetails,
    el('div', { class: 'setup-step-heading' }, [el('span', { text: '02' }), el('h2', { text: '确认参与者与设备' })]),
    el('div', { class: 'setup-fields' }, [field('参与者', participant), aliasField]),
    el('div', { class: 'device-setup' }, [
      el('div', { class: 'device-setup__head' }, [el('strong', { text: '脑电设备' }), scanButton]),
      field('选择 EEG 数据流', device), deviceNote, calibration, preview, blinkCalibration.node,
    ]),
    preflight.node,
  ]);
  function calibrationSelection() {
    const documented = preflight.data?.calibration;
    if (preflight.data?.device === device.value && documented?.origin === 'manufacturer_manual') return documented;
    return unit.value ? { source_unit: unit.value,
      ...(unit.value === 'ADC' ? { scale_to_uv: Number.isFinite(factor.valueAsNumber) ? factor.valueAsNumber : null } : {}) } : null;
  }
  function update(refreshCheck = true, forceCheck = false) {
    const full = purpose.value === 'full';
    const profile = profiles.find(row => row.key === purpose.value);
    modeField.hidden = !full;
    const minutes = profile && Math.round((profile.total_sec_by_training_mode?.[mode.value] ?? profile.total_sec) / 60);
    plan.textContent = profile ? '约 ' + minutes + ' 分钟。记录睁眼和闭眼时的脑电，填写量表，完成数字判断和反应速度任务。'
      + (full ? '随后根据实时专注度反馈练习。' : '') : '';
    if (purpose.value === 'ten_min') plan.textContent = '10分钟采集：质检30秒、睁眼60秒、练习30秒、PVT3分钟、训练4分钟、训练后60秒。报告生成另计；质量不足时自动切换为质量反馈或固定引导，不追加基线时间。';
    if (purpose.value === 'effective_time') plan.textContent = '前后基线各累计60秒合格区间、每次补采上限3分钟；训练4轮各累计50秒有效反馈、每轮补采上限100秒，轮间休息10秒。质检30秒、PVT练习30秒和正式3分钟保持固定。采集约10分钟起、补采上限约17分20秒；设备连接与报告另计，设备读取等待按实际记录。信号不合格时暂停有效时间，达到上限仍不足会如实标记数据不足，可随时取消。';
    for (const item of planCards) {
      item.input.checked = item.value === purpose.value;
      const itemProfile = profiles.find(row => row.key === item.value);
      const seconds = itemProfile?.total_sec_by_training_mode?.[mode.value] ?? itemProfile?.total_sec;
      if (Number.isFinite(seconds) && item.value !== 'effective_time') item.time.textContent = `约 ${Math.round(seconds / 60)} 分钟`;
    }
    if (full || purpose.value === 'effective_time') protocolDetails.open = true;
    form.disabled = submitting;
    scanButton.disabled = scanning || submitting;
    preview.hidden = !device.value;
    const documented = preflight.data?.device === device.value && preflight.data?.calibration?.origin === 'manufacturer_manual';
    unitField.hidden = !device.value || documented;
    factorField.hidden = unitField.hidden || unit.value !== 'ADC';
    calibration.hidden = !device.value || documented;
    blinkCalibration.update();
    if (refreshCheck && !scanning && !submitting) preflight.refresh({
      device: device.value, protocol: purpose.value, participant: participant.value,
      eeg_calibration: unit.value ? calibrationSelection() : null,
    }, { force: forceCheck });
    const calibrationReady = unit.value !== 'ADC' || (Number.isFinite(factor.valueAsNumber) && factor.valueAsNumber > 0);
    readiness.textContent = scanning ? '正在扫描设备' : !device.value ? '等待连接脑电设备'
      : !calibrationReady ? '请补充有效的 ADC 校准倍率' : !subjectsReady || !profile ? '正在加载评估设置'
      : submitting ? '正在创建评估会话' : preflight.pending ? '正在核对评分配置'
      : !preflight.ready ? '请处理开始前的配置检查' : '配置已检查 · 开始后进行信号质检';
    readiness.dataset.ready = String(Boolean(device.value && calibrationReady && subjectsReady && profile && preflight.ready));
    start.disabled = submitting || scanning || !subjectsReady || !device.value || !profile
      || !preflight.ready || (unit.value === 'ADC' && !(Number.isFinite(factor.valueAsNumber) && factor.valueAsNumber > 0));
    start.textContent = submitting ? '正在启动…' : purpose.value === 'effective_time' ? '开始有效时长训练' : (full ? '开始评估与训练' : '开始评估');
    demo.disabled = submitting || !profiles.some(row => row.key === 'short');
  }
  purpose.onchange = mode.onchange = update;
  unit.onchange = factor.oninput = update;
  async function scan() {
    scanning = true;
    deviceNote.textContent = '正在扫描脑电设备…';
    update();
    const previous = device.value;
    const requested = previous || ctx.params.get('device') || '';
    try {
      const response = await api.devices(1.2);
      if (ctx.signal.aborted) return;
      const rows = response.data.sources.filter(row => (row.real || row.simulated) && row.is_eeg && row.usable !== false && row.key.startsWith('lsl:'));
      device.replaceChildren(...rows.map(row => option(row.key, (row.device || row.key) + (row.simulated ? '（模拟流）' : ''))));
      if (rows.some(row => row.key === requested)) device.value = requested;
      else if (requested) device.value = '';
      else device.value = rows.find(row => !row.simulated)?.key || '';
      if (!device.value) device.prepend(option('', rows.length ? '请选择脑电流' : '未发现脑电设备'));
      deviceNote.textContent = rows.length ? (requested && !rows.some(row => row.key === requested) ? '指定设备未发现，请确认采集程序或选择当前数据源。' : '')
        : '请在采集软件中开启 EEG 推流，然后重新扫描。';
    } catch (error) {
      device.replaceChildren(option('', '扫描失败'));
      deviceNote.textContent = describeError(error);
    } finally {
      scanning = false;
      if (!ctx.signal.aborted) {
        if (device.value !== previous) {
          unit.value = ''; factor.value = '';
          previewController?.selectionChanged();
        }
        update(true, true);
      }
    }
  }
  device.onchange = () => { unit.value = ''; factor.value = ''; previewController?.selectionChanged(); update(); };

  async function launch(isDemo) {
    if (submitting || (isDemo ? demo.disabled : start.disabled)) return;
    const checkedCalibration = !isDemo ? preflight.data?.calibration || calibrationSelection() : null;
    submitting = true;
    message.textContent = '';
    update();
    try {
      let id = isDemo ? demoId : participant.value;
      if (!id) {
        const { data } = await api.createSubject({ auto_id: true, label: isDemo ? '模拟演示' : alias.value.trim() || null });
        id = data.public_id;
        if (ctx.signal.aborted) return;
        if (isDemo) demoId = id;
        else {
          participant.add(option(id, id + (alias.value.trim() ? ' · ' + alias.value.trim() : '')));
          participant.value = id;
          participant.onchange();
        }
      }
      const { data } = await api.createSession({
        participant: id, create_subject: false, device: isDemo ? 'sim-bsense' : device.value,
        protocol: purpose.value, training_mode: isDemo ? 'quick' : mode.value,
        time_scale: isDemo ? 0.05 : 1,
        eeg_calibration: checkedCalibration,
        blink_calibration_token: isDemo ? null : blinkCalibration.token,
      });
      if (ctx.signal.aborted) return;
      ctx.store.setState({ currentSession: data.session, phases: data.phases, sessionRuntime: data.runtime });
      ctx.navigate('#/flow?session=' + data.session.uuid + '&subject=1');
    } catch (error) {
      message.textContent = describeError(error);
    } finally {
      submitting = false;
      if (!ctx.signal.aborted) update();
    }
  }
  container.append(el('div', { class: 'assessment-page' }, [
    el('header', { class: 'assessment-heading' }, [
      el('span', { class: 'eyebrow', text: '专注，从了解自己开始' }),
      el('h1', { text: '开启一次专注力评估' }),
      el('p', { text: '连接脑电设备，结合量表与行为任务，了解本次状态并进行反馈训练。' }),
    ]),
    el('div', { class: 'assessment-layout' }, [
      el('section', { class: 'card setup-card', 'aria-label': '评估设置' }, [form,
        el('div', { class: 'setup-submit' }, [el('div', {}, [readiness,
          el('p', { class: 'muted', text: '开始后将先检查信号质量，再进入引导流程。' })]), message, start]),
      ]),
      el('aside', { class: 'assessment-aside', 'aria-label': '评估说明与演示' }, [
        el('section', { class: 'journey-card' }, [
          el('span', { class: 'eyebrow', text: '本次体验' }), el('h2', { text: '让每一步都清晰' }),
          el('ol', { class: 'assessment-journey' }, [
            ['连接与质检', '确认设备与信号质量'], ['采集与评估', '建立基线，完成当前方案任务'],
            ['反馈与练习', '按所选方案进行反馈训练'], ['结果与回顾', '查看报告，追踪历史变化'],
          ].map(([title, note]) => el('li', {}, [el('strong', { text: title }), el('span', { text: note })]))),
          el('p', { class: 'journey-note', text: '信号质量不足时会暂停评分，并给出调整提示。' }),
        ]),
        el('section', { class: 'demo-card' }, [el('span', { class: 'demo-label', text: '无设备也能了解流程' }),
          el('h2', { text: '先体验，再开始' }),
          el('p', { text: '使用模拟脑电与自动作答，体验评估流程。演示记录单独标记。' }), demo,
        ]),
      ]),
    ]),
    el('p', { class: 'assessment-footnote', text: '脑电指数反映相对个人基线的变化；模型估计、自评与信号质量分别呈现。产品用于研究与训练，不能替代医学诊断。' }),
  ]));
  update();
  const loadSubjects = async () => {
    let page = 1, count = 0;
    while (!ctx.signal.aborted) {
      const { data } = await api.listSubjects({ limit: 100, page });
      if (ctx.signal.aborted) return;
      for (const row of data.items) participant.add(option(row.public_id, [row.public_id, row.label].filter(Boolean).join(' · ')));
      count += data.items.length;
      if (!data.items.length || count >= data.total) break;
      page++;
    }
    const requested = ctx.params.get('participant');
    if (requested && Array.from(participant.options).some(row => row.value === requested)) participant.value = requested;
    participant.onchange();
    subjectsReady = true;
    update();
  };
  const results = await Promise.allSettled([scan(), loadSubjects(), api.config().then(({ data }) => {
    profiles = data.profiles;
    update();
  })]);
  if (ctx.signal.aborted) return;
  const errors = results.filter(row => row.status === 'rejected');
  if (errors.length) message.append(errors.map(row => describeError(row.reason)).join('。'), ' ', button('重新加载', ctx.reload));
}
