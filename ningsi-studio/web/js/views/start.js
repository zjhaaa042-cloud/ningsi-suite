import { api } from '../api.js';
import { describeError } from '../store.js';
import { button, card, el, field } from '../util.js';
import { renderPreviewCard } from '../preview.js';

const option = (value, text) => el('option', { value, text });

export async function render(container, ctx) {
  const purpose = el('select', { id: 'purpose' }, [
    option('ten_min', '十分钟测试＋训练'), option('short', '评估专注力'), option('full', '评估并进行反馈训练'),
  ]);
  const participant = el('select', { id: 'participant' }, [option('', '新参与者（自动分配编号）')]);
  const alias = el('input', { id: 'alias', placeholder: '可留空', maxLength: 80 });
  const aliasField = field('参与者别名', alias);
  participant.onchange = () => { aliasField.hidden = Boolean(participant.value); };
  const device = el('select', { id: 'device' });
  const mode = el('select', { id: 'training-length' }, [
    option('quick', '2 段，约 4 分钟'), option('full', '4 段，约 8 分钟'),
  ]);
  const modeField = field('反馈训练', mode);
  const plan = el('p', { class: 'muted' });
  const message = el('p', { role: 'alert' });
  const deviceNote = el('p', { class: 'muted', role: 'status' });
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
  const form = el('fieldset', { class: 'assessment-form' }, [
    field('检测目的', purpose), plan, modeField, field('参与者', participant), aliasField,
    field('脑电设备', device), deviceNote, el('div', { class: 'row' }, [scanButton]), preview,
  ]);
  function update() {
    const full = purpose.value === 'full';
    const profile = profiles.find(row => row.key === purpose.value);
    modeField.hidden = !full;
    const minutes = profile && Math.round((profile.total_sec + (full ? 30 + (mode.value === 'full' ? 240 : 0) : 0)) / 60);
    plan.textContent = profile ? '约 ' + minutes + ' 分钟。记录睁眼和闭眼时的脑电，填写量表，完成数字判断和反应速度任务。'
      + (full ? '随后根据实时专注度反馈练习。' : '') : '';
    if (purpose.value === 'ten_min') plan.textContent = '10分钟采集：质检30秒、睁眼60秒、练习30秒、PVT3分钟、训练4分钟、训练后60秒。报告生成另计；质量不足时自动切换为质量反馈或固定引导，不追加基线时间。';
    form.disabled = submitting;
    scanButton.disabled = scanning || submitting;
    preview.hidden = !device.value;
    start.disabled = submitting || scanning || !subjectsReady || !device.value || !profile;
    start.textContent = submitting ? '正在启动…' : (full ? '开始评估与训练' : '开始评估');
    demo.disabled = submitting || !profiles.some(row => row.key === 'short');
  }
  purpose.onchange = mode.onchange = update;
  async function scan() {
    scanning = true;
    deviceNote.textContent = '正在扫描脑电设备…';
    update();
    const previous = device.value;
    try {
      const response = await api.devices(1.2);
      if (ctx.signal.aborted) return;
      const rows = response.data.sources.filter(row => row.real && row.is_eeg && row.usable !== false && row.key.startsWith('lsl:'));
      device.replaceChildren(...rows.map(row => option(row.key, (row.device || row.key) + (row.simulated ? '（模拟流）' : ''))));
      if (rows.some(row => row.key === previous)) device.value = previous;
      else device.value = rows.find(row => !row.simulated)?.key || '';
      if (!device.value) device.prepend(option('', rows.length ? '请选择脑电流' : '未发现脑电设备'));
      deviceNote.textContent = rows.length ? '' : '请在采集软件中开启 EEG 推流，然后重新扫描。';
    } catch (error) {
      device.replaceChildren(option('', '扫描失败'));
      deviceNote.textContent = describeError(error);
    } finally {
      scanning = false;
      if (!ctx.signal.aborted) {
        if (device.value !== previous) previewController?.selectionChanged();
        update();
      }
    }
  }
  device.onchange = () => { previewController?.selectionChanged(); update(); };

  async function launch(isDemo) {
    if (submitting || (isDemo ? demo.disabled : start.disabled)) return;
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
  container.append(el('div', { class: 'wizard stack' }, [
    el('h1', { text: '专注力评估' }),
    el('p', { text: '结合脑电、量表和按键表现，查看本次专注度、放松度与认知负荷。' }),
    card(null, el('div', { class: 'stack' }, [form, message, el('div', { class: 'row' }, [start])])),
    card('无设备演示', el('div', { class: 'stack' }, [
      el('p', { class: 'muted', text: '用模拟脑电和自动作答生成一份示例报告，单独保存演示记录。' }), demo,
    ])),
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
