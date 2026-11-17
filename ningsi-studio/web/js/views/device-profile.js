import { api } from '../api.js';
import { describeError } from '../store.js';
import { button, card, el, field, list } from '../util.js';

const UNIT_FACTORS = { uV: 1, mV: 1000, V: 1000000, nV: .001 };
const FIELD_LIMITS = { manufacturer: 160, model: 160, electrodes: 256, reference: 256, unit_basis: 1500, document_reference: 2000, notes: 2000 };
const option = (value, text) => el('option', { value, text });

/** Device declarations are scoped to the exact source key, never equipment certification. */
export function createDeviceProfile(ctx) {
  const source = el('select', { id: 'device-profile-source' }, [option('', '选择 EEG 数据源')]);
  const manual = el('input', { id: 'device-profile-key', type: 'text', placeholder: '例如 lsl:实际流名', maxLength: 120 });
  const status = el('p', { class: 'device-profile-status', role: 'status', text: '选择设备来源后查看或补充档案。' });
  const missing = el('p', { class: 'device-profile-missing' });
  const messages = el('p', { class: 'device-profile-error', role: 'alert' });
  const summary = el('div', { class: 'device-profile-summary' });
  const fields = {};
  const grid = el('div', { class: 'device-profile-fields' });
  for (const [key, label, placeholder] of [
    ['manufacturer', '设备厂家', '按设备资料填写'], ['model', '设备型号', '填写完整型号'],
    ['electrodes', '实际 EEG 电极', '例如 Fp1,Fp2；按实际配置填写'], ['reference', '参考电极', '填写实际参考位置'],
    ['srate', '标称采样率（Hz）', '按采集软件配置填写'], ['source_unit', '输入数值单位', ''],
    ['scale_to_uv', '换算倍率（µV/输入单位）', 'ADC 需要校准后的 µV/计数'],
    ['unit_basis', '单位与标定依据', '设备说明或校准记录'],
    ['document_reference', '设备资料出处', '文档名称、网址或资料编号'], ['notes', '佩戴与配置备注', '记录需要现场核对的事项'],
  ]) {
    const input = key === 'source_unit'
      ? el('select', { id: `device-profile-${key}` }, [option('', '待确认'), ...['uV', 'mV', 'V', 'nV', 'ADC'].map(unit => option(unit, unit === 'uV' ? 'µV' : unit === 'ADC' ? 'ADC 计数' : unit))])
      : el(key === 'notes' ? 'textarea' : 'input', { id: `device-profile-${key}`, type: key === 'notes' ? null : ['srate', 'scale_to_uv'].includes(key) ? 'number' : 'text',
        step: 'any', min: ['srate', 'scale_to_uv'].includes(key) ? '0' : null, placeholder, maxLength: FIELD_LIMITS[key] || 256 });
    fields[key] = input;
    grid.append(field(label, input));
  }
  fields.source_unit.onchange = () => {
    fields.scale_to_uv.value = UNIT_FACTORS[fields.source_unit.value] ?? '';
    fields.scale_to_uv.readOnly = fields.source_unit.value !== 'ADC';
  };
  const form = el('fieldset', { class: 'device-profile-form' }, [grid]);
  const save = button('保存设备档案', saveProfile, { primary: true });
  const editor = el('details', { class: 'device-profile-editor' }, [
    el('summary', { text: '填写或修改设备资料' }), form, save,
  ]);
  editor.hidden = true;
  const loadManual = button('读取该设备档案', () => selectSource(manual.value.trim()), { small: true });
  const reload = button('重新读取', () => selectSource(selected), { small: true });
  const node = card('设备档案', [
    el('p', { class: 'device-profile-intro', text: '按数据源分别保存实际硬件、导联和单位依据。资料属于操作员声明，保存后仍需现场核验。' }),
    el('div', { class: 'device-profile-source-row' }, [field('设备数据源', source), reload]),
    el('details', { class: 'device-profile-manual' }, [el('summary', { text: '配置暂未连接的设备' }),
      el('div', { class: 'row' }, [field('来源标识', manual), loadManual]),
      el('p', { class: 'muted', text: '使用采集程序发布的完整来源标识；保存离线档案不表示设备已连接。' }),
    ]), status, messages, summary, missing, editor,
  ], { class: 'device-profile-card' });
  let selected = '', sources = [], ticket = 0, saving = false, loading = false, disposed = false;
  const requested = ctx.params.get('device') || '';
  const alive = () => !disposed && !ctx.signal.aborted;
  ctx.onCleanup(() => { disposed = true; ticket++; });

  function setBusy() { save.disabled = saving || loading || !selected; form.disabled = saving || loading; source.disabled = saving; loadManual.disabled = saving; reload.disabled = saving || loading || !selected; }
  function fill(profile) {
    for (const [key, input] of Object.entries(fields)) input.value = profile[key] ?? '';
    fields.scale_to_uv.readOnly = fields.source_unit.value !== 'ADC';
  }
  function present(data) {
    fill(data.profile || {});
    summary.textContent = '';
    const profile = data.profile || {};
    for (const [label, value] of [['设备', [profile.manufacturer, profile.model].filter(Boolean).join(' · ')],
      ['导联 / 参考', [profile.electrodes, profile.reference].filter(Boolean).join(' / ')], ['单位', profile.source_unit]]) {
      summary.append(el('div', {}, [el('span', { text: label }), el('strong', { text: value || '待补充' })]));
    }
    const gaps = list(data.missing_fields).map(item => item.label || item.key).filter(Boolean);
    missing.textContent = gaps.length ? `待补资料：${gaps.join('、')}。` : '声明字段已齐备；设备、导联和标定仍需现场核验。';
    status.textContent = data.saved ? '档案已保存 · 资料待核验' : '尚未保存此设备的档案';
    editor.hidden = false; editor.open = !data.saved;
    if (data.read_error) messages.textContent = '档案未能正常读取，请核对资料后重新保存。';
  }
  function rebuildSources() {
    source.replaceChildren(option('', '选择 EEG 数据源'), ...sources.filter(row => row?.key && row.is_eeg !== false).map(row =>
      option(row.key, `${row.device || row.key}${row.simulated ? '（模拟来源）' : ''}`)));
    if (selected && !sources.some(row => row.key === selected)) source.append(option(selected, `${selected}（当前未发现）`));
    source.value = selected;
  }
  async function selectSource(value) {
    if (saving || !alive()) return;
    selected = value; manual.value = value; rebuildSources();
    const currentTicket = ++ticket;
    messages.textContent = ''; summary.textContent = ''; missing.textContent = ''; fill({});
    editor.hidden = true;
    if (!selected) { status.textContent = '选择设备来源后查看或补充档案。'; loading = false; setBusy(); return; }
    loading = true; status.textContent = '正在读取该设备的档案…'; setBusy();
    try {
      const response = await api.deviceProfile(selected);
      if (!alive() || currentTicket !== ticket) return;
      present(response.data || {});
    } catch (err) {
      if (!alive() || currentTicket !== ticket) return;
      status.textContent = '设备档案读取未完成'; messages.textContent = describeError(err);
    } finally { if (alive() && currentTicket === ticket) { loading = false; setBusy(); } }
  }
  source.onchange = () => selectSource(source.value);

  async function saveProfile() {
    if (saving || loading || !selected) return;
    const profile = {};
    messages.textContent = '';
    for (const [key, input] of Object.entries(fields)) {
      const value = String(input.value).trim();
      if (['srate', 'scale_to_uv'].includes(key)) {
        if (value && !(Number.isFinite(Number(value)) && Number(value) > 0)) { messages.textContent = '采样率和换算倍率应为有效正数。'; return; }
        profile[key] = value ? Number(value) : null;
      } else profile[key] = value;
    }
    if (profile.source_unit === 'ADC' && !profile.scale_to_uv) { messages.textContent = 'ADC 计数必须填写校准后的 µV/计数倍率。'; return; }
    saving = true; setBusy(); const currentTicket = ticket;
    try {
      const response = await api.saveDeviceProfile({ device: selected, profile });
      if (!alive() || currentTicket !== ticket) return;
      present(response.data || {});
    } catch (err) { if (alive() && currentTicket === ticket) messages.textContent = describeError(err); }
    finally { if (alive()) { saving = false; setBusy(); } }
  }
  setBusy();
  return { node, setSources(rows) {
    sources = list(rows); rebuildSources();
    if (!selected && !loading) {
      const initial = requested || sources.find(row => row.real === true && row.is_eeg !== false)?.key || '';
      if (initial) selectSource(initial);
    }
  } };
}
