import { api } from '../api.js';
import { describeError } from '../store.js';
import { button, card, el, field, fmtTime } from '../util.js';

function rating(id, label, anchors) {
  const select = el('select', { id });
  select.append(el('option', { value: '', text: '未填写' }));
  for (let n = 0; n <= 10; n += 1) {
    select.append(el('option', { value: String(n), text: anchors[n] ? `${n} · ${anchors[n]}` : String(n) }));
  }
  return { select, node: field(label, select) };
}

const REASONS = {
  simulation_source: '仿真信号',
  quality_rejected: '脑电质量未通过',
  unconfirmed_input_units: '设备单位或换算倍率尚未确认',
  requires_fp1_fp2: '需要 Fp1、Fp2 双通道',
  requires_250hz: '需要 250 Hz 采样率',
  invalid_band_features: '频带特征不完整',
  model_unavailable: '模型文件未就绪或校验失败',
  eeg_quality_failed: '脑电质量未通过',
};

export function stateCheckinPanel(uuid, refresh) {
  const stress = rating('state-stress', '现在感觉多大压力？', { 0: '没有压力', 10: '压力极大' });
  const valence = rating('state-valence', '现在的心情愉悦程度？', { 0: '非常不愉快', 5: '中性', 10: '非常愉快' });
  const message = el('p', { attrs: { role: 'status' } });
  const recent = el('p', { class: 'muted' });
  const predictions = el('div');
  const hint = el('p', { class: 'muted' });
  let canSubmit = false;
  let pending = false;
  let request = null;
  const submit = button('记录本人自评', async () => {
    const value = select => select.value === '' ? null : Number(select.value);
    const answers = { stress: value(stress.select), valence: value(valence.select) };
    if (answers.stress === null && answers.valence === null) {
      message.textContent = '请至少选择一项。';
      return;
    }
    if (!canSubmit || pending) return;
    if (!request || answers.stress !== request.stress || answers.valence !== request.valence) {
      request = { ...answers, request_id: crypto.randomUUID().replaceAll('-', '') };
    }
    pending = true;
    submit.disabled = true;
    try {
      const { data } = await api.submitStateCheckin(uuid, request);
      message.textContent = data.high_stress
        ? '已记录。你刚才自评的压力较高，可以暂停任务并稍作休息。' : '已记录本次自评。';
      stress.select.value = '';
      valence.select.value = '';
      request = null;
      await refresh();
    } catch (error) {
      message.textContent = describeError(error);
    } finally {
      pending = false;
      submit.disabled = !canSubmit;
    }
  });
  submit.disabled = true;
  const form = el('div', {}, [el('div', { class: 'stack' }, [stress.node, valence.node, submit, message])]);
  form.hidden = true;
  const node = card('当下压力与情绪', [predictions, recent, hint, form], {
    sub: '脑电模型在公开数据上训练，本机真人表现待验证。情绪效价表示愉快与不愉快的方向。',
  });
  return {
    node,
    update({ open, checkins, model }) {
      if (typeof open === 'boolean') {
        canSubmit = open;
        form.hidden = !open;
        submit.disabled = pending || !open;
        hint.textContent = open ? '可选填一项或两项。压力自评达到 7/10 时立即提醒。' : '可在量表、监测和训练阶段记录本人当下感受。';
      }
      if (checkins) {
        const last = checkins.records.at(-1);
        recent.textContent = last
          ? [fmtTime(last.recorded_at), last.stress === null ? null : `自报压力 ${last.stress}/10`,
             last.valence === null ? null : `自报愉悦程度 ${last.valence}/10`].filter(Boolean).join(' · ')
          : '本次尚无本人自评。';
      }
      if (model !== undefined) {
        predictions.textContent = '';
        for (const [key, label] of [['stress', '压力'], ['valence', '情绪效价']]) {
          const item = model?.[key];
          const result = item?.available
            ? '已记录，完成后可在报告中核对'
            : item ? (REASONS[item.reason] || item.reason) : '等待可用监测窗';
          predictions.append(el('p', { text: `脑电${label}研究估计 · ${result}` }));
        }
      }
    },
  };
}
