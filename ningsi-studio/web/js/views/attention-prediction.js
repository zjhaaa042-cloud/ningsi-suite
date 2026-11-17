import { card, el, fmtNum, fmtPercent, table } from '../util.js';

const REASONS = {
  real_timestamped_eeg_required: '需要真实设备提供带时间戳的连续脑电。',
  simulation_source: '当前为仿真数据，暂停真人模型推理。',
  timestamp_processing_mismatch: '采集时间戳处理方式与模型要求不一致，请启用模型后新建会话。',
  signal_stale: '等待设备恢复连续信号。',
  device_channels_units_mismatch: '当前设备、通道、采样率或单位与训练记录不匹配。',
  insufficient_raw_history: '正在积累连续脑电。',
  baseline_invalid: '需要先完成有效的睁眼基线。',
  insufficient_usable_eeg: '当前时间段的合格脑电不足。',
  lookback_crosses_phase: '正在积累当前阶段的脑电，请稍候。',
};
const NAMES = { rel_theta: 'θ 相对功率', rel_alpha: 'α 相对功率', rel_beta: 'β 相对功率',
  log_beta_theta: 'β/θ 对数比值', log_alpha_beta: 'α/β 对数比值', log_theta_alpha: 'θ/α 对数比值' };

export function attentionPredictionCard(prediction) {
  if (!prediction || !Object.keys(prediction).length) return null;
  const body = [];
  if (!prediction.available) {
    const reasons = String(prediction.reason || '').split('|');
    body.push(el('p', { text: '暂不可评估' }), el('p', { class: 'muted', text: reasons.map(r => REASONS[r] || '当前输入未满足模型要求，请检查模型与采集配置。').join(' ') }));
  } else {
    body.push(el('p', { class: 'metric-value', text: `内部走神概率 ${fmtPercent(prediction.mind_wandering_probability, 1)}` }),
      el('p', { text: `模型判断：${prediction.prediction === 'internal_mind_wandering' ? '内部走神倾向' : '当前任务专注倾向'}；阈值 ${fmtNum(prediction.threshold, 3)}。` }));
    const top = [...(prediction.contributions || [])].sort((a, b) => Math.abs(b.log_odds_contribution) - Math.abs(a.log_odds_contribution)).slice(0, 3);
    body.push(table([{ title: '主要特征依据', field: 'feature' }, { title: '对本次模型判断的贡献', field: 'direction' }],
      top.map(row => ({ feature: NAMES[row.feature] || row.feature,
        direction: row.log_odds_contribution >= 0 ? '提高走神判断' : '降低走神判断' }))));
    body.push(el('p', { class: 'muted', text: '依据当前时刻前 12～2 秒内的合格脑电估计；特征贡献是统计解释，不代表因果关系。' }));
    if (prediction.domain_warning) body.push(el('p', { class: 'muted', text: prediction.domain_warning }));
  }
  body.push(el('p', { class: 'muted', text: `模型版本 ${String(prediction.model_id || '未知').slice(0, 16)}。研究估计，不代表绝对专注能力。` }));
  return card('真人注意状态模型', body);
}
