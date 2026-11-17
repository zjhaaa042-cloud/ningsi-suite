/** Competition workbench: each requirement opens a working product capability. */
import { api } from '../api.js';
import { describeError } from '../store.js';
import { button, card, el, empty, field, fmtInt, fmtNum, fmtPercent, list, statusText, table } from '../util.js';
import { analysisSummary, metric, readRawFile, reasonText, renderBandFeatures, renderQuality,
  renderSpectrum, renderWaveforms } from './showcase-analysis.js';
import { renderHeatmapData, renderJointResults, renderModelResult, renderSavedReportPreview,
  renderTrendData } from './showcase-session.js';

const STEPS = [
  { id: 'hardware', title: '设备采集', detail: '接入与资料核对', number: '01' },
  { id: 'preprocessing', title: '信号预处理', detail: '滤波与伪迹处理', number: '02' },
  { id: 'features', title: '特征提取', detail: 'PSD 与时域特征', number: '03' },
  { id: 'models', title: '训练与部署', detail: '模型与独立测试', number: '04' },
  { id: 'joint', title: '联合评估', detail: '证据与结构化报告', number: '05' },
  { id: 'trends', title: '反馈与趋势', detail: '热力图和周月变化', number: '06' },
];
const ARTIFACTS = [['mixed', '混合干扰'], ['clean', '干净信号对照'], ['blink', '眼动样干扰'],
  ['muscle', '肌电样干扰'], ['movement', '运动样干扰']];
const STATES = [['focused', '专注样频谱'], ['rest', '静息样频谱'], ['drowsy', '困倦样频谱'], ['loaded', '负荷样频谱']];
const METRIC_NAMES = {
  windows_total: '总窗数', windows_usable: '合格窗数', baseline_valid: '基线有效', formal_probes: '正式探针',
  paired: '合格配对', channels: '通道', srate: '采样率 Hz', duration_seconds_by_sample_count: '采集秒数',
  windows_with_valid_history_and_clock: '时标与历史合格窗',
  duration_sec: '采集秒数', windows_with_valid_clock: '时标与历史合格窗', timing_pass: '时标合格窗',
  valid_ratio: '合格比例', train_eligible: '可训练样本', test_eligible: '可测试样本',
};

export function showcaseStepForRequirement(id) {
  return ({ hardware: 'hardware', acquisition: 'hardware', preprocessing: 'preprocessing', preprocess: 'preprocessing',
    features: 'features', models: 'models', model: 'models', joint: 'joint', assessment: 'joint', report: 'joint',
    visualization: 'preprocessing', trends: 'trends', history: 'trends' })[id] || null;
}

function choose(id, choices, current) {
  const select = el('select', { id });
  for (const [value, label] of choices) select.append(el('option', { value, text: label }));
  select.value = current || choices[0]?.[0] || '';
  return select;
}
function routeLink(route, text, uuid) {
  return el('a', { class: 'btn', href: route + (uuid ? `?session=${encodeURIComponent(uuid)}` : ''), text });
}
function details(title, children) {
  return el('details', { class: 'showcase-details' }, [el('summary', { text: title }), ...children]);
}
function downloadJSON(data, name) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }));
  const anchor = el('a', { href: url, download: name });
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export async function render(container, ctx) {
  container.textContent = '';
  const state = { active: STEPS.some(step => step.id === ctx.params.get('step')) ? ctx.params.get('step') : 'preprocessing',
    metadata: null, evidence: null, profile: null, analysis: null, artifact: 'mixed', signalState: 'focused',
    channel: 0, analyzing: false, modelBusy: false, modelResult: null, human: null, sessions: [],
    selected: ctx.params.get('session') || ctx.store.state.currentSession?.uuid || '', session: null,
    report: null, heatmap: null, trend: null, period: 'week', trendField: 'focus', demoBusy: false };
  let disposed = false, timer = null, sessionRequest = 0, trendRequest = 0;
  ctx.onCleanup(() => { disposed = true; if (timer !== null) window.clearTimeout(timer); });
  const alive = () => !disposed && !ctx.signal.aborted;
  const root = el('div', { class: 'showcase' });
  const status = el('div', { class: 'showcase-status', role: 'status', 'aria-live': 'polite' });
  const errors = el('div', { class: 'showcase-error', role: 'alert', hidden: true });
  const panel = el('div', { class: 'showcase-panel', id: 'showcase-panel', role: 'tabpanel' });
  const nav = el('nav', { class: 'showcase-steps', 'aria-label': '赛题技术流程', role: 'tablist' });
  const evidenceHost = el('div', { class: 'showcase-evidence-host' });
  const sourceTag = el('span', { class: 'showcase-source-tag', text: '待选择分析来源' });
  const heroRun = button('运行模拟演示', () => runAnalysis({ mode: 'demo', artifact: state.artifact, state: state.signalState }), { primary: true });
  heroRun.id = 'showcase-run-demo';
  root.append(el('header', { class: 'showcase-hero' }, [
    el('div', {}, [el('span', { class: 'showcase-eyebrow', text: 'A09 · 便携脑电专注力强化训练系统' }),
      el('h1', { text: '从脑电采集，到专注反馈' }),
      el('p', { text: '沿六个步骤查看信号分析、模型训练与联合报告。' }),
      el('div', { class: 'showcase-hero__tags' }, [sourceTag,
        el('span', { text: '数据与结果可追溯' })]),
    ]),
    el('div', { class: 'showcase-hero__actions' }, [heroRun,
      routeLink('#/start', '开始真实检测'),
      el('span', { text: '模拟数据始终单独标注' }),
    ]),
  ]));
  for (const step of STEPS) nav.append(el('button', { type: 'button', class: 'showcase-step',
    id: `showcase-tab-${step.id}`, role: 'tab', 'aria-controls': 'showcase-panel',
    onClick: () => { state.active = step.id; renderStage(); },
  }, [el('span', { class: 'showcase-step__number', text: step.number }),
    el('span', {}, [el('strong', { text: step.title }), el('small', { text: step.detail })])]));
  nav.addEventListener('keydown', event => {
    const index = STEPS.findIndex(step => step.id === state.active);
    const offsets = { ArrowRight: 1, ArrowLeft: -1, Home: -index, End: STEPS.length - index - 1 };
    if (!Object.hasOwn(offsets, event.key)) return;
    event.preventDefault();
    state.active = STEPS[(index + offsets[event.key] + STEPS.length) % STEPS.length].id;
    renderStage();
    nav.querySelector(`#showcase-tab-${state.active}`)?.focus();
  });
  root.append(nav, status, errors, panel, evidenceHost);
  container.append(root);

  function error(message) { if (alive()) { errors.hidden = !message; errors.textContent = message; } }
  function message(text) { if (alive()) status.textContent = text; }

  async function runAnalysis(body) {
    if (state.analyzing) return;
    state.analyzing = true;
    state.active = 'preprocessing';
    heroRun.disabled = true;
    heroRun.textContent = '正在分析…';
    error(''); message('后端正在处理输入，生成真实计算的波形、PSD、频带与质量结果。');
    renderStage();
    try {
      const response = await api.showcaseAnalyze(body);
      if (!alive()) return;
      state.analysis = response.data;
      state.channel = 0;
      sourceTag.textContent = response.data?.source?.label || (body.mode === 'demo' ? '模拟数据' : '导入原始数据');
      sourceTag.dataset.kind = response.data?.source?.simulated ? 'sim' : 'imported';
      message(body.mode === 'demo' ? '模拟分析完成。可切换干扰类型对照，或进入特征页查看计算结果。'
        : '导入片段分析完成。文件来源未经自动核验，缺少独立基线时暂停出分。');
    } catch (err) { error(describeError(err)); message('分析未完成，请核对输入后重试。'); }
    finally {
      state.analyzing = false;
      if (alive()) { heroRun.disabled = false; heroRun.textContent = '运行模拟演示'; renderStage(); }
    }
  }

  function analysisControls() {
    const artifact = choose('showcase-artifact', ARTIFACTS, state.artifact);
    const signalState = choose('showcase-signal-state', STATES, state.signalState);
    artifact.addEventListener('change', () => { state.artifact = artifact.value; });
    signalState.addEventListener('change', () => { state.signalState = signalState.value; });
    const run = button(state.analyzing ? '分析中…' : '运行此模拟条件', () =>
      runAnalysis({ mode: 'demo', artifact: artifact.value, state: signalState.value }), { primary: true, disabled: state.analyzing });
    const file = el('input', { id: 'showcase-raw-file', type: 'file', accept: '.json,application/json' });
    const importButton = button('分析所选原始文件', async () => {
      try { const body = await readRawFile(file.files?.[0]); await runAnalysis(body); }
      catch (err) { error(describeError(err)); }
    }, { disabled: state.analyzing });
    return el('div', {}, [
      el('div', { class: 'showcase-toolbar' }, [field('演示干扰', artifact), field('信号条件', signalState), run]),
      details('导入自己的原始 EEG 片段', [
        el('p', { text: '选择本地 JSON（≤1 MiB），含 samples（通道×采样点）、rate 与 scale_to_uv。可选 timestamps、channel_labels、独立 baseline_samples。' }),
        el('div', { class: 'showcase-toolbar' }, [file, importButton]),
        el('pre', { class: 'showcase-json-example', text: '{ "rate": 250, "scale_to_uv": 1, "samples": [[通道1样点…], [通道2样点…]] }' }),
        el('p', { class: 'showcase-caption', text: '只处理手动选择的文件内容，不读取文件中的路径。未知单位应先核对倍率；导入不等于真人来源已经确认。' }),
      ]),
    ]);
  }

  function preprocessingStage() {
    const data = state.analysis;
    const body = el('div');
    const pipeline = details('查看信号处理流程', [
      el('div', { class: 'showcase-pipeline' }, ['确认单位与时标', '去漂移与带通', '50/60 Hz 陷波', '伪迹标记与门控', '保留有效窗口']
        .map((text, i) => el('span', {}, [el('b', { text: String(i + 1) }), document.createTextNode(text)]))),
      el('p', { class: 'showcase-caption', text: '原始输入不改写；信号清理、质量判断与状态估计分别展示。' }),
    ]);
    if (data) body.append(details('调整分析条件或导入文件', [analysisControls(), pipeline]));
    else body.append(card('选择分析输入', [analysisControls(), pipeline], { sub: '运行模拟条件，或导入自己的原始 EEG 片段。' }));
    if (!data) {
      body.append(card('处理前后对照', [el('p', { class: 'showcase-caption', text: '分析完成后，在同一时间轴比较以下三层波形。' }),
        el('div', { class: 'showcase-empty-flow' }, ['原始 EEG', '滤波后 EEG', '污染片段屏蔽后'].map((text, i) =>
          el('div', {}, [el('span', { text: `0${i + 1}` }), el('strong', { text }), el('p', { text: i === 2 ? '疑似污染留空，不伪造样点' : '幅度与时间轴同时保留' })])))]));
      return body;
    }
    body.append(analysisSummary(data));
    const channels = list(data.traces?.channels).map((item, i) => [String(i), item.label || `Ch${i + 1}`]);
    const selector = choose('showcase-channel', channels, String(state.channel));
    selector.addEventListener('change', () => { state.channel = Number(selector.value); renderStage(); });
    const actions = el('div', { class: 'showcase-actions' }, [selector,
      button('导出分析结果 JSON', () => downloadJSON(data, 'ningsi-analysis.json'))]);
    body.append(card('同一窗口的处理前后波形', renderWaveforms(data, state.channel), { actions,
      sub: data.source?.label || '按所选数据来源解释' }));
    body.append(el('div', { class: 'showcase-grid showcase-grid--spectrum' }, [
      card('功率谱与工频干扰', renderSpectrum(data), { sub: '原始与处理后使用同一套 PSD 计算。' }),
      card('质量门控与基线', renderQuality(data)),
    ]));
    body.append(details('处理参数与响应时间', [table([
      { title: '参数', field: 'name' }, { title: '本次配置', field: 'value' },
    ], [
      { name: '分析窗 / 更新步长', value: `${fmtNum(data.timing?.window_sec, 0)} s / ${fmtNum(data.timing?.step_sec, 0)} s` },
      { name: '滤波历史', value: `${fmtNum(data.timing?.filter_history_sec, 0)} s` },
      { name: '低通自身延迟', value: `约 ${fmtNum(data.timing?.lowpass_delay_ms, 0)} ms；总延迟随频率变化` },
      { name: '本次后端计算', value: `${fmtNum(data.timing?.compute_ms, 2)} ms；不等于状态响应时长` },
    ]), el('p', { class: 'showcase-caption', text: '无同步参考时以保守检测与拒评为主；有 EOG 参考的校正能力在真实设备入口按参考配置启用。滤波和拒评不构成真人识别准确性证明。' })]));
    return body;
  }

  function featuresStage() {
    const data = state.analysis;
    if (!data) return card('先取得有效输入', [empty('请在预处理步骤运行模拟条件，或导入原始片段。'),
      button('打开信号工作台', () => { state.active = 'preprocessing'; renderStage(); }, { primary: true })]);
    const scores = data.latest?.scores || {};
    return el('div', {}, [
      card('θ / α / β 可解释特征', renderBandFeatures(data), { sub: data.source?.label || '分析结果' }),
      el('div', { class: 'showcase-grid' }, [card('频谱密度 PSD', renderSpectrum(data)),
        card('个人基线与相对指数', [el('div', { class: 'showcase-metrics showcase-metrics--compact' }, [
          metric('专注相关指数', fmtNum(scores.focus, 3)), metric('放松指数', fmtNum(scores.relax, 3)), metric('负荷指数', fmtNum(scores.load, 3)),
        ]), el('p', { text: data.latest?.indicator_reason ? reasonText(data.latest.indicator_reason) : '有效窗口与基线配置匹配后才计算指数。' }),
        el('p', { class: 'showcase-caption', text: '指数为相对本次个人基线的变化，不是专注概率。采样率、导联或处理版本变化时需要重新核对基线。' })]),
      ]),
    ]);
  }

  function hardwareStage() {
    const profile = state.profile?.profile || {};
    const controls = {};
    const fields = [
      ['manufacturer', '厂家', '按设备资料填写'], ['model', '实际设备型号', '例如设备铭牌上的完整型号'],
      ['electrodes', '实际 EEG 电极', '例如 Fp1,Fp2；未确认时留空'], ['reference', '参考电极', '填写实际参考位置'],
      ['srate', '标称采样率 Hz', '核对厂商发布配置'], ['source_unit', '输入单位', '选择已确认单位'],
      ['scale_to_uv', '换算到 µV 的倍率', 'ADC 时填写已校准的 µV/计数'], ['unit_basis', '单位与倍率依据', '厂商说明或校准记录'],
      ['document_reference', '设备资料与赛事认可依据', '记录文档名称、网址或资料位置'], ['notes', '核对备注', '记录佩戴、导联与待确认事项'],
    ];
    const grid = el('div', { class: 'showcase-device-fields' });
    for (const [key, label, placeholder] of fields) {
      const input = key === 'source_unit'
        ? choose(`showcase-device-${key}`, [['', '待确认'], ['uV', 'µV'], ['mV', 'mV'], ['V', 'V'], ['nV', 'nV'], ['ADC', 'ADC 计数']], profile[key])
        : el('input', { id: `showcase-device-${key}`, type: ['srate', 'scale_to_uv'].includes(key) ? 'number' : 'text',
          step: 'any', value: profile[key] ?? '', placeholder, maxLength: key === 'notes' ? 2000 : 1000 });
      controls[key] = input;
      grid.append(field(label, input));
    }
    const saved = el('p', { class: 'showcase-caption', role: 'status', text: state.profile?.saved
      ? '已保存操作员声明；设备合规、实际导联与标定仍需资料核验。' : '尚未保存设备核对资料。' });
    const save = button('保存设备核对资料', async () => {
      save.disabled = true; error('');
      const payload = {};
      for (const [key, control] of Object.entries(controls)) payload[key] = ['srate', 'scale_to_uv'].includes(key)
        ? (control.value.trim() ? Number(control.value) : null) : control.value.trim();
      try { const response = await api.saveShowcaseDeviceProfile(payload); if (!alive()) return;
        state.profile = response.data; message('设备资料已保存为待核验声明，不自动判定合规。'); renderStage();
      } catch (err) { error(describeError(err)); save.disabled = false; }
    }, { primary: true });
    const missing = list(state.profile?.missing_fields);
    return el('div', {}, [
      el('div', { class: 'showcase-grid' }, [card('真实设备接入', [
        el('p', { text: '开启厂商采集程序的 EEG 数据流，核对信号与配置后开始评估。' }),
        el('div', { class: 'showcase-actions' }, [routeLink('#/devices', '发现设备与实时预览'), routeLink('#/start', '创建真实检测会话')]),
        details('查看设备接入链路', [el('div', { class: 'showcase-hardware-flow' }, ['便携 EEG 硬件', '厂商采集程序', '实时 EEG 数据流', '质量与时标检查']
          .map((text, i) => el('div', {}, [el('span', { text: `0${i + 1}` }), el('strong', { text })]))),
          el('p', { class: 'showcase-caption', text: '至少一个 EEG 通道。分析前确认来源、单位、电极与参考；Capture 保存连续原始记录，Studio 默认保存会话结果与逐窗诊断。' })]),
      ]), card('设备核验状态', [
        el('span', { class: 'showcase-status-chip', text: '资料核验待完成' }), saved,
        missing.length ? details(`查看待补资料（${missing.length}项）`, [el('ul', { class: 'showcase-advice' }, missing.map(item => el('li', { text: `${item.label || item.key}待补充` })))])
          : el('p', { text: '填写完整仅表示声明字段齐备，需现场核对资料与实际配置。' }),
        el('p', { class: 'showcase-caption', text: '软件连通不能证明设备合规。真实信号质量不足时暂停评分，原始记录继续保留。' }),
      ])]),
      details('填写或修改设备与标定资料', [
        el('p', { class: 'showcase-caption', text: '记录实际资料；手动填写不等同于验证通过。' }), grid, save,
      ]),
    ]);
  }

  function modelsStage() {
    const run = button(state.modelBusy ? '正在训练…' : '运行模拟训练', async () => {
      state.modelBusy = true; error(''); renderStage();
      try { const response = await api.trainModel({ subjects: 10, windows_per_state: 12 });
        if (alive()) { state.modelResult = response.data; message('模拟训练完成；真实训练与独立测试使用单独入口。'); }
      } catch (err) { error(describeError(err)); }
      finally { state.modelBusy = false; if (alive()) renderStage(); }
    }, { primary: true, disabled: state.modelBusy });
    const active = state.human?.active;
    return el('div', {}, [card('可训练、可部署的模型', [
      el('p', { text: '查看按被试划分、模型拟合与独立测试的完整流程。模拟结果单独标注。' }),
      el('div', { class: 'showcase-actions' }, [run, routeLink('#/models', '导入真人数据与训练部署')]),
      renderModelResult(state.modelResult),
      details('训练演示说明', [el('p', { class: 'showcase-caption', text: '演示实际运行传统机器学习训练。SVM、随机森林与深度网络是赛题举例，不要求同时实现；模拟训练结果不代表真人识别准确性。' })]),
    ]), el('div', { class: 'showcase-grid' }, [card('真人模型生命周期', [
      el('ol', { class: 'showcase-advice' }, ['导入采集器特征与来源清单', '按参与者交叉验证，选择模型和校准',
        '冻结后评估封存测试参与者', '核对设备与实现版本，启用到新会话'].map(text => el('li', { text }))),
      el('p', { class: 'showcase-caption', text: active ? '已有启用的真人模型，详细适用范围见模型训练页。' : '当前尚未取得可展示的已启用真人模型信息。' }),
    ]), card('状态识别的适用范围', [
      el('p', { text: '专注模型估计既定任务中的专注／内部走神标签；压力与情绪效价候选保留研究分数。' }),
      el('span', { class: 'showcase-status-chip', text: '目标设备的三态有效性待验证' }),
      el('p', { class: 'showcase-caption', text: '模型概率、频带相对指数与本人自评分别保存，不用模拟训练结果代替真人准确性。' }),
    ])])]);
  }

  async function loadTrend() {
    const ticket = ++trendRequest;
    if (!state.selected || !state.session?.participant) { state.trend = null; return; }
    try {
      const response = await api.reportsTrend({ participant: state.session.participant,
        field: state.trendField, period: state.period });
      if (!alive() || ticket !== trendRequest) return;
      state.trend = response.data; if (state.active === 'trends') renderStage();
    } catch (err) { if (alive() && ticket === trendRequest) error(`趋势加载失败：${describeError(err)}`); }
  }

  async function loadSession(uuid, poll = false) {
    const ticket = ++sessionRequest;
    if (timer !== null) { window.clearTimeout(timer); timer = null; }
    state.selected = uuid;
    if (!poll) { ++trendRequest; state.session = null; state.report = null; state.heatmap = null; state.trend = null; }
    if (!uuid) { renderStage(); return; }
    const results = await Promise.allSettled([api.getSession(uuid), api.report(uuid), api.sessionHeatmap(uuid)]);
    if (!alive() || ticket !== sessionRequest) return;
    if (results[0].status === 'fulfilled') {
      state.session = results[0].value.data;
      state.sessions = state.sessions.some(item => item.uuid === uuid)
        ? state.sessions.map(item => item.uuid === uuid ? state.session : item)
        : [state.session, ...state.sessions];
      ctx.store.setState({ currentSession: state.session });
    } else error(`会话加载失败：${describeError(results[0].reason)}`);
    if (results[1].status === 'fulfilled') state.report = results[1].value.data;
    if (results[2].status === 'fulfilled') state.heatmap = results[2].value.data;
    if (state.active === 'joint' || state.active === 'trends') renderStage();
    if (!poll || state.session?.status === 'done') loadTrend();
    if (state.demoBusy && state.selected === uuid) {
      if (['done', 'failed', 'cancelled'].includes(state.session?.status)) {
        state.demoBusy = false;
        message(state.session.status === 'done' ? '完整模拟会话已完成，联合评估、报告与历史数据已保存。'
          : `模拟会话${statusText(state.session.status)}，请查看会话流程。`);
        if (alive()) renderStage();
      } else {
        message(`模拟会话：${state.session?.phase_label || state.session?.phase || '运行中'} · ${fmtPercent(state.session?.progress, 0)}`);
        timer = window.setTimeout(() => loadSession(uuid, true), 1500);
      }
    }
  }

  async function createDemoSession() {
    if (state.demoBusy) return;
    state.demoBusy = true; error(''); message('正在创建明确标注的加速模拟会话。'); renderStage();
    try {
      const response = await api.createShowcaseDemo();
      if (!alive()) return;
      const session = response.data?.session || response.data;
      if (!session?.uuid) throw new Error('服务器没有返回模拟会话编号。');
      state.sessions = [session, ...state.sessions.filter(item => item.uuid !== session.uuid)];
      await loadSession(session.uuid);
    } catch (err) { state.demoBusy = false; error(describeError(err)); if (alive()) renderStage(); }
  }

  function sessionChooser() {
    const choices = [['', '选择已保存的会话'], ...state.sessions.map(item => [item.uuid,
      `${item.participant || '参与者'} · ${statusText(item.status)} · ${String(item.device || '').startsWith('sim') ? '模拟' : '按记录来源'} · ${item.uuid.slice(0, 8)}`])];
    if (state.selected && !choices.some(([value]) => value === state.selected)) choices.push([state.selected, state.selected.slice(0, 12)]);
    const select = choose('showcase-session', choices, state.selected);
    select.addEventListener('change', () => loadSession(select.value));
    return card('选择展示会话', [el('div', { class: 'showcase-toolbar' }, [
      field('会话记录', select),
      button(state.demoBusy ? '模拟会话运行中…' : '运行完整模拟流程', createDemoSession, { primary: true, disabled: state.demoBusy }),
      routeLink('#/start', '开始真实检测'),
    ]), el('p', { class: 'showcase-caption', text: '完整模拟流程会创建真实保存的软件演示记录，量表与按键由模拟流程完成，来源始终标注为模拟。' }),
    state.session ? el('div', { class: 'showcase-actions' }, [
      routeLink('#/flow', '查看会话流程', state.selected), routeLink('#/live', '实时波形与状态监测', state.selected),
      routeLink('#/training', '神经反馈训练画面', state.selected),
    ]) : null]);
  }

  function trendsStage() {
    const period = choose('showcase-period', [['week', '按周'], ['month', '按月']], state.period);
    const fieldSelect = choose('showcase-trend-field', [['focus', '专注相关指数'], ['stress', '本人压力自评']], state.trendField);
    const change = () => { state.period = period.value; state.trendField = fieldSelect.value; state.trend = null; loadTrend(); renderStage(); };
    period.addEventListener('change', change); fieldSelect.addEventListener('change', change);
    return el('div', {}, [sessionChooser(),
      card('状态热力图', renderHeatmapData(state.heatmap), { sub: state.session
        ? `${state.session.participant || '参与者'} · ${String(state.session.device || '').startsWith('sim') ? '模拟会话' : '已有会话'}` : '选择会话后显示实际保存的分档与缺失窗' }),
      card('历史查询与周 / 月趋势', [el('div', { class: 'showcase-toolbar' }, [field('指标', fieldSelect), field('周期', period)]),
        renderTrendData(state.trend, state.trendField, state.period), routeLink('#/history', '完整历史查询与导出')]),
      card('实时预警与个性化反馈', [el('div', { class: 'showcase-evidence-grid' }, [
        metric('质量不足', '暂停评分', '原始记录保留'), metric('持续越界', '按有效时间预警', '阈值与解除条件明确'),
        metric('训练目标', '分轮自适应', '覆盖不足时不提高目标'),
      ]), el('div', { class: 'showcase-actions' }, [routeLink('#/training', '打开神经反馈训练', state.selected),
        routeLink('#/live', '打开实时预警', state.selected)]),
      el('p', { class: 'showcase-caption', text: '反馈流程可演示；是否改善真实专注表现需独立前后测与对照。压力曲线使用本人自评，不伪装为脑电压力测量。' })]),
    ]);
  }

  function renderStage() {
    if (!alive()) return;
    for (const step of STEPS) {
      const node = nav.querySelector(`#showcase-tab-${step.id}`);
      if (node) { node.setAttribute('aria-selected', String(state.active === step.id)); node.tabIndex = state.active === step.id ? 0 : -1; }
    }
    panel.setAttribute('aria-labelledby', `showcase-tab-${state.active}`);
    panel.textContent = '';
    const step = STEPS.find(item => item.id === state.active);
    panel.append(el('div', { class: 'showcase-section-heading' }, [
      el('span', { text: `${step.number} / 06` }), el('h2', { text: step.title }),
      el('p', { text: step.detail }),
    ]));
    if (state.active === 'hardware') panel.append(hardwareStage());
    else if (state.active === 'preprocessing') panel.append(preprocessingStage());
    else if (state.active === 'features') panel.append(featuresStage());
    else if (state.active === 'models') panel.append(modelsStage());
    else if (state.active === 'joint') panel.append(sessionChooser(), renderSavedReportPreview(state.report, state.session),
      renderJointResults(state.report, state.session, api));
    else panel.append(trendsStage());
  }

  function renderEvidence() {
    evidenceHost.textContent = '';
    const data = state.evidence;
    if (!data) { evidenceHost.append(details('查看验证证据', [empty('证据暂不可用，请刷新重试。')])); return; }
    evidenceHost.append(details('查看验证证据 · 赛题要求与当前进度', [
      el('p', { class: 'showcase-caption', text: '已保存的开发验收记录 · 非当前设备实时测量。软件实现、公开数据验证与目标设备验收分别记录；展示画面不会使待验收项自动通过。' }),
      data.evidence_origin === 'bundled_development_snapshot' ? el('p', { class: 'showcase-source',
        text: `开发证据快照 · 非当前设备实时测量${data.snapshot_at ? ` · ${data.snapshot_at}` : ''}。${data.snapshot_note || ''}` }) : null,
      table([
        { title: '赛题要求', render: row => row.title },
        { title: '当前范围', render: row => row.status_label || row.verification_status || '待核验' },
        { title: '下一步验收', render: row => row.next_action || list(row.gaps).join('；') || '见对应证据' },
        { title: '画面', render: row => button('查看', () => {
          const step = showcaseStepForRequirement(row.id);
          if (!step) { ctx.navigate(row.route || '#/start'); return; }
          state.active = step; renderStage(); panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, { small: true }) },
      ], list(data.requirements)),
      el('div', { class: 'showcase-grid' }, list(data.evidence).map(item => card(item.title, [
        el('p', { class: 'showcase-caption', text: item.available ? '已有保存的验证记录' : '本机证据文件不可用，未补造结果' }),
        item.available ? el('div', { class: 'showcase-record-metrics' }, Object.entries(item.metrics || {}).map(([key, value]) =>
          el('span', {}, [el('b', { text: METRIC_NAMES[key] || '其他记录项' }), document.createTextNode(`：${typeof value === 'boolean' ? value ? '是' : '否'
            : value === null ? '—' : key === 'valid_ratio' ? fmtPercent(value, 1) : Number.isInteger(value) ? value : fmtNum(value, 2)}`)]))) : null,
        ...list(item.notes).map(text => el('p', { class: 'showcase-caption', text })),
      ]))),
      data.research?.conclusion ? el('p', { class: 'showcase-caption', text: data.research.conclusion }) : null,
    ]));
  }

  renderStage();
  message('正在读取本机功能状态与已保存的验收证据…');
  const results = await Promise.allSettled([api.showcaseMetadata(), api.showcaseEvidence(), api.showcaseDeviceProfile(),
    api.listSessions({ limit: 30, page: 1 }), api.humanModels()]);
  if (!alive()) return;
  const keys = ['metadata', 'evidence', 'profile', 'sessions', 'human'];
  results.forEach((result, i) => {
    if (result.status === 'fulfilled') state[keys[i]] = i === 3 ? list(result.value.data?.items) : result.value.data;
  });
  const failures = results.map((result, i) => result.status === 'rejected' ? keys[i] : null).filter(Boolean);
  message(failures.length ? '部分状态暂不可用；仍可选择演示条件，错误操作会给出具体原因。' : '已连接本机算法服务。选择模拟条件或已有会话开始展示。');
  if (!state.selected) state.selected = state.sessions.find(item => item.status === 'done')?.uuid || '';
  renderStage(); renderEvidence();
  if (state.selected) await loadSession(state.selected);
  else loadTrend();
}

export default render;
