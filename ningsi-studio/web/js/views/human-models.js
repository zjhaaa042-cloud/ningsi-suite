/** Human-data import, grouped training, held-out evaluation and portable models. */
import { api } from '../api.js';
import { describeError } from '../store.js';
import { button, card, el, empty, fmtInt, fmtNum, fmtPercent, pageFrame, table } from '../util.js';

let selectedDataset = '';
const INPUT_NAMES = new Set(['manifest.json', 'export.json', 'features.csv', 'SHA256.json', 'baseline.json',
  'dataset.json', 'development.csv', 'test.csv', 'excluded.csv']);

export async function readCaptureFiles(fileList) {
  const candidates = Array.from(fileList).filter(file => INPUT_NAMES.has(file.name));
  const parent = file => (file.webkitRelativePath || file.name).split('/').slice(0, -1).join('/');
  // A merged dataset is often saved under the parent of its source exports.
  // When the user selects that parent, match the local-directory import.
  const direct = candidates.find(file => ['dataset.json', 'export.json'].includes(file.name)
    && !parent(file).includes('/'));
  const roots = new Set(direct ? [parent(direct)]
    : candidates.filter(file => file.name === 'export.json').map(parent));
  const files = candidates.filter(file => roots.has(parent(file)));
  if (!files.length) throw new Error('目录中没有采集器特征文件。请先在采集器中停止采集并导出。');
  if (files.reduce((sum, file) => sum + file.size, 0) > 12 * 1024 * 1024) throw new Error('特征文件合计超过 12 MiB，请分批整理数据集。');
  const result = {};
  for (const file of files) {
    const key = file.webkitRelativePath || file.name;
    // Preserve BOM bytes: the collector's checksums cover the original UTF-8 CSV.
    const text = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true }).decode(await file.arrayBuffer());
    if (Object.hasOwn(result, key)) throw new Error('选中的目录中有重复文件。');
    result[key] = text;
  }
  return result;
}

function datasetCard(data) {
  const pilot = data.pilot;
  return card('数据验收', [table([
    { title: '预分组', field: 'role' }, { title: '有效参与者', field: 'participants' },
    { title: '有效探针', field: 'rows' }, { title: '任务专注 / 内部走神', field: 'labels' },
  ], [['development', '开发组'], ['test', '封存测试组']].map(([key, label]) => ({
    role: label, participants: fmtInt(data[key]?.participants), rows: fmtInt(data[key]?.rows),
    labels: `${fmtInt(data[key]?.labels?.['0'] || 0)} / ${fmtInt(data[key]?.labels?.['1'] || 0)}`,
  }))), el('p', { text: `排除 ${fmtInt(data.excluded_rows)} 条探针，缺失和不合格记录保留在数据集中。` }),
  ...(data.research_protocols || []).map(protocol => el('p', {
    text: `${protocol.label} · ${fmtInt(protocol.sessions)} 次采集` })),
  pilot?.participants ? el('p', { text: `预试 ${pilot.participants} 人，共 ${pilot.formal_probes} 个非练习探针。其中 ${pilot.usable_eeg} 个可提取脑电特征，${pilot.binary_labels} 个取得有效二分类回答。` }) : null,
  ...Object.entries(data.excluded_reasons || {}).map(([reason, count]) => el('p', { class: 'muted',
    text: `${({pilot_cohort: '预试组', unassigned_cohort: '未分组', practice: '练习', simulation_or_unverified_origin: '模拟或未核实来源'})[reason] || reason}：${count}` })),
  ...(data.notices || []).map(text => el('p', { class: 'muted', text })),
  el('p', { class: 'muted', text: data.training_blocker || '标签 0=当前任务专注，1=内部走神。' }),
  ]);
}

function resultsCard(job, onActivate) {
  const report = job.result.evaluation;
  const current = report.test;
  const baseline = report.original_index;
  const rows = [
    ['主要：被试等权 AUROC', 'primary_macro_auc'], ['合并样本 AUROC（次要）', 'auc'],
    ['平衡准确率', 'balanced_accuracy'], ['走神灵敏度', 'sensitivity'], ['专注特异度', 'specificity'],
  ].map(([name, key]) => ({ name, model: fmtNum(current[key], 3), baseline: fmtNum(baseline[key], 3) }));
  const links = [['bundle', '导出完整成果 ZIP'], ['model', '导出模型 JSON'], ['report', '下载独立测试报告'],
    ['metrics', '下载评估 JSON'], ['predictions', '下载逐探针预测 CSV']].map(([kind, text]) => el('a', {
    class: 'btn', href: api.humanModelArtifactUrl(job.run_id, kind), download: '', text,
  }));
  return card('真人模型与独立测试结果', [
    el('p', { text: report.evidence_status === 'independent_test_sufficient_size'
      ? '已完成独立测试；具体性能与限制见报告。'
      : '探索性结果：满足每类至少 5 个标签的测试参与者不足 20 人，暂不形成稳定性能结论。' }),
    el('p', { text: `测试 ${current.participants} 人、${current.n} 条探针；主要分析 ${current.primary_participants} 人。` }),
    table([{ title: '指标', field: 'name' }, { title: '真人模型', field: 'model' }, { title: '原指数', field: 'baseline' }], rows),
    el('p', { text: `主要 AUROC 95% CI：${current.primary_auc_ci95 ? current.primary_auc_ci95.map(n => fmtNum(n, 3)).join(' ～ ') : '样本不足，暂不可估计'}` }),
    el('p', { text: `有效二分类标签下的脑电覆盖率 ${fmtPercent(report.coverage.eeg_coverage_given_binary_label, 1)}；全部正式探针最终可用率 ${fmtPercent(report.coverage.end_to_end_probe_coverage, 1)}。` }),
    el('p', { class: 'muted', text: `模型 ${job.result.model_id.slice(0, 16)}；阈值 ${fmtNum(job.result.threshold, 4)}。原指数保留原有阈值，模型阈值仅使用开发数据选择。` }),
    el('div', { class: 'row' }, [el('a', {class:'btn', href:api.humanModelReportUrl(job.run_id), target:'_blank', rel:'noopener noreferrer', text:'预览独立测试报告'}), ...links]),
    el('div', { class: 'row', style: 'margin-top:12px' }, [button('启用此模型（新会话）', () => onActivate({ run_id: job.run_id }), { primary: true })]),
    el('p', { class: 'muted', text: '模型估计任务专注与内部走神之间的状态倾向，不代表绝对能力，也不证明强化训练有效。' }),
  ]);
}

export async function renderHumanModels(container, ctx) {
  const host = pageFrame(container, '真人数据与模型', []);
  const status = el('p', { class: 'muted', role: 'status', 'aria-live': 'polite' });
  const errors = el('div');
  const dataHost = el('div');
  const resultsHost = el('div');
  const activeHost = el('div');
  const folder = el('input', { id: 'human-capture-folder', type: 'file', webkitdirectory: true, multiple: true });
  const localFolder = el('input', {id:'human-local-folder', type:'text', placeholder:'例如 D:\\实验数据\\采集导出', style:'width:100%'});
  const selector = el('select', { id: 'human-dataset', 'aria-label': '已导入的数据集' });
  const modelFile = el('input', { id: 'human-model-file', type: 'file', accept: '.json,application/json' });
  let state = { datasets: [], jobs: [], active: null };
  let busy = false;
  let timer = null;
  ctx.signal.addEventListener('abort', () => { if (timer !== null) window.clearTimeout(timer); });
  const showError = error => {
    if (!ctx.signal.aborted) { errors.replaceChildren(card('操作未完成', empty(describeError(error)))); status.textContent = ''; }
  };
  const activate = async body => {
    try { const response = await api.activateHumanModel(body); status.textContent = response.data.effective; await refresh(); }
    catch (error) { showError(error); }
  };
  const importData = async () => {
    const local = localFolder.value.trim();
    const response = local ? await api.importHumanLocalData([local])
      : await api.importHumanData(await readCaptureFiles(folder.files || []));
    selectedDataset = response.data.dataset_id;
    folder.value = '';
    localFolder.value = '';
    return response.data;
  };
  const run = async (withImport) => {
    if (busy) return;
    busy = true; importButton.disabled = true; trainButton.disabled = true;
    selector.disabled = true; folder.disabled = true; errors.replaceChildren();
    localFolder.disabled = true;
    try {
      let targetId = selectedDataset;
      status.textContent = withImport ? '正在验收采集数据…' : '正在提交训练任务…';
      if (withImport) {
        const imported = await importData();
        targetId = imported.dataset_id;
        await refresh();
        if (imported.training_blocker) {
          status.textContent = '数据验收完成。' + imported.training_blocker;
          return;
        }
      }
      if (!targetId) throw new Error('请先选择并导入采集数据。');
      const response = await api.trainHumanModel(targetId);
      await refresh(response.data.run_id);
    } catch (error) { showError(error); }
    finally { busy = false; importButton.disabled = false; selector.disabled = false; folder.disabled = false; localFolder.disabled = false; updateTrainButton(); }
  };
  const importButton = button('导入并验收', async () => {
    if (busy) return;
    busy = true; importButton.disabled = true; trainButton.disabled = true; selector.disabled = true; folder.disabled = true; errors.replaceChildren();
    localFolder.disabled = true;
    try {
      status.textContent = '正在导入特征与标签…';
      const imported = await importData();
      await refresh();
      status.textContent = imported.training_blocker ? '数据验收完成。' + imported.training_blocker : '数据已导入，可开始训练评估。';
    }
    catch (error) { showError(error); }
    finally { busy = false; importButton.disabled = false; selector.disabled = false; folder.disabled = false; localFolder.disabled = false; updateTrainButton(); }
  });
  const trainButton = button('一键训练与独立评估', () => run(Boolean(folder.files?.length || localFolder.value.trim())), { primary: true });
  function updateTrainButton() {
    const selected = state.datasets.find(item => item.dataset_id === selectedDataset);
    const hasInput = folder.files?.length || localFolder.value?.trim();
    trainButton.disabled = busy || (!hasInput && (!selected || Boolean(selected.training_blocker)));
  }
  folder.addEventListener('change', () => { if (folder.files?.length) localFolder.value = ''; updateTrainButton(); });
  localFolder.addEventListener('input', () => { if (localFolder.value.trim()) folder.value = ''; updateTrainButton(); });
  selector.addEventListener('change', () => {
    selectedDataset = selector.value;
    folder.value = '';
    localFolder.value = '';
    status.textContent = '';
    renderData(); renderJobs();
  });
  const renderData = () => {
    const data = state.datasets.find(item => item.dataset_id === selectedDataset);
    updateTrainButton();
    dataHost.replaceChildren(data ? datasetCard(data) : empty('尚未导入真人实验数据。可选择单个导出目录、多个导出的父目录，或采集器合并的数据集目录。'));
  };
  const renderJobs = (runId) => {
    const jobs = state.jobs.filter(job => job.dataset_id === selectedDataset);
    const job = jobs.find(item => item.run_id === runId) || jobs.find(item => item.state === 'running') || jobs.at(-1);
    resultsHost.replaceChildren();
    if (!job) return;
    if (job.state === 'completed') {
      status.textContent = '训练与独立测试已完成，可下载成果或启用模型。';
      resultsHost.append(resultsCard(job, activate));
    }
    else if (job.state === 'failed') resultsHost.append(card('任务停止，记录已保留', empty(job.error || '训练失败')));
    else {
      status.textContent = `${job.stage}（${fmtPercent(job.progress)}）`;
      resultsHost.append(card('训练进度', [el('progress', { max: 1, value: job.progress || 0, style: 'width:100%' }),
        el('p', { text: '任务在本机后台运行，可以刷新页面查看进度。封存测试组不参与模型和阈值选择。' })]));
      if (timer !== null) window.clearTimeout(timer);
      timer = window.setTimeout(() => refresh(job.run_id).catch(showError), 1000);
    }
  };
  async function refresh(runId) {
    const response = await api.humanModels();
    if (ctx.signal.aborted) return;
    state = response.data;
    if (!state.datasets.some(item => item.dataset_id === selectedDataset)) selectedDataset = state.datasets.at(-1)?.dataset_id || '';
    selector.replaceChildren(...state.datasets.map(data => el('option', { value: data.dataset_id,
      text: `${data.dataset_id.slice(0, 10)}｜开发 ${data.development.participants} 人｜测试 ${data.test.participants} 人` })));
    selector.value = selectedDataset;
    renderData(); renderJobs(runId);
    activeHost.replaceChildren(card('平台模型', [
      el('p', { text: state.active ? `已启用真人模型 ${state.active.model_id.slice(0, 16)}，新会话将加载此版本。` : '当前没有启用真人模型。' }),
      el('p', { class: 'muted', text: '实时监测单独展示模型走神风险、质控状态与特征依据；设备、单位或算法版本不匹配时暂停输出。' }),
      state.active ? button('停用模型（新会话）', async () => {
        try { const r = await api.deactivateHumanModel(); status.textContent = r.data.effective; await refresh(); } catch (error) { showError(error); }
      }) : null,
    ]));
  }
  host.append(el('p', { text: '导入独立采集器的实验数据，在本机完成按被试训练、独立测试与模型导出。' }),
    card('导入与训练', [el('label', { for: 'human-capture-folder', text: '选择采集器导出目录或合并数据集目录' }),
      folder, el('label', {for:'human-local-folder',text:'本机采集目录（也可直接粘贴路径）'}), localFolder,
      el('div', { class: 'row', style: 'margin-top:12px' }, [importButton, trainButton]),
      el('p', { class: 'muted', text: '仅导入特征、标签和配套清单；原始脑电继续保留在采集目录。仿真、练习和不合格探针不进入真人训练。' }),
      el('label', { for: 'human-dataset', text: '已导入的数据集' }), selector, status, errors]),
    dataHost, resultsHost, activeHost,
    card('导入已有模型', [el('label', { for: 'human-model-file', text: '选择 attention_model.json' }), modelFile, button('导入并启用模型', async () => {
      try {
        const file = modelFile.files?.[0];
        if (!file || file.size > 1024 * 1024) throw new Error('请选择小于 1 MiB 的 attention_model.json。');
        await activate({ model: JSON.parse(await file.text()) });
      } catch (error) { showError(error); }
    })], { sub: '使用真人训练器导出的模型 JSON，可在另一台使用相同算法版本的平台上加载。' }));
  try { await refresh(); } catch (error) { showError(error); }
}
