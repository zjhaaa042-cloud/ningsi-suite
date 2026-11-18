"""Render the training view with real SSE handlers and controlled API replies."""
import json
from pathlib import Path
import shutil
import subprocess
import unittest

from .test_frontend_dom import NODE_STUB

ROOT = Path(__file__).resolve().parents[1]

HARNESS = r'''
import assert from 'node:assert/strict';
// Match the browser: replacing textContent also removes existing children.
Object.defineProperty(StubNode.prototype, 'textContent', {
  get() { return this._text || ''; },
  set(value) { this._text = String(value); this.children = []; },
});
ClassList.prototype.add = function (...names) { names.forEach(n => this.toggle(n, true)); };
StubNode.prototype.append = function (...nodes) { this.children.push(...nodes); };
document.createElementNS = (ns, tag) => new StubNode(tag);
document.querySelector = () => null;
let poll, events;
const cleanups = [];
globalThis.window = {
  location: {search: ''}, setTimeout, clearTimeout,
  setInterval: callback => { poll = callback; return 1; }, clearInterval() {},
};
globalThis.EventSource = class {
  constructor() { events = this; this.handlers = new Map(); }
  addEventListener(type, callback) { this.handlers.set(type, callback); }
  removeEventListener() {} close() {}
};
const {api} = await import(API_URI);
const saved = (seq, mean, n=2) => ({seq, index:seq+1, target:.5, hold_sec:6,
  stats:{mean, n, achieved:n>0, on_target_ratio:n>0?1:0, current_hold_sec:6, max_hold_sec:6}});
let response = {segments:[saved(0,.8)], summary:{}};
let session = {status:'running', phase:'training', runs:[]};
api.training = async () => ({data:response});
api.getSession = async () => ({data:session});
const {render} = await import(VIEW_URI);
const root = new StubNode('main');
const mount = () => render(root, {
  params:new URLSearchParams('session=test'), signal:{aborted:false},
  store:{state:{},setState(){}}, onCleanup:f=>cleanups.push(f), navigate(){},
});
const settle = async () => { for (let i=0;i<8;i++) await Promise.resolve(); };
const emit = (type,data) => events.handlers.get(type)({data:JSON.stringify({type,data})});
const all = node => [node,...(node.children||[]).flatMap(all)];
const card = title => all(root).find(n => n.classList?.contains('card') &&
  all(n).some(child => child.tagName==='H3' && child.textContent===title));
const field = label => {
  const status = card('连续保持状态');
  assert.ok(status, 'training status card must render');
  const caption = all(status).find(n=>n.tagName==='P' && n.textContent===label);
  assert.ok(caption, `missing status field: ${label}`);
  const row = all(status).find(n=>n.children?.includes(caption));
  return row.children[1].textContent;
};
'''


@unittest.skipUnless(shutil.which('node'), 'Node is required for training view tests')
class TrainingFrontendTests(unittest.TestCase):
    def run_view(self, scenario):
        program = (NODE_STUB + HARNESS + scenario + '\ncleanups.forEach(f=>f());')
        program = program.replace('API_URI', json.dumps((ROOT / 'web/js/api.js').as_uri()))
        program = program.replace('VIEW_URI', json.dumps((ROOT / 'web/js/views/training.js').as_uri()))
        result = subprocess.run(['node', '--input-type=module', '-e', program],
                                capture_output=True, text=True, encoding='utf-8', timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_invalid_live_window_and_poll_never_reuse_saved_score(self):
        self.run_view(r'''
mount(); await settle();
emit('feedback', {segment:2,score:.65,target:.5,usable:true,achieved:false,current_hold_sec:2});
assert.equal(field('当前专注相关指数'),'0.650');
emit('feedback', {segment:2,score:null,target:.5,usable:false,achieved:true,current_hold_sec:6});
assert.equal(field('当前专注相关指数'),'—');
assert.equal(field('连续保持状态'),'—');
assert.equal(field('本段达标时间占比'),'—');
assert.equal(field('当前段'),'2');
assert.equal(field('当前连续保持 / 本段最长'),'0.0s / —');
poll(); await settle();
assert.equal(field('当前专注相关指数'),'—','empty summary must not enable saved-result fallback');
assert.equal(field('连续保持状态'),'—');
emit('feedback', {segment:2,score:.7,target:.5,usable:true,achieved:false,current_hold_sec:2});
assert.equal(field('当前专注相关指数'),'0.700','feedback must recover after a missing window');
poll(); await settle();
assert.equal(field('当前专注相关指数'),'0.700','polling cannot erase a valid live score');
emit('feedback', {segment:2,score:0,target:.5,usable:true,achieved:false});
assert.equal(field('当前专注相关指数'),'0.000','a genuine zero is a valid score');
assert.equal(field('连续保持状态'),'未达标');
emit('feedback', {segment:2,score:null,target:.5,usable:true,achieved:true});
assert.equal(field('当前专注相关指数'),'—','a valid raw window without a score remains missing');
assert.equal(field('连续保持状态'),'—');
emit('feedback', {segment:2,score:.9,target:null,usable:false,achieved:true});
assert.equal(field('当前专注相关指数'),'—','unusable flag overrides even a numeric score');
assert.equal(field('连续保持状态'),'—');
emit('segment', saved(1,.6));
emit('phase', {key:'training',state:'done'});
assert.equal(field('末段平均专注相关指数'),'0.600');
assert.equal(field('连续保持状态'),'达标','completed training retains its saved result');
''')

    def test_completed_session_loads_saved_result_without_live_feedback(self):
        self.run_view(r'''
session = {status:'running',phase:'baseline_after',runs:[{phase:'training',status:'done'}]};
mount(); await settle();
assert.equal(field('末段平均专注相关指数'),'0.800');
assert.equal(field('连续保持状态'),'达标');
''')

    def test_index_gauge_and_feedback_states(self):
        self.run_view(r'''
mount(); await settle();
const gaugeTexts = () => all(card('专注相关指数')).map(n => n.textContent);
assert.ok(gaugeTexts().includes('等待训练反馈'));
emit('feedback', {segment:1,score:.964,target:.65,usable:true,feedback_mode:'eeg'});
for (const text of ['0.00', '0.50', '1.00', '0.964', '指数', '目标线 0.650', '脑电反馈']) {
  assert.ok(gaugeTexts().includes(text), `missing gauge text: ${text}`);
}
assert.ok(gaugeTexts().some(text => text.includes('0.50 对应本次训练前的个人基线')));
assert.ok(!gaugeTexts().includes('分'));

emit('feedback', {segment:1,score:null,usable:false,feedback_mode:'quality',
  heatmap_cell:{reason_labels:['波幅过大']}});
assert.ok(gaugeTexts().includes('信号质量反馈'));
assert.ok(gaugeTexts().some(text => text.includes('波幅过大') && text.includes('不计入有效脑电反馈时间')));
assert.ok(!gaugeTexts().includes('0.964'));

emit('feedback', {segment:1,score:null,usable:false,feedback_mode:'guided',
  effective_time:{baseline_valid:false}});
assert.ok(gaugeTexts().includes('固定引导'));
assert.ok(gaugeTexts().some(text => text.includes('睁眼基线未通过')));
emit('feedback', {segment:1,score:null,usable:false,feedback_mode:'rest'});
assert.ok(gaugeTexts().includes('轮间休息'));
assert.ok(gaugeTexts().includes('等待下一轮训练。'));

const { drawGauge } = await import(new URL('../charts.js', VIEW_URI));
const defaultGauge = new StubNode('div');
drawGauge(defaultGauge, .964);
const defaultTexts = all(defaultGauge).map(n => n.textContent);
for (const text of ['0', '50', '100', '96', '分']) assert.ok(defaultTexts.includes(text));
''')

    def test_terminal_session_preserves_partial_saved_training(self):
        self.run_view(r'''
session = {status:'cancelled',phase:'cancelled',runs:[]};
mount(); await settle();
assert.equal(field('末段平均专注相关指数'),'0.800');
assert.equal(field('连续保持状态'),'达标');
assert.equal(poll,undefined);
assert.equal(events,undefined);
''')

    def test_short_protocol_does_not_subscribe_or_poll_for_training(self):
        self.run_view(r'''
session={status:'running',protocol:'short',phase:'qc',runs:[]};
let requests=0;api.training=async()=>{requests++;return {data:{}};};
mount();await settle();
assert.ok(all(root).some(node=>node.textContent.includes('短协议不包含反馈训练阶段')));
assert.equal(requests,0);assert.equal(poll,undefined);assert.equal(events,undefined);
''')

    def test_empty_last_segment_cannot_fall_back_to_session_average(self):
        self.run_view(r'''
response = {segments:[saved(0,.8),saved(1,0,0)],
  summary:{mean_focus:.8,on_target_ratio:1,segments:[saved(0,.8),saved(1,0,0)]}};
session = {status:'done',phase:'done',runs:[]};
mount(); await settle();
assert.equal(field('末段平均专注相关指数'),'—','empty historic segment is missing, even if its old mean is zero');
assert.equal(field('连续保持状态'),'—');
assert.equal(field('本段达标时间占比'),'—');
''')

    def test_saved_time_coverage_and_valid_time_success_are_separate(self):
        self.run_view(r'''
const segment = saved(0,.8);
segment.duration_sec=10;
Object.assign(segment.stats, {timing_spec:'neurofeedback-v3',valid_time_ratio:.2,
  valid_seconds:2,on_target_valid_ratio:1,on_target_ratio:.2,achieved:false,coverage_sufficient:false});
response = {segments:[segment],summary:{spec:'neurofeedback-v3'}};
session = {status:'done',phase:'done',runs:[]};
mount(); await settle();
assert.equal(field('本段达标时间占比'),'20%（实际时间）');
assert.equal(field('有效时间覆盖率'),'20%');
assert.equal(field('有效时间内达标率'),'100%');
assert.equal(field('有效反馈秒数 / 实际时长'),'2.0s / 10.0s');
assert.ok(all(card('连续保持状态')).some(n=>n.textContent.includes('样本不足')));
assert.equal(field('连续保持状态'),'未达标');
''')

    def test_empty_and_legacy_records_do_not_invent_valid_time_success(self):
        self.run_view(r'''
const missing=saved(0,0,0);
missing.duration_sec=10;
Object.assign(missing.stats,{timing_spec:'neurofeedback-v3',valid_seconds:0,
  valid_time_ratio:0,on_target_valid_ratio:0,on_target_ratio:0});
response={segments:[missing],summary:{spec:'neurofeedback-v3'}};
session={status:'done',phase:'done',runs:[]};
mount();await settle();
assert.equal(field('末段平均专注相关指数'),'—');
assert.equal(field('有效时间内达标率'),'—','all-bad data never becomes 0% success');
assert.equal(field('有效时间覆盖率'),'0%');
assert.equal(field('有效反馈秒数 / 实际时长'),'0.0s / 10.0s');
assert.ok(all(card('连续保持状态')).some(n=>n.textContent.includes('暂无有效评分')));
root.textContent='';response={segments:[saved(0,.8)],summary:{spec:'old'}};
mount();await settle();
assert.equal(field('有效时间内达标率'),'—','a historic sample ratio cannot become a time ratio');
assert.equal(field('有效时间覆盖率'),'—');
assert.equal(field('本段达标时间占比'),'100%（旧版有效样本）');
assert.ok(all(card('连续保持状态')).some(n=>n.textContent.includes('旧版记录缺少有效时间字段')));
''')

    def test_effective_time_live_pause_reload_and_max_duration_are_consistent(self):
        self.run_view(r'''
response={segments:[],summary:{}};
mount();await settle();
const progress={spec:'effective-time-v1',target_seconds:50,valid_seconds:2,
  elapsed_seconds:10,max_seconds:100,extension_seconds:0,quality_paused:true,
  completion_reason:null,target_reached:false,completed:false};
const current={segment:1,score:null,target:.5,usable:false,feedback_mode:'guided',
  mode_label:'固定引导',current_hold_sec:6,max_hold_sec:2,hold_sec:6,
  effective_time:progress,segment_stats:{duration_sec:10,n:1,valid_seconds:2,
    on_target_valid_ratio:1,valid_time_ratio:.2,on_target_ratio:.2,coverage_sufficient:false}};
emit('feedback',current);
assert.equal(field('当前专注相关指数'),'—');
assert.equal(field('当前连续保持 / 本段最长'),'0.0s / 2.0s');
assert.equal(field('有效时间内达标率'),'100%');
assert.equal(field('有效时间覆盖率'),'20%');
const before=all(card('有效时长训练进度')).map(n=>n.textContent).join(' ');
assert.ok(before.includes('2.0s / 50.0s'));
assert.ok(before.includes('质量暂停'));
assert.ok(before.includes('累计有效时间保留'));
const envelope={key:'training',phase:'training',segment:1,effective_time:progress};
root.textContent='';response={segments:[],summary:{},current_feedback:current,effective_time:envelope};
session={status:'running',phase:'training',runs:[],runtime:{latest_feedback:current,effective_time:envelope}};
mount();await settle();
assert.equal(field('有效时间内达标率'),'100%');
assert.equal(field('有效时间覆盖率'),'20%');
assert.equal(field('当前专注相关指数'),'—');
assert.equal(all(card('有效时长训练进度')).map(n=>n.textContent).join(' '),before);
emit('effective_progress',{key:'training',effective_time:{...progress,
  elapsed_seconds:102,max_seconds:100,extension_seconds:52,quality_paused:false,completion_reason:'max_duration'}});
const ended=all(card('有效时长训练进度')).map(n=>n.textContent).join(' ');
assert.ok(ended.includes('数据不足'));
assert.ok(ended.includes('1:42.0 / 1:40.0'),'actual elapsed may exceed bounded-read scheduling limit');
assert.equal(field('当前专注相关指数'),'—','progress events never fabricate scores');
''')

    def test_invalid_baseline_and_simulation_time_basis_are_explicit(self):
        self.run_view(r'''
response={segments:[],summary:{}};mount();await settle();
emit('feedback',{segment:1,score:null,usable:false,feedback_mode:'guided',mode_label:'固定引导',
  effective_time:{spec:'effective-time-v1',target_seconds:50,valid_seconds:0,
    elapsed_seconds:20,max_seconds:100,extension_seconds:0,quality_paused:false,
    baseline_valid:false,completion_reason:null,completed:false,time_scale:.05,
    time_basis:'protocol_seconds',actual_elapsed_seconds:1},
  segment_stats:{duration_sec:20,n:0,valid_seconds:0,valid_time_ratio:0,on_target_valid_ratio:null}});
const note=all(card('有效时长训练进度')).map(n=>n.textContent).join(' ');
assert.ok(note.includes('基线未通过'));assert.ok(note.includes('固定引导'));
assert.ok(note.includes('数据不足'));assert.ok(note.includes('协议用时 / 补采上限'));
assert.ok(note.includes('20.0s / 1:40.0'));assert.ok(note.includes('实际墙钟用时'));
assert.ok(note.includes('1.0s'));assert.ok(note.includes('加速模拟'));
assert.equal(field('有效反馈秒数 / 协议时长'),'0.0s / 20.0s');
assert.equal(field('有效时间内达标率'),'—');
emit('effective_progress',{key:'training',effective_time:{target_seconds:50,valid_seconds:49.9,
  elapsed_seconds:50,max_seconds:100,completion_reason:'timing_issue',completed:false}});
assert.ok(all(card('有效时长训练进度')).some(n=>n.textContent.includes('计时记录不一致')));
assert.ok(all(card('有效时长训练进度')).some(n=>n.textContent.includes('未记为完成')));
''')

    def test_success_chart_preserves_missing_segment_gaps(self):
        self.run_view(r'''
const good=(seq,ratio)=>{const s=saved(seq,.8);s.duration_sec=10;
  Object.assign(s.stats,{valid_seconds:10,valid_time_ratio:1,on_target_valid_ratio:ratio,on_target_ratio:ratio});return s;};
const bad=saved(1,0,0);bad.duration_sec=10;
Object.assign(bad.stats,{valid_seconds:0,valid_time_ratio:0,on_target_valid_ratio:null,on_target_ratio:null});
response={segments:[good(0,1),bad,good(2,.5)],summary:{spec:'neurofeedback-v3'}};
session={status:'done',phase:'done',runs:[]};mount();await settle();
const success=all(card('训练分段')).find(n=>n.tagName==='PATH'&&n.attributes.stroke==='#1B3B6F');
assert.ok(success);assert.equal((success.attributes.d.match(/M/g)||[]).length,2);
assert.equal((success.attributes.d.match(/L/g)||[]).length,0,'missing middle segment breaks the performance line');
''')

    def test_baseline_features_are_not_labeled_as_normalized_scores(self):
        self.run_view(r'''
response = {segments:[saved(0,.8)],summary:{spec:'neurofeedback-v3',baseline_before:{
  valid:true,n_windows:15,device:'test',srate:250,indices:{focus:{mean:-.2,n:15}}}}};
session = {status:'done',phase:'done',runs:[]};
mount(); await settle();
const text=all(card('训练前后基线对比')).map(n=>n.textContent).join(' ');
assert.ok(text.includes('基线对数特征'));
assert.ok(text.includes('不是0–1评分'));
''')
