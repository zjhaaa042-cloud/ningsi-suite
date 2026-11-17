"""Exercise competition actions and source/evidence labels in the native DOM."""
import json
from pathlib import Path
import shutil
import subprocess
import unittest

from .test_frontend_dom import NODE_STUB

ROOT = Path(__file__).resolve().parents[1]
VIEW = ROOT / "web/js/views/showcase.js"
ANALYSIS = ROOT / "web/js/views/showcase-analysis.js"

DOM = NODE_STUB + r'''
StubNode.prototype.append=function(...children){this.children.push(...children);};
Object.defineProperty(StubNode.prototype,'textContent',{
  get(){return this._text||'';},set(value){this._text=String(value);this.children=[];}
});
const all=node=>!node||typeof node==='string'?[]:[node,...(node.children||[]).flatMap(all)];
const text=node=>typeof node==='string'?node:(node?.textContent||'')+(node?.children||[]).map(text).join('');
StubNode.prototype.querySelector=function(selector){
  return all(this).find(node=>selector.startsWith('#')&&(node.id||node.attributes?.id)===selector.slice(1))||null;
};
StubNode.prototype.focus=function(){};
document.createElementNS=(ns,tag)=>new StubNode(tag);
document.querySelector=()=>null;
globalThis.window={setTimeout,clearTimeout,location:{search:''}};
const id=(root,name)=>all(root).find(node=>(node.id||node.attributes?.id)===name);
const byText=(root,label)=>all(root).find(node=>node.tagName==='BUTTON'&&text(node)===label);
const click=async node=>{if(!node)throw Error('Missing button');for(const listener of node.listeners.click||[])await listener({preventDefault(){}});};
const change=async node=>{for(const listener of node.listeners.change||[])await listener({});};
'''

FIXTURES = r'''
const {api}=await import(API_URI);
const calls={analyze:[],train:[],save:[],trend:[],demo:0};
const analysis={source:{label:'模拟 EEG 分析',simulated:true,kind:'sim'},rate:250,channels:1,
  timing:{compute_ms:4,window_sec:4,step_sec:2,filter_history_sec:2,lowpass_delay_ms:180},
  quality:{windows_total:7,windows_usable:6,valid_ratio:6/7,usable_seconds:14,reason_counts:{muscle_burst:1}},
  baseline:{valid:true},traces:{time_sec:[0,1,2,3,4],channels:[{label:'Ch1',raw_uv:[0,1,-1,2,0],filtered_uv:[0,.5,-.5,1,0],
    cleaned_uv:[0,null,-.5,1,0]}],masked_intervals:[{start_sec:.5,end_sec:1.5,reasons:['muscle_burst']}]},
  spectrum:{freqs_hz:[4,8,13,30],raw_psd_uv2_hz:[3,4,2,1],filtered_psd_uv2_hz:[2,3,1,.5]},
  bands:[{name:'theta',low_hz:4,high_hz:8,raw_power_uv2:4,filtered_power_uv2:3,relative_power:.2},
    {name:'alpha',low_hz:8,high_hz:13,raw_power_uv2:6,filtered_power_uv2:4,relative_power:.3},
    {name:'beta',low_hz:13,high_hz:30,raw_power_uv2:8,filtered_power_uv2:5,relative_power:.4}],
  latest:{usable:true,reasons:[],scores:{focus:.65,relax:.4,load:.45},time_domain:{rms_uv:5,ptp_uv:12,channel_corr_min:null}},windows:[]};
const session={uuid:'demo-existing',participant:'demo-p',status:'done',device:'sim-bsense',phase:'done',progress:1};
const report={participant:'demo-p',extras:{source_kind:'sim'},quality:{windows_total:35,windows_usable:35},
  indicators:{summary:{focus:{mean:.65},relax:{mean:.4},load:{mean:.45}}},
  scales:{SAS:{name:'焦虑自评量表',standard_score:42,level:'量表结果'}},
  behavior:{sart:{commission_rate:.05,omission_rate:.04},pvt:{rt_median:.3}},
  assessment:{conclusion:'保存的联合评估结论',advice:['保存的改善建议'],
    evidence:[{source:'eeg',summary:'保存的脑电证据',available:true},{source:'scales',summary:'保存的量表证据',available:true}]}};
api.showcaseMetadata=async()=>({data:{}});
api.showcaseEvidence=async()=>({data:{evidence_origin:'bundled_development_snapshot',snapshot_at:'2026-11-17',snapshot_note:'快照说明',
  requirements:[{id:'features',title:'PSD特征',status_label:'已实现',gaps:[],next_action:'核对固定频带'},
    {id:'trends',title:'历史趋势',status_label:'待验证',gaps:[]}],
  evidence:[{title:'完整记录',available:true,metrics:{windows_total:573,windows_usable:3,baseline_valid:false,paired:0},notes:['未取得有效基线']}],
  research:{conclusion:'不能代替目标设备验证'}}});
api.showcaseDeviceProfile=async()=>({data:{saved:false,profile:{},missing_fields:[{key:'reference',label:'参考电极'}]}});
api.saveShowcaseDeviceProfile=async body=>{calls.save.push(body);return {data:{saved:true,profile:body,missing_fields:[]}};};
api.showcaseAnalyze=async body=>{calls.analyze.push(body);return {data:analysis};};
api.listSessions=async()=>({data:{items:[session]}});
api.getSession=async uuid=>({data:{...session,uuid}});
api.report=async()=>({data:report});
api.sessionHeatmap=async()=>({data:{cells:[{index:2,label:'指数分档',color:'#6e9e8a',score:.65,t:4}],missing_color:'#666'}});
api.reportsTrend=async params=>{calls.trend.push(params);return {data:{sessions:1,points:[{period:'2026-W47',mean:.65}],note:'按实际保存记录汇总'}};};
api.humanModels=async()=>({data:{active:null}});
api.trainModel=async body=>{calls.train.push(body);return {data:{participants:['sim01','sim02','sim03'],samples:60,
  subject_split:{train:['sim01'],validation:['sim02'],test:['sim03']},train:{n:20,accuracy:.9,auc:.93},
  validation:{n:20,accuracy:.8,auc:.83},test:{n:20,accuracy:.7,auc:.73}}};};
api.createShowcaseDemo=async()=>{calls.demo++;return {data:{session:{...session,uuid:'new-demo',status:'running'}}};};
let cleanups=[];
const ctx={params:new URLSearchParams(),signal:{aborted:false},store:{state:{},setState(patch){Object.assign(this.state,patch);}},
  onCleanup(fn){cleanups.push(fn);},navigate(){}};
const {render,showcaseStepForRequirement}=await import(VIEW_URI);
'''


@unittest.skipUnless(shutil.which("node"), "Node required for native DOM interaction tests")
class ShowcaseFrontendTests(unittest.TestCase):
    def node(self, body, *, fixtures=True):
        program = DOM + (FIXTURES if fixtures else "") + body
        program = program.replace("API_URI", json.dumps((ROOT / "web/js/api.js").as_uri()))
        program = program.replace("VIEW_URI", json.dumps(VIEW.as_uri()))
        program = program.replace("ANALYSIS_URI", json.dumps(ANALYSIS.as_uri()))
        result = subprocess.run(["node", "--input-type=module", "-e", program],
                                capture_output=True, text=True, encoding="utf-8", timeout=25)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_selected_file_requires_units_and_forwards_only_signal_fields(self):
        self.node(r'''
import assert from 'node:assert/strict';
const {readRawFile}=await import(ANALYSIS_URI);
const make=data=>({size:100,text:async()=>JSON.stringify(data)});
const data={rate:250,scale_to_uv:1,samples:[[1,2]],baseline_samples:[[2,3]],baseline_timestamps:[0,.004],
  source_path:'C:/private/hidden',mode:'demo'};
const body=await readRawFile(make(data));
assert.equal(body.mode,'raw');assert.ok(!Object.hasOwn(body,'source_path'));
assert.deepEqual(body.baseline_timestamps,[0,.004]);
await assert.rejects(readRawFile(make({rate:250,samples:[[1,2]]})),/scale_to_uv/);
await assert.rejects(readRawFile({size:1024*1024+1}),/1 MiB/);
''', fixtures=False)

    def test_demo_charts_model_training_and_device_declaration_are_operable(self):
        self.node(r'''
import assert from 'node:assert/strict';
const root=new StubNode('main');await render(root,ctx);
assert.equal(all(root).filter(node=>node.attributes?.role==='tab').length,6);
assert.ok(text(root).includes('非当前设备实时测量'));
assert.ok(text(root).includes('开发证据快照'));
await click(id(root,'showcase-run-demo'));
assert.equal(calls.analyze.length,1);assert.equal(calls.analyze[0].mode,'demo');
assert.ok(text(root).includes('模拟 EEG 分析'));assert.ok(text(root).includes('污染片段屏蔽后'));
assert.ok(all(root).some(node=>node.tagName==='SVG'));
await click(id(root,'showcase-tab-features'));
assert.ok(text(root).includes('θ / α / β'));assert.ok(text(root).includes('0.650'));
await click(id(root,'showcase-tab-models'));
await click(byText(root,'运行模拟训练'));
assert.equal(calls.train.length,1);assert.ok(text(root).includes('非真人准确率'));
assert.ok(text(root).includes('70.0%'));
await click(id(root,'showcase-tab-hardware'));
id(root,'showcase-device-model').value='实际测试型号';
await click(byText(root,'保存设备核对资料'));
assert.equal(calls.save[0].model,'实际测试型号');assert.equal(calls.save[0].srate,null);
assert.ok(text(root).includes('不自动判定合规')||text(root).includes('仍需资料核验'));
assert.equal(showcaseStepForRequirement('assessment'),'joint');
assert.equal(showcaseStepForRequirement('visualization'),'preprocessing');
assert.equal(showcaseStepForRequirement('unknown'),null);
assert.ok(!text(root).includes('NaN'));assert.ok(!text(root).includes('undefined'));
cleanups.forEach(fn=>fn());
''')

    def test_saved_joint_report_and_participant_specific_trends_are_displayed(self):
        self.node(r'''
import assert from 'node:assert/strict';
const root=new StubNode('main');await render(root,ctx);
await click(id(root,'showcase-tab-joint'));
assert.ok(text(root).includes('保存的联合评估结论'));
assert.ok(text(root).includes('保存的脑电证据'));assert.ok(text(root).includes('保存的改善建议'));
await click(byText(root,'运行完整模拟流程'));
assert.equal(calls.demo,1);assert.equal(ctx.store.state.currentSession.uuid,'new-demo');
assert.ok(text(id(root,'showcase-session')).includes('demo-p · 已完成 · 模拟 · new-demo'));
await click(id(root,'showcase-tab-trends'));
const field=id(root,'showcase-trend-field');field.value='stress';await change(field);await Promise.resolve();
assert.ok(text(root).includes('本人压力自评'));
assert.ok(calls.trend.every(params=>params.participant==='demo-p'));
assert.ok(!text(root).includes('NaN'));
cleanups.forEach(fn=>fn());
''')

    def test_incomplete_reports_do_not_offer_unformed_downloads(self):
        self.node(r'''
import assert from 'node:assert/strict';
api.getSession=async uuid=>({data:{...session,uuid,status:'running'}});
api.report=async()=>({status:202,data:{...report,partial:true}});
const root=new StubNode('main');await render(root,ctx);
await click(id(root,'showcase-tab-joint'));
assert.ok(text(root).includes('打开完整报告'));
assert.ok(!text(root).includes('离线 HTML 报告'));
assert.ok(!text(root).includes('导出会话 ZIP'));
cleanups.forEach(fn=>fn());
''')

    def test_empty_workspace_does_not_aggregate_unselected_people_and_late_analysis_is_discarded(self):
        self.node(r'''
import assert from 'node:assert/strict';
api.listSessions=async()=>({data:{items:[]}});
const root=new StubNode('main');await render(root,ctx);
assert.equal(calls.trend.length,0);
await click(id(root,'showcase-tab-joint'));
assert.ok(text(root).includes('请先选择已有会话'));
let finish;
api.showcaseAnalyze=()=>new Promise(resolve=>{finish=resolve;});
const pending=click(id(root,'showcase-run-demo'));await Promise.resolve();
const before=text(root);cleanups.forEach(fn=>fn());
finish({data:analysis});await pending;
assert.equal(text(root),before);
''')
