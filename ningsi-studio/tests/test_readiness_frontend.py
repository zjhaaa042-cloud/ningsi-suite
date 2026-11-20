"""Exercise source isolation and start-time gating in the product's native DOM modules."""
import json
from pathlib import Path
import shutil
import subprocess
import unittest

from .test_frontend_dom import NODE_STUB

ROOT = Path(__file__).resolve().parents[1]

DOM = NODE_STUB + r'''
StubNode.prototype.append=function(...children){this.children.push(...children);};
StubNode.prototype.replaceChildren=function(...children){this.children=[];this.append(...children);};
StubNode.prototype.add=function(child){this.append(child);};
Object.defineProperty(StubNode.prototype,'textContent',{
  get(){return this._text||'';},set(value){this._text=String(value);this.children=[];}
});
Object.defineProperty(StubNode.prototype,'value',{
  get(){if(this._value!==undefined)return this._value;
    if(this.tagName==='SELECT')return this.children.find(node=>node.hasAttribute?.('selected'))?.value??this.children[0]?.value??'';
    return this.attributes.value||'';},set(value){this._value=String(value);}
});
Object.defineProperty(StubNode.prototype,'valueAsNumber',{get(){return this.value===''?NaN:Number(this.value);}});
Object.defineProperty(StubNode.prototype,'options',{get(){return this.children.filter(node=>node.tagName==='OPTION');}});
document.createElement=tag=>{const node=new StubNode(tag);
  if(node.tagName==='TEXTAREA')Object.defineProperty(node,'type',{get(){return 'textarea';}});
  return node;};
const all=node=>!node||typeof node==='string'?[]:[node,...(node.children||[]).flatMap(all)];
const text=node=>typeof node==='string'?node:(node?.textContent||'')+(node?.children||[]).map(text).join('');
const id=(root,name)=>all(root).find(node=>(node.id||node.attributes?.id)===name);
const byText=(root,label)=>all(root).find(node=>node.tagName==='BUTTON'&&text(node)===label);
const click=async node=>{if(!node)throw Error('Missing button');for(const listener of node.listeners.click||[])await listener({preventDefault(){}});};
const flush=()=>new Promise(resolve=>setTimeout(resolve,8));
document.querySelector=()=>null;
globalThis.window={setTimeout:(fn,ms)=>setTimeout(fn,ms===250?0:ms),clearTimeout,setInterval,clearInterval,location:{search:''}};
const makeCtx=()=>{const controller=new AbortController();const cleanups=[()=>controller.abort()];
  return {params:new URLSearchParams(),signal:controller.signal,cleanups,
  store:{state:{},setState(patch){Object.assign(this.state,patch);}},
  onCleanup(fn){cleanups.push(fn);},navigate(hash){this.destination=hash;},reload(){}};};
const {api}=await import(API_URI);
'''


@unittest.skipUnless(shutil.which("node"), "Node required for native DOM interaction tests")
class ReadinessFrontendTests(unittest.TestCase):
    def node(self, body):
        program = DOM + body
        for name, path in {
            "API_URI": "web/js/api.js", "PREFLIGHT_URI": "web/js/views/session-preflight.js",
            "START_URI": "web/js/views/start.js",
        }.items():
            program = program.replace(name, json.dumps((ROOT / path).as_uri()))
        result = subprocess.run(["node", "--input-type=module", "-e", program],
                                capture_output=True, text=True, encoding="utf-8", timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_preflight_discards_old_device_responses_and_disposed_requests(self):
        self.node(r'''
import assert from 'node:assert/strict';
const {createSessionPreflight}=await import(PREFLIGHT_URI);
const ctx=makeCtx();const completions={};
api.sessionPreflight=body=>new Promise(resolve=>{completions[body.device]=resolve;});
const check=createSessionPreflight(ctx);
const old=check.refresh({device:'lsl:old',protocol:'short'},{immediate:true});
const current=check.refresh({device:'lsl:current',protocol:'full'},{immediate:true});
completions['lsl:current']({data:{ready:false,scoring:{attention_model:{status:'incompatible',model_id:'current-v2',message:'单位不匹配'}},checks:[]}});
await current;
completions['lsl:old']({data:{ready:true,scoring:{attention_model:{status:'compatible',model_id:'old-v1'}},checks:[]}});
await old;
assert.equal(check.ready,false);assert.equal(check.data.scoring.attention_model.model_id,'current-v2');
assert.ok(text(check.node).includes('模型暂停'));assert.ok(!text(check.node).includes('old-v1'));
const last=check.refresh({device:'lsl:later',protocol:'short'},{immediate:true});
const before=text(check.node);ctx.cleanups.forEach(fn=>fn());
completions['lsl:later']({data:{ready:true,checks:[]}});await last;
assert.equal(text(check.node),before);assert.equal(check.ready,false);
''')

    def test_preflight_errors_are_visible_and_retryable(self):
        self.node(r'''
import assert from 'node:assert/strict';
const {createSessionPreflight}=await import(PREFLIGHT_URI);
const ctx=makeCtx();let calls=0;
api.sessionPreflight=async()=>{if(++calls===1)throw Error('offline');return {data:{ready:true,scoring:{attention_model:{status:'none'}},checks:[]}};};
const check=createSessionPreflight(ctx);await check.refresh({device:'lsl:test',protocol:'short'},{immediate:true});
assert.equal(check.ready,false);assert.ok(text(check.node).includes('开始前检查失败'));
await click(byText(check.node,'重新检查'));
assert.equal(check.ready,true);assert.ok(text(check.node).includes('未启用真人模型'));
assert.ok(text(check.node).includes('训练反馈沿用相对基线专注指数'));
ctx.cleanups.forEach(fn=>fn());
''')

    def test_unknown_device_keeps_existing_explicit_unit_selection(self):
        self.node(r'''
import assert from 'node:assert/strict';
const {render}=await import(START_URI);const ctx=makeCtx();const bodies=[],sessions=[];
api.devices=async()=>({data:{sources:[{key:'lsl:headset',device:'Headset',real:true,is_eeg:true}]}});
api.listSubjects=async()=>({data:{items:[],total:0}});
api.config=async()=>({data:{profiles:[{key:'ten_min',total_sec:600},{key:'short',total_sec:960},{key:'full',total_sec:1500}]}});
api.sessionPreflight=async body=>{bodies.push(body);return {data:{ready:body.eeg_calibration?.source_unit==='uV',calibration:body.eeg_calibration,
  device_profile:{saved:true,profile:{source_unit:'uV',scale_to_uv:1}},checks:[{key:'units',status:body.eeg_calibration?'pass':'fail',label:'输入单位',message:'本次设备单位需确认'}],
  scoring:{relative_indices:{status:'pending'},attention_model:{status:'incompatible',model_id:'frozen-model',message:'模型通道不匹配，原指数仍可评估'}}}};};
api.createSubject=async()=>({data:{public_id:'new01'}});
api.createSession=async body=>{sessions.push(body);return {data:{session:{uuid:'session1'},phases:[],runtime:{}}};};
const root=new StubNode('main');await render(root,ctx);await flush();
assert.equal(byText(root,'开始评估').disabled,true);
assert.equal(id(root,'eeg-source-unit').value,'');assert.ok(text(root).includes('模型暂停'));
id(root,'eeg-source-unit').value='uV';id(root,'eeg-source-unit').onchange();await flush();
assert.equal(id(root,'eeg-source-unit').value,'uV');assert.equal(byText(root,'开始评估').disabled,false);
await click(byText(root,'开始评估'));
assert.equal(sessions.length,1);assert.deepEqual(sessions[0].eeg_calibration,{source_unit:'uV'});
assert.equal(sessions[0].device,'lsl:headset');assert.ok(ctx.destination.startsWith('#/flow?session=session1'));
assert.ok(text(root).includes('需要开始采集后确认'));ctx.cleanups.forEach(fn=>fn());
''')

    def test_no_device_keeps_real_start_disabled_and_demo_explicit(self):
        self.node(r'''
import assert from 'node:assert/strict';
const {render}=await import(START_URI);const ctx=makeCtx();const sessions=[];
api.devices=async()=>({data:{sources:[]}});api.listSubjects=async()=>({data:{items:[],total:0}});
api.config=async()=>({data:{profiles:[{key:'ten_min',total_sec:600},{key:'short',total_sec:960}]}});
api.sessionPreflight=async()=>{throw Error('No device must not be checked');};
api.createSubject=async()=>({data:{public_id:'demo01'}});
api.createSession=async body=>{sessions.push(body);return {data:{session:{uuid:'demo-session'},phases:[],runtime:{}}};};
const root=new StubNode('main');await render(root,ctx);await flush();
assert.equal(byText(root,'开始评估').disabled,true);assert.equal(byText(root,'运行模拟演示').disabled,false);
await click(byText(root,'运行模拟演示'));
assert.equal(sessions[0].device,'sim-bsense');assert.equal(sessions[0].eeg_calibration,null);
ctx.cleanups.forEach(fn=>fn());
''')

    def test_personal_calibration_subject_creation_preserves_later_selection(self):
        self.node(r'''
import assert from 'node:assert/strict';
const {render}=await import(START_URI);const ctx=makeCtx();const calibrations=[];
api.devices=async()=>({data:{sources:[{key:'lsl:headset',device:'Headset',real:true,is_eeg:true}]}});
api.listSubjects=async()=>({data:{items:[{public_id:'existing-person'}],total:1}});
api.config=async()=>({data:{profiles:[{key:'ten_min',total_sec:600},{key:'short',total_sec:960}]}});
api.sessionPreflight=async body=>({data:{ready:true,checks:[],calibration:body.eeg_calibration}});
let resolveSubject;
api.createSubject=()=>new Promise(resolve=>{resolveSubject=resolve;});
api.blinkCalibration=async body=>{calibrations.push(body);return {data:{token:'existing-token',state:'collecting',count:0}};};
const root=new StubNode('main');await render(root,ctx);await flush();
id(root,'alias').value='requested label';
const pending=click(byText(root,'开始个人眨眼校准'));await flush();
id(root,'participant').value='existing-person';id(root,'participant').onchange();
id(root,'alias').value='later label';
resolveSubject({data:{public_id:'created-person'}});await pending;
assert.equal(id(root,'participant').value,'existing-person','subject creation must not replace the selection made while waiting');
assert.equal(calibrations.length,0,'a changed selection requires a new explicit calibration start');
const created=id(root,'participant').options.find(item=>item.value==='created-person');
assert.ok(text(created).includes('requested label'));assert.ok(!text(created).includes('later label'));
await click(byText(root,'开始个人眨眼校准'));
assert.equal(calibrations.length,1);assert.equal(calibrations[0].participant,'existing-person');
ctx.cleanups.forEach(fn=>fn());
''')

    def test_documented_device_uses_server_units_and_does_not_leak_them_to_another_source(self):
        self.node(r'''
import assert from 'node:assert/strict';
const {render}=await import(START_URI);const ctx=makeCtx();const checks=[],sessions=[];
const vendor='lsl:BioMulti Lite EEG-Documented';
api.devices=async()=>({data:{sources:[{key:vendor,device:'BioMulti Lite EEG-Documented',real:true,is_eeg:true},
  {key:'lsl:OtherEEG',device:'Other EEG',real:true,is_eeg:true}]}});
api.listSubjects=async()=>({data:{items:[{public_id:'p01'}],total:1}});
api.config=async()=>({data:{profiles:[{key:'ten_min',total_sec:600},{key:'short',total_sec:960}]}});
api.sessionPreflight=async body=>{checks.push(body);return {data:{device:body.device,ready:body.device===vendor,
  calibration:body.device===vendor?{source_unit:'uV',scale_to_uv:1,origin:'manufacturer_manual'}:null,
  checks:[],scoring:{attention_model:{status:'none'}}}};};
api.createSession=async body=>{sessions.push(body);return {data:{session:{uuid:'manual-session'},phases:[],runtime:{}}};};
const root=new StubNode('main');await render(root,ctx);await flush();
const unitPanel=all(root).find(node=>node.tagName==='DETAILS'&&text(node).includes('单位与标定设置'));
id(root,'participant').value='p01';id(root,'participant').onchange();await flush();
assert.equal(unitPanel.hidden,true,'documented hardware does not ask the operator to re-enter units');
assert.equal(byText(root,'开始评估').disabled,false);
await click(byText(root,'开始评估'));
assert.deepEqual(sessions[0].eeg_calibration,{source_unit:'uV',scale_to_uv:1,origin:'manufacturer_manual'});
id(root,'device').value='lsl:OtherEEG';id(root,'device').onchange();await flush();
assert.equal(checks.at(-1).eeg_calibration,null,'manual units stay bound to their matching device');
assert.equal(unitPanel.hidden,false);assert.equal(byText(root,'开始评估').disabled,true);
ctx.cleanups.forEach(fn=>fn());
''')

    def test_blink_draft_survives_preflight_refresh_and_retryable_error(self):
        self.node(r'''
import assert from 'node:assert/strict';
const {render}=await import(START_URI);const ctx=makeCtx(), root=new StubNode('main');
const vendor='lsl:BioMulti Lite EEG-Documented';
api.devices=async()=>({data:{sources:[{key:vendor,device:'BioMulti Lite EEG-Documented',real:true,is_eeg:true}]}});
api.listSubjects=async()=>({data:{items:[],total:0}});
api.createSubject=async()=>({data:{public_id:'new-person'}});
api.config=async()=>({data:{profiles:[{key:'ten_min',total_sec:600}]}});
api.sessionPreflight=async body=>({data:{device:body.device,ready:true,checks:[],
  calibration:{source_unit:'uV',scale_to_uv:1,origin:'manufacturer_manual'}}});
let tokenSeen, attempts=0;
api.blinkCalibration=async body=>{
  if(body.action==='start') {
    assert.equal(body.eeg_calibration,null);
    await flush();
    return {data:{token:'draft',state:'collecting',count:0,minimum_events:8}};
  }
  tokenSeen=body.token;
  if(++attempts===1) throw Object.assign(new Error('本次眨眼附近波形出现突变。'),{
    status:422,detail:{retryable:true,accepted_count:0},
  });
  return {data:{token:'draft',state:'collecting',count:1,minimum_events:8}};
};
await render(root,ctx);await flush();
await click(byText(root,'开始个人眨眼校准'));
const mark=byText(root,'已眨眼，记录这一次');
assert.equal(mark.disabled,false,'a refreshed manufacturer calibration must not discard the draft');
await click(mark);
assert.ok(text(root).includes('已保留 0 次'));assert.equal(mark.disabled,false);
await click(mark);
assert.equal(tokenSeen,'draft');assert.ok(text(root).includes('已通过单次筛查 1 次'));
ctx.cleanups.forEach(fn=>fn());
''')

    def test_rescan_rechecks_configuration_for_same_device_name(self):
        self.node(r'''
import assert from 'node:assert/strict';
const {render}=await import(START_URI); const ctx=makeCtx(), root=new StubNode('main');
const source={key:'lsl:BioMulti Lite EEG', device:'BioMulti Lite EEG', real:true, is_eeg:true};
let probes=0;
api.devices=async()=>{probes++;return {data:{sources:[source]}};};
api.listSubjects=async()=>({data:{items:[],total:0}});
api.config=async()=>({data:{profiles:[{key:'ten_min',total_sec:600}]}});
api.sessionPreflight=async body=>({data:{device:body.device,ready:probes===1,
  calibration:probes===1?{source_unit:'uV',scale_to_uv:1,origin:'manufacturer_manual'}:null,checks:[]}});
await render(root,ctx);await flush();
assert.equal(byText(root,'开始评估').disabled,false);
await click(byText(root,'重新扫描'));await flush();
assert.equal(byText(root,'开始评估').disabled,true,'a changed device can reuse the same stream name');
ctx.cleanups.forEach(fn=>fn());
''')

    def test_effective_time_is_an_explicit_plan_and_fixed_ten_minutes_remain_selectable(self):
        self.node(r'''
import assert from 'node:assert/strict';
const {render}=await import(START_URI);const ctx=makeCtx();const checks=[],sessions=[];
api.devices=async()=>({data:{sources:[{key:'lsl:headset',device:'Headset',real:true,is_eeg:true}]}});
api.listSubjects=async()=>({data:{items:[{public_id:'p01'}],total:1}});
api.config=async()=>({data:{profiles:[{key:'ten_min',total_sec:600},
  {key:'effective_time',total_sec:600,max_acquisition_seconds:1040},{key:'short',total_sec:960},{key:'full',total_sec:1500}]}});
api.sessionPreflight=async body=>{checks.push(body);return {data:{ready:true,checks:[],calibration:{source_unit:'uV'}}};};
api.createSubject=async()=>({data:{public_id:'p02'}});
api.createSession=async body=>{sessions.push(body);return {data:{session:{uuid:'session'+sessions.length},phases:[],runtime:{}}};};
const root=new StubNode('main');await render(root,ctx);await flush();
const choice=value=>all(root).find(node=>node.tagName==='INPUT'&&node.value===value);
assert.equal(choice('ten_min').checked,true,'fixed ten minutes remains the default');
assert.ok(text(root).includes('不追加基线时间'));
choice('effective_time').onchange();await flush();
assert.equal(checks.at(-1).protocol,'effective_time');
assert.ok(text(root).includes('4 轮各累计 50 秒有效反馈'));
assert.ok(text(root).includes('补采上限约17分20秒'));
assert.ok(text(root).includes('设备读取等待按实际记录'));
assert.equal(byText(root,'开始有效时长训练').disabled,false);
await click(byText(root,'开始有效时长训练'));
assert.equal(sessions[0].protocol,'effective_time');assert.equal(sessions[0].time_scale,1);
choice('ten_min').onchange();await flush();
assert.ok(text(root).includes('不追加基线时间'));
await click(byText(root,'开始评估'));
assert.equal(sessions[1].protocol,'ten_min');
ctx.cleanups.forEach(fn=>fn());
''')
