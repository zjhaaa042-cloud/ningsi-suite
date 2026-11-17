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
globalThis.window={setTimeout:(fn,ms)=>setTimeout(fn,ms===250?0:ms),clearTimeout,location:{search:''}};
const makeCtx=()=>{const cleanups=[];return {params:new URLSearchParams(),signal:{aborted:false},cleanups,
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
            "PROFILE_URI": "web/js/views/device-profile.js", "START_URI": "web/js/views/start.js",
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

    def test_device_profile_isolation_saves_selected_source_and_keeps_drafts_on_poll(self):
        self.node(r'''
import assert from 'node:assert/strict';
const {createDeviceProfile}=await import(PROFILE_URI);
const ctx=makeCtx();const completions={};const writes=[];
api.deviceProfile=key=>new Promise(resolve=>{completions[key]=resolve;});
api.saveDeviceProfile=async body=>{writes.push(body);return {data:{saved:true,profile:body.profile,missing_fields:[]}};};
const profile=createDeviceProfile(ctx);const sources=[{key:'lsl:A',real:true,is_eeg:true},{key:'lsl:B',real:true,is_eeg:true}];
profile.setSources(sources);
const source=id(profile.node,'device-profile-source');source.value='lsl:B';const selected=source.onchange();
completions['lsl:B']({data:{saved:true,profile:{manufacturer:'B厂商',source_unit:'uV',scale_to_uv:1},missing_fields:[]}});await selected;
completions['lsl:A']({data:{saved:true,profile:{manufacturer:'A厂商'},missing_fields:[]}});await flush();
assert.equal(id(profile.node,'device-profile-manufacturer').value,'B厂商');
const notes=id(profile.node,'device-profile-notes');notes.value='尚未保存的现场备注';profile.setSources(sources);
assert.equal(notes.value,'尚未保存的现场备注');
const unit=id(profile.node,'device-profile-source_unit');unit.value='ADC';unit.onchange();
await click(byText(profile.node,'保存设备档案'));assert.equal(writes.length,0);
assert.ok(text(profile.node).includes('必须填写校准后的'));
id(profile.node,'device-profile-scale_to_uv').value='0.5';await click(byText(profile.node,'保存设备档案'));
assert.equal(writes.length,1);assert.equal(writes[0].device,'lsl:B');assert.equal(writes[0].profile.scale_to_uv,.5);
assert.equal(writes[0].profile.notes,'尚未保存的现场备注');assert.ok(text(profile.node).includes('资料待核验'));
assert.ok(!text(profile.node).includes('验收通过'));ctx.cleanups.forEach(fn=>fn());
''')

    def test_start_waits_for_checks_and_requires_explicit_use_of_profile_units(self):
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
await click(byText(root,'采用档案单位'));await flush();
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
