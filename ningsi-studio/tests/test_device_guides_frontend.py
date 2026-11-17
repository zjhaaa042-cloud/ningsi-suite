"""Manual references stay readable while declaration seeds remain explicit drafts."""
import json
from pathlib import Path
import shutil
import subprocess
import unittest

from ningsi_studio.domain.device_guides import device_guides
from .test_readiness_frontend import DOM

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("node"), "Node required for device-guide DOM tests")
class DeviceGuidesFrontendTests(unittest.TestCase):
    def node(self, body):
        program = DOM + r'''
import assert from 'node:assert/strict';
ClassList.prototype.add=function(...names){names.forEach(name=>this.toggle(name,true));};
const {createDeviceProfile}=await import(PROFILE_URI);
const guides=GUIDES_JSON;
const at=(profile,name)=>id(profile.node,'device-profile-'+name);
const apply=profile=>all(profile.node).find(node=>node.classList?.contains('device-guide-apply'));
const feedback=profile=>all(profile.node).find(node=>node.classList?.contains('device-guide-feedback'));
const record=(profile={},saved=false)=>({data:{profile,saved,missing_fields:[{key:'reference',label:'参考电极'},
  {key:'source_unit',label:'源数值单位'}]}});
const ctxFor=(config={device_guides:guides})=>{
  const ctx=makeCtx();ctx.store.state.config=config;
  const listeners=new Set();ctx.store.subscribe=(keys,fn)=>{assert.equal(keys,'config');listeners.add(fn);return ()=>listeners.delete(fn);};
  ctx.config=next=>{ctx.store.state.config=next;for(const fn of [...listeners])fn(ctx.store.state,'config');};
  return ctx;
};
const real=[{key:'lsl:A',device:'BioMulti Lite EEG',real:true,is_eeg:true},
  {key:'lsl:B',device:'Another EEG',real:true,is_eeg:true}];
let writes=[];
api.saveDeviceProfile=async body=>{writes.push(body);return record(body.profile,true);};
''' + body
        for name, path in {
            "API_URI": "web/js/api.js", "PROFILE_URI": "web/js/views/device-profile.js",
        }.items():
            program = program.replace(name, json.dumps((ROOT / path).as_uri()))
        program = program.replace("GUIDES_JSON", json.dumps(device_guides(), ensure_ascii=False))
        result = subprocess.run(["node", "--input-type=module", "-e", program],
                                capture_output=True, text=True, encoding="utf-8", timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_readable_without_a_source_and_old_or_late_config_is_supported(self):
        self.node(r'''
api.deviceProfile=async()=>{throw Error('No source must be loaded automatically');};
const ctx=ctxFor({});const profile=createDeviceProfile(ctx);
assert.equal(apply(profile),undefined);
ctx.config({device_guides:guides});
assert.ok(text(profile.node).includes('BSense-R 手册参考'));
assert.ok(text(profile.node).includes('2 通道 · 250 Hz · cf_float32'));
assert.ok(text(profile.node).includes('不自动证明当前 LSL'));
assert.ok(text(profile.node).includes('Filters'));
assert.ok(text(profile.node).includes('不自动保存'));
assert.ok(text(profile.node).includes('手册未说明参考/地'));
assert.ok(text(profile.node).includes(guides[0].source.sha256));
assert.ok(text(profile.node).includes('PDF 第 23 页'));
assert.equal(apply(profile).disabled,true);
await click(apply(profile));assert.equal(at(profile,'model').value,'');assert.equal(writes.length,0);
ctx.config({});assert.equal(apply(profile),undefined);
ctx.cleanups.forEach(fn=>fn());
''')

    def test_seed_is_explicit_whitelisted_and_requires_separate_save(self):
        self.node(r'''
guides[0].declaration_seed={...guides[0].declaration_seed,manufacturer:'Invented manufacturer',
  electrodes:'Fp1,Fp2',reference:'A1',source_unit:'uV',scale_to_uv:1,srate:250,unit_basis:'Unverified'};
api.deviceProfile=async()=>record();
const ctx=ctxFor();const profile=createDeviceProfile(ctx);profile.setSources(real);await flush();
assert.equal(at(profile,'model').value,'','a matching-looking stream never applies a guide');
assert.equal(apply(profile).disabled,false);assert.equal(writes.length,0);
await click(apply(profile));
for(const key of ['model','document_reference','notes'])assert.equal(at(profile,key).value,guides[0].declaration_seed[key]);
for(const key of ['manufacturer','electrodes','reference','source_unit','scale_to_uv','srate','unit_basis'])
  assert.equal(at(profile,key).value,'',key+' remains unknown');
assert.ok(text(feedback(profile)).includes('尚未保存'));
assert.ok(text(profile.node).includes('尚未保存此设备的档案'));
assert.ok(text(profile.node).includes('待补资料：参考电极、源数值单位'));
assert.equal(writes.length,0);
await click(byText(profile.node,'保存设备档案'));
assert.equal(writes.length,1);assert.equal(writes[0].device,'lsl:A');
assert.equal(writes[0].profile.source_unit,'');assert.equal(writes[0].profile.srate,null);
assert.equal(writes[0].profile.reference,'');assert.equal(text(feedback(profile)),'');
assert.ok(text(profile.node).includes('档案已保存 · 资料待核验'));
ctx.cleanups.forEach(fn=>fn());
''')

    def test_saved_measurements_and_unsaved_edits_survive_seed_and_config_refresh(self):
        self.node(r'''
const saved={manufacturer:'Actual manufacturer',model:'Actual BSense-R setup',electrodes:'Actual EEG placement',
  reference:'Measured reference',srate:250,source_unit:'mV',scale_to_uv:1000,unit_basis:'Current firmware calibration',
  document_reference:'Original source evidence',notes:'Original physical setup'};
api.deviceProfile=async()=>record(saved,true);
const ctx=ctxFor();const profile=createDeviceProfile(ctx);profile.setSources(real);await flush();
at(profile,'notes').value='Unsaved contact adjustment';
at(profile,'document_reference').value='Unsaved independent calibration';
const snapshot=Object.fromEntries(Object.keys(saved).map(key=>[key,at(profile,key).value]));
ctx.config({device_guides:JSON.parse(JSON.stringify(guides))});
await click(apply(profile));
assert.deepEqual(Object.fromEntries(Object.keys(saved).map(key=>[key,at(profile,key).value])),snapshot);
assert.equal(writes.length,0);assert.ok(text(feedback(profile)).includes('已有填写已保留'));
at(profile,'document_reference').value='';await click(apply(profile));
assert.equal(at(profile,'document_reference').value,guides[0].declaration_seed.document_reference);
assert.equal(at(profile,'notes').value,'Unsaved contact adjustment');
assert.equal(at(profile,'model').value,'Actual BSense-R setup');
assert.ok(text(feedback(profile)).includes('尚未保存'));
ctx.cleanups.forEach(fn=>fn());
''')

    def test_loading_failure_and_simulation_cannot_apply_a_draft(self):
        self.node(r'''
let resolveLoad;api.deviceProfile=()=>new Promise(resolve=>{resolveLoad=resolve;});
const ctx=ctxFor();const profile=createDeviceProfile(ctx);profile.setSources(real);
assert.equal(apply(profile).disabled,true);await click(apply(profile));assert.equal(at(profile,'model').value,'');
resolveLoad(record());await flush();assert.equal(apply(profile).disabled,false);
profile.setSources([{...real[0],simulated:true}]);
assert.equal(apply(profile).disabled,true);await click(apply(profile));assert.equal(at(profile,'model').value,'');
api.deviceProfile=async()=>{throw Error('Profile read unavailable');};
at(profile,'key').value='lsl:B';await click(byText(profile.node,'读取该设备档案'));
assert.equal(apply(profile).disabled,true);await click(apply(profile));assert.equal(at(profile,'model').value,'');
assert.ok(text(profile.node).includes('设备档案读取未完成'));assert.equal(writes.length,0);
api.deviceProfile=async()=>record();
at(profile,'key').value='sim-bsense';await click(byText(profile.node,'读取该设备档案'));
assert.equal(apply(profile).disabled,true);await click(apply(profile));assert.equal(writes.length,0);
assert.equal(at(profile,'model').value,'');ctx.cleanups.forEach(fn=>fn());
''')

    def test_source_changes_clear_draft_messages_and_ignore_stale_loads(self):
        self.node(r'''
const pending={};api.deviceProfile=key=>new Promise(resolve=>{pending[key]=resolve;});
const ctx=ctxFor();const profile=createDeviceProfile(ctx);profile.setSources(real);
at(profile,'source').value='lsl:B';const loadB=at(profile,'source').onchange();
pending['lsl:B'](record({model:'Actual B model'},true));await loadB;
await click(apply(profile));assert.ok(text(feedback(profile)).includes('尚未保存'));
assert.equal(at(profile,'model').value,'Actual B model');
pending['lsl:A'](record({model:'Stale A model'},true));await flush();
assert.equal(at(profile,'model').value,'Actual B model','late load cannot replace the chosen source');
at(profile,'source').value='lsl:A';const loadA=at(profile,'source').onchange();
assert.equal(text(feedback(profile)),'');assert.equal(apply(profile).disabled,true);
pending['lsl:A'](record());await loadA;
assert.equal(at(profile,'notes').value,'','seeded notes do not leak to another source');
assert.equal(at(profile,'document_reference').value,'');assert.equal(writes.length,0);
ctx.cleanups.forEach(fn=>fn());
''')

    def test_saving_and_disposal_block_repeat_seed_actions(self):
        self.node(r'''
api.deviceProfile=async()=>record();
let finishSave;api.saveDeviceProfile=body=>{writes.push(body);return new Promise(resolve=>{finishSave=resolve;});};
const ctx=ctxFor();const profile=createDeviceProfile(ctx);profile.setSources(real);await flush();
await click(apply(profile));const pendingSave=click(byText(profile.node,'保存设备档案'));
assert.equal(apply(profile).disabled,true);
at(profile,'model').value='';await click(apply(profile));assert.equal(at(profile,'model').value,'');
finishSave(record(writes[0].profile,true));await pendingSave;
assert.equal(at(profile,'model').value,'BSense-R');assert.equal(writes.length,1);
ctx.cleanups.forEach(fn=>fn());at(profile,'model').value='';await click(apply(profile));
assert.equal(at(profile,'model').value,'');
const before=profile.node.children;ctx.config({device_guides:[]});
assert.equal(profile.node.children,before,'a disposed view no longer receives configuration callbacks');
''')
