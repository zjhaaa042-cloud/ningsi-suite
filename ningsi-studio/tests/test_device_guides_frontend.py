"""Device connections use documented parameters without a second data-entry workflow."""
import json
from pathlib import Path
import shutil
import subprocess
import unittest

from .test_readiness_frontend import DOM

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("node"), "Node required for device connection DOM tests")
class DeviceConnectionsFrontendTests(unittest.TestCase):
    def node(self, body):
        program = DOM + r'''
import assert from 'node:assert/strict';
const {devices}=await import(PANELS_URI);
const {createSessionPreflight}=await import(PREFLIGHT_URI);
api.deviceProfile=async()=>{throw Error('Device connections no longer read operator profiles');};
api.saveDeviceProfile=async()=>{throw Error('Device connections must not write operator profiles');};
api.devicePreview=async()=>({data:{active:false,note:'未开始预览：当前没有脑机信号'}});
const configured={id:'bsense-r',model:'BSense-R',channels:2,srate:250,source_unit:'uV',scale_to_uv:1};
const source={key:'lsl:BioMulti Lite EEG-Review',device:'BioMulti Lite EEG-Review',
  kind:'lsl',real:true,is_eeg:true,srate:250,channels:2,note:'EEG',device_configuration:configured};
const response=sources=>({data:{devices:[],sources,active_sessions:[]}});
const parameter=root=>all(root).find(node=>node.classList?.contains('device-parameters'));
''' + body
        for name, path in {
            "API_URI": "web/js/api.js", "PANELS_URI": "web/js/views/panels.js",
            "PREFLIGHT_URI": "web/js/views/session-preflight.js",
        }.items():
            program = program.replace(name, json.dumps((ROOT / path).as_uri()))
        result = subprocess.run(["node", "--input-type=module", "-e", program],
                                capture_output=True, text=True, encoding="utf-8", timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_connections_show_waveform_and_one_parameter_line_without_profile_workflow(self):
        self.node(r'''
api.deviceStatus=async()=>response([source]);
const ctx=makeCtx();const root=new StubNode('main');await devices.render(root,ctx);await flush();
assert.equal(parameter(root).hidden,false);
assert.equal(text(parameter(root)),'BSense-R · EEG 2 通道 · 250 Hz · µV');
assert.ok(text(root).includes('设备实时预览'));assert.ok(text(root).includes('可用数据源'));
assert.ok(byText(root,'开始预览（自动选检测到的设备）'));
for(const label of ['设备档案','保存设备档案','待补资料','填写或修改设备资料','手册参考','补充草稿'])
  assert.ok(!text(root).includes(label),label+' is absent');
assert.ok(!all(root).some(node=>['FIELDSET','TEXTAREA','INPUT'].includes(node.tagName)));
const headings=all(root).filter(node=>node.tagName==='H3').map(text);
assert.ok(headings.indexOf('设备实时预览')<headings.indexOf('可用数据源'),'preview precedes source diagnostics');
assert.ok(!text(root).includes('当前没有正在运行的会话占用信号源'),'no empty health card');
ctx.cleanups.forEach(fn=>fn());
''')

    def test_unknown_simulated_or_mismatched_sources_are_not_labelled_as_documented_bsense(self):
        self.node(r'''
const unconfigured={...source,device_configuration:null};
for(const sources of [[],[{...unconfigured,simulated:true}],[{...unconfigured,srate:256}],
  [{...unconfigured,channels:4}],[{...unconfigured,nominal_srate:0}],
  [{...unconfigured,declared_units:['mV','mV']}],
  [{...unconfigured,device:'Other EEG',key:'lsl:Other EEG'}]]){
  api.deviceStatus=async()=>response(sources);
  const ctx=makeCtx();const root=new StubNode('main');await devices.render(root,ctx);await flush();
  assert.equal(parameter(root).hidden,true);assert.equal(text(parameter(root)),'');
  assert.ok(text(root).includes('设备实时预览'));assert.ok(!text(root).includes('保存设备档案'));
  ctx.cleanups.forEach(fn=>fn());
}
''')

    def test_bsense_name_uses_the_same_backend_configuration_without_client_name_rules(self):
        self.node(r'''
api.deviceStatus=async()=>response([{...source,device:'BSense-R EEG-Review',key:'lsl:BSense-R EEG-Review'}]);
const ctx=makeCtx();const root=new StubNode('main');await devices.render(root,ctx);await flush();
assert.equal(parameter(root).hidden,false);
assert.equal(text(parameter(root)),'BSense-R · EEG 2 通道 · 250 Hz · µV');
ctx.cleanups.forEach(fn=>fn());
''')

    def test_connection_failure_does_not_resurrect_registration_or_fake_parameters(self):
        self.node(r'''
api.deviceStatus=async()=>{throw Object.assign(Error('unavailable'),{status:401});};
const ctx=makeCtx();const root=new StubNode('main');await devices.render(root,ctx);await flush();
assert.equal(parameter(root).hidden,true);assert.ok(text(root).includes('没有权限读取设备状态'));
assert.ok(!text(root).includes('设备档案'));assert.ok(!text(root).includes('待补资料'));
ctx.cleanups.forEach(fn=>fn());
''')

    def test_preflight_ignores_old_profile_actions_but_keeps_real_unit_and_model_gates(self):
        self.node(r'''
api.sessionPreflight=async body=>({data:{ready:body.device==='lsl:documented',
  device_profile:{saved:true,profile:{source_unit:'uV',scale_to_uv:1}},
  scoring:{attention_model:{status:'incompatible',message:'模型配置不匹配'}},
  checks:[{key:'device_profile',status:'warning',label:'设备资料',message:'请填写设备档案'},
    {key:'units',status:body.device==='lsl:documented'?'pass':'fail',label:'单位与标定',message:'单位不一致'}]}});
const ctx=makeCtx();const check=createSessionPreflight(ctx);
await check.refresh({device:'lsl:documented'},{immediate:true});assert.equal(check.ready,true);
assert.ok(text(check.node).includes('模型配置不匹配'));
for(const label of ['采用档案单位','设备连接与档案','请填写设备档案'])assert.ok(!text(check.node).includes(label));
assert.ok(all(check.node).some(node=>node.tagName==='A'&&text(node)==='设备连接'));
await check.refresh({device:'lsl:other'},{immediate:true});assert.equal(check.ready,false);
assert.ok(text(check.node).includes('单位不一致'));ctx.cleanups.forEach(fn=>fn());
''')
