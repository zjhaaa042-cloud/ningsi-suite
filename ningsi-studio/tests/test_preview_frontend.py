"""Preview navigation and delayed HTTP responses must not freeze or revive UI."""
import json
from pathlib import Path
import shutil
import subprocess
import unittest

from .test_readiness_frontend import DOM

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which('node'), 'Node required for native DOM interaction tests')
class DevicePreviewFrontendTests(unittest.TestCase):
    def node(self, body):
        program = DOM + r'''
import assert from 'node:assert/strict';
const {renderPreviewCard}=await import(PREVIEW_URI);
let nextTimer=0;const timers=new Map();
window.setInterval=(fn)=>{const id=++nextTimer;timers.set(id,fn);return id;};
window.clearInterval=id=>timers.delete(id);
const active={active:true,source:'sim-bsense',kind:'sim',srate:250,channels:1};
''' + body
        for token, path in {'API_URI':'web/js/api.js', 'PREVIEW_URI':'web/js/preview.js'}.items():
            program = program.replace(token, json.dumps((ROOT/path).as_uri()))
        result = subprocess.run(['node','--input-type=module','-e',program],
                                capture_output=True, text=True, encoding='utf-8', timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_reentering_an_active_preview_resumes_fetching_windows(self):
        self.node(r'''
let windows=0;
api.devicePreview=async()=>({data:active});
api.devicePreviewWindow=async()=>{windows++;return {data:{...active,samples:[]}};};
const root=new StubNode('main'),ctx=makeCtx();renderPreviewCard(ctx,root);await flush();
assert.ok(byText(root,'停止预览'));assert.equal(timers.size,1);
await [...timers.values()][0]();
assert.equal(windows,1);assert.equal(timers.size,1,'one poller per component');
ctx.cleanups.forEach(fn=>fn());assert.equal(timers.size,0);
''')

    def test_failed_stop_reads_back_state_and_preserves_retry_controls(self):
        self.node(r'''
let state = active;
api.devicePreview = async () => ({data:state});
api.stopDevicePreview = async () => {throw Error('reply lost');};
const root=new StubNode('main'),ctx=makeCtx();renderPreviewCard(ctx,root);await flush();
await click(byText(root,'停止预览'));
assert.ok(byText(root,'停止预览'));assert.equal(timers.size,1);
assert.ok(text(root).includes('停止请求未确认'));assert.ok(!text(root).includes('预览已停止'));
api.stopDevicePreview = async () => {state={active:false};throw Error('reply lost');};
await click(byText(root,'停止预览'));
assert.ok(!byText(root,'停止预览'));assert.equal(timers.size,0);
ctx.cleanups.forEach(fn=>fn());
''')

    def test_disconnected_window_clears_stale_quality_then_recovers(self):
        self.node(r'''
api.devicePreview=async()=>({data:active});
const frame={...active,quality:{ok:true},peak_uv:37,std_uv:4};
api.devicePreviewWindow=async()=>({data:frame});
const root=new StubNode('main'),ctx=makeCtx();renderPreviewCard(ctx,root);await flush();
const tick=[...timers.values()][0];await tick();
assert.ok(text(root).includes('本窗质检：可用'));
api.devicePreviewWindow=async()=>{throw Error('offline');};await tick();
assert.ok(text(root).includes('offline'));assert.ok(!text(root).includes('本窗质检：可用'));
api.devicePreviewWindow=async()=>({data:{...active,window_error:'读取失败：单位冲突'}});
await tick();assert.ok(text(root).includes('单位冲突'));assert.ok(!text(root).includes('没有取到样本'));
api.devicePreviewWindow=async()=>({data:frame});await tick();assert.ok(text(root).includes('本窗质检：可用'));
api.devicePreviewWindow=async()=>({data:{...frame,kind:'lsl',simulated:true}});await tick();
assert.ok(text(root).includes('仿真源，仅演示'));assert.ok(!text(root).includes('（实时设备）'));
ctx.cleanups.forEach(fn=>fn());
''')

    def test_late_window_response_cannot_revive_a_stopped_preview(self):
        self.node(r'''
api.devicePreview=async()=>({data:{active:false}});
api.startDevicePreview=async()=>({data:active});
api.stopDevicePreview=async()=>({data:{active:false}});
let resolveWindow;api.devicePreviewWindow=()=>new Promise(resolve=>{resolveWindow=resolve;});
const root=new StubNode('main'),ctx=makeCtx();renderPreviewCard(ctx,root);await flush();
await click(byText(root,'预览仿真源（仅演示）'));
const refresh=[...timers.values()][0]();await Promise.resolve();
await click(byText(root,'停止预览'));
assert.ok(text(root).includes('预览已停止'));assert.equal(timers.size,0);
resolveWindow({data:active});await refresh;
assert.ok(text(root).includes('预览已停止'));assert.ok(!byText(root,'停止预览'));
assert.equal(timers.size,0);ctx.cleanups.forEach(fn=>fn());
''')

    def test_changing_device_discards_a_pending_start_response(self):
        self.node(r'''
api.devicePreview=async()=>({data:{active:false}});
let finish;api.startDevicePreview=()=>new Promise(resolve=>{finish=resolve;});
let selected='lsl:first';const root=new StubNode('main'),ctx=makeCtx();
const preview=renderPreviewCard(ctx,root,{getSource:()=>selected});await flush();
const pending=click(byText(root,'预览所选设备'));
selected='lsl:second';preview.selectionChanged();
assert.equal(byText(root,'正在连接…').disabled,true,'serialize mutations across selection changes');
finish({data:{...active,kind:'lsl',source:'lsl:first'}});await pending;
assert.equal(timers.size,0);assert.ok(!text(root).includes('正在接收信号'));
assert.equal(byText(root,'预览所选设备').disabled,false);
ctx.cleanups.forEach(fn=>fn());
''')

    def test_another_selected_source_is_not_polled_and_slow_windows_do_not_overlap(self):
        self.node(r'''
api.devicePreview=async()=>({data:active});
let windows=0,finish;api.devicePreviewWindow=()=>{windows++;return new Promise(resolve=>{finish=resolve;});};
const unrelated=new StubNode('main'),otherCtx=makeCtx();
renderPreviewCard(otherCtx,unrelated,{getSource:()=> 'lsl:other'});await flush();
assert.equal(timers.size,0);assert.ok(text(unrelated).includes('当前预览与所选设备不同'));
otherCtx.cleanups.forEach(fn=>fn());
const root=new StubNode('main'),ctx=makeCtx();renderPreviewCard(ctx,root);await flush();
const tick=[...timers.values()][0];const pending=tick();await Promise.resolve();await tick();
assert.equal(windows,1,'the next timer waits for the in-flight window');
finish({data:active});await pending;ctx.cleanups.forEach(fn=>fn());
''')
