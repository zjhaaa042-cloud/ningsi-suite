"""Personal calibration never revives a response for a previous selection."""
import json
from pathlib import Path
import shutil
import subprocess
import unittest

from .test_readiness_frontend import DOM


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("node"), "Node required for personal calibration DOM tests")
class BlinkCalibrationFrontendTests(unittest.TestCase):
    def node(self, body):
        program = DOM + r'''
import assert from 'node:assert/strict';
const {createBlinkCalibration}=await import(BLINK_URI);
const intervals=new Map();let intervalId=0;
window.setInterval=fn=>{intervals.set(++intervalId,fn);return intervalId;};
window.clearInterval=id=>intervals.delete(id);
const ctx=makeCtx();let settings={participant:'person-a',device:'lsl:headset',eeg_calibration:null};
const check=createBlinkCalibration(ctx,{getSettings:()=>({...settings}),ensureParticipant:async()=>{}});
check.update();
''' + body
        for name, path in {"API_URI": "web/js/api.js", "BLINK_URI": "web/js/views/blink-calibration.js"}.items():
            program = program.replace(name, json.dumps((ROOT / path).as_uri()))
        result = subprocess.run(["node", "--input-type=module", "-e", program],
                                capture_output=True, text=True, encoding="utf-8", timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_pending_finish_cannot_attach_previous_participant_token(self):
        self.node(r'''
let resolveFinish;
api.blinkCalibration=async body=>body.action==='finish'?new Promise(resolve=>{resolveFinish=resolve;})
  :{data:{token:'person-a-token',state:'collecting',count:15}};
await click(byText(check.node,'开始个人眨眼校准'));
const pending=click(byText(check.node,'结束并使用校准'));await flush();
settings.participant='person-b';check.update();
resolveFinish({data:{token:'person-a-token',state:'ready',count:15}});await pending;
assert.equal(check.token,null);assert.ok(!text(check.node).includes('校准已就绪'));
assert.equal(byText(check.node,'已眨眼，记录这一次').disabled,true);
assert.equal(intervals.size,0);ctx.cleanups.forEach(fn=>fn());
''')

    def test_ready_profile_is_rechecked_and_reconnect_error_invalidates_local_token(self):
        self.node(r'''
api.blinkCalibration=async body=>({data:{token:'person-a-token',state:body.action==='finish'?'ready':'collecting',count:15}});
await click(byText(check.node,'开始个人眨眼校准'));
await click(byText(check.node,'结束并使用校准'));
assert.equal(check.token,'person-a-token');assert.equal(intervals.size,1);
api.blinkCalibration=async()=>{throw Object.assign(Error('设备已重连，请重新校准'),{status:422});};
await [...intervals.values()][0]();
assert.equal(check.token,null);assert.equal(intervals.size,0);
assert.ok(text(check.node).includes('设备已重连'));
assert.equal(byText(check.node,'开始个人眨眼校准').disabled,false);
ctx.cleanups.forEach(fn=>fn());
''')

    def test_settings_roundtrip_and_disposed_request_cannot_revive_calibration(self):
        self.node(r'''
let resolveStart;
api.blinkCalibration=()=>new Promise(resolve=>{resolveStart=resolve;});
let pending=click(byText(check.node,'开始个人眨眼校准'));await flush();
settings.device='lsl:another-headset';check.update();
settings.device='lsl:headset';check.update();
resolveStart({data:{token:'old-token',state:'ready',count:15}});await pending;
assert.equal(check.token,null);assert.equal(intervals.size,0);
pending=click(byText(check.node,'开始个人眨眼校准'));await flush();
ctx.cleanups.forEach(fn=>fn());const before=text(check.node);
resolveStart({data:{token:'disposed-token',state:'ready',count:15}});await pending;
assert.equal(check.token,null);assert.equal(intervals.size,0);assert.equal(text(check.node),before);
''')
