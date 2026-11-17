"""Older reports can lack newer optional diagnostic fields."""
import json
from pathlib import Path
import shutil
import subprocess
import unittest

from .test_readiness_frontend import DOM

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which('node'), 'Node required for native DOM interaction tests')
class ReportDiagnosticsFrontendTests(unittest.TestCase):
    def test_partial_or_absent_saved_diagnostics_remain_readable(self):
        program = DOM + r'''
import assert from 'node:assert/strict';
Object.defineProperty(StubNode.prototype,'childNodes',{get(){return this.children;}});
const {renderDiagnostics}=await import(DIAGNOSTICS_URI);
for(const report of [
 {quality:{windows_total:2,windows_usable:1}},
 {extras:{signal_diagnostics:{total:2,accepted:1}},quality:{}},
 {extras:{signal_diagnostics:{total:2,accepted:1,note:'保存的诊断说明'}}},
]) {
 const root=renderDiagnostics(report),output=text(root);
 assert.ok(output.includes('严格合格 1/2 窗'));
 assert.ok(!output.includes('undefined'));assert.ok(!output.includes('NaN'));
 if(report.extras?.signal_diagnostics?.note)assert.ok(output.includes('保存的诊断说明'));
 else assert.ok(output.includes('旧版本未保存完整逐窗信号')||output.includes('缺失的诊断项目不补算'));
}
'''
        for token, path in {'API_URI':'web/js/api.js', 'DIAGNOSTICS_URI':'web/js/views/report-diagnostics.js'}.items():
            program=program.replace(token,json.dumps((ROOT/path).as_uri()))
        result=subprocess.run(['node','--input-type=module','-e',program],capture_output=True,
                              text=True,encoding='utf-8',timeout=15)
        self.assertEqual(result.returncode,0,result.stderr)
