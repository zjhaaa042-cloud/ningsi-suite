"""Run actual frontend import/formatting and button handlers in a DOM fixture."""
import json
from pathlib import Path
import shutil
import subprocess
import unittest

from .test_frontend_dom import NODE_STUB

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("node"), "Node is required for frontend regression tests")
class HumanModelsFrontendTests(unittest.TestCase):
    def run_node(self, body):
        program = NODE_STUB + r'''
import assert from 'node:assert/strict';
StubNode.prototype.append = function (...nodes) { this.children.push(...nodes.filter(Boolean)); };
StubNode.prototype.replaceChildren = function (...nodes) { this.children = nodes; this._text = ''; };
// Chromium exposes webkitdirectory as a reflected boolean property. Assigning
// an empty string disables it, even though a bare HTML attribute enables it.
Object.defineProperty(StubNode.prototype, 'webkitdirectory', {
  get() { return this.hasAttribute('webkitdirectory'); },
  set(value) { if (value) this.setAttribute('webkitdirectory',''); else this.removeAttribute('webkitdirectory'); },
});
document.querySelector = () => null;
globalThis.window = { location: {search: ''}, setTimeout, clearTimeout };
const ui = await import('./web/js/views/human-models.js');
const prediction = await import('./web/js/views/attention-prediction.js');
const { api } = await import('./web/js/api.js');
function text(node) { return node ? (node.textContent || '') + (node.children || []).map(text).join(' ') : ''; }
function nodes(node) { return node ? [node, ...(node.children || []).flatMap(nodes)] : []; }
''' + body + "\nconsole.log('OK');\n"
        result = subprocess.run(["node", "--input-type=module", "-e", program], cwd=ROOT,
                                text=True, capture_output=True, timeout=25)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_original_csv_bytes_preserved_and_raw_eeg_ignored(self):
        self.run_node(r'''
const content = '\ufeffa,b\r\n1,2\r\n';
const bytes = new TextEncoder().encode(content);
const files = [{name: 'features.csv', webkitRelativePath: 'export/features.csv', size: bytes.length,
  arrayBuffer: async () => bytes.buffer}, {name: 'export.json', webkitRelativePath: 'export/export.json',
  size: 2, arrayBuffer: async () => new TextEncoder().encode('{}').buffer}, {name: 'raw_samples.npy', size: 999999999}];
const read = await ui.readCaptureFiles(files);
assert.equal(read['export/features.csv'], content);
assert.equal(Object.keys(read).length, 2);
const nested = files.slice(0,2).map(f => ({...f, webkitRelativePath: 'root/' + f.webkitRelativePath}));
nested.push({name: 'dataset.json', webkitRelativePath: 'root/dataset_01/dataset.json', size: 2,
  arrayBuffer: () => { throw Error('Nested merge duplicates source exports'); }});
assert.deepEqual(Object.keys(await ui.readCaptureFiles(nested)).sort(), ['root/export/export.json','root/export/features.csv']);
await assert.rejects(ui.readCaptureFiles([]));
''')

    def test_unavailable_predictions_do_not_show_fabricated_probability(self):
        self.run_node(r'''
const unavailable = prediction.attentionPredictionCard({available:false, model_id:'abc', reason:'baseline_invalid'});
assert.match(text(unavailable), /暂不可评估/);
assert.doesNotMatch(text(unavailable), /NaN|undefined|0%/);
const available = prediction.attentionPredictionCard({available:true, model_id:'abc', mind_wandering_probability:.8,
 prediction:'internal_mind_wandering', threshold:.4, contributions:[{feature:'rel_theta', log_odds_contribution:.2}]});
assert.match(text(available), /80.0%/);
assert.match(text(available), /提高走神判断/);
''')

    def test_persistent_completed_result_downloads_and_activation(self):
        self.run_node(r'''
let activated = null;
const metric = {primary_macro_auc:.75, auc:.76, balanced_accuracy:.7, sensitivity:.8, specificity:.6,
  primary_auc_ci95:[.65,.85], participants:3,n:36,primary_participants:3};
const report = {test:metric, original_index:metric, evidence_status:'exploratory_insufficient_participants',
 coverage:{eeg_coverage_given_binary_label:.8,end_to_end_probe_coverage:.7}};
const dataset = {dataset_id:'abc',development:{participants:6,rows:72,labels:{0:36,1:36}},
  test:{participants:3,rows:36,labels:{0:18,1:18}},excluded_rows:0,excluded_reasons:{},notices:[]};
const job = {run_id:'run1',dataset_id:'abc',state:'completed',result:{evaluation:report,model_id:'m123',threshold:.4}};
api.humanModels = async () => ({data:{datasets:[dataset],jobs:[job],active:null}});
api.activateHumanModel = async body => { activated=body; return {data:{effective:'新会话生效'}}; };
const container = new StubNode('main');
await ui.renderHumanModels(container,{signal:new AbortController().signal});
const content = text(container);
assert.match(content,/探索性结果/);
assert.match(content,/训练与独立测试已完成/);
assert.doesNotMatch(content,/NaN|undefined/);
assert.equal(nodes(container).filter(n => n.tagName==='A').length,6);
const activate = nodes(container).find(n => n.tagName==='BUTTON' && text(n).includes('启用此模型'));
assert.ok(activate);
await activate.listeners.click[0]();
assert.deepEqual(activated,{run_id:'run1'});
let imported = null, trained = null;
api.importHumanData = async files => { assert.equal(folder.disabled,true); imported=files; return {data:dataset}; };
api.trainHumanModel = async id => { trained=id; return {data:job}; };
const folder = nodes(container).find(n => n.getAttribute('id')==='human-capture-folder');
assert.equal(folder.webkitdirectory,true);
assert.equal(folder.hasAttribute('multiple'),true);
const bytes = new TextEncoder().encode('{}');
folder.files = [{name:'export.json',webkitRelativePath:'capture/export.json',size:2,arrayBuffer:async()=>bytes.buffer}];
const train = nodes(container).find(n => n.tagName==='BUTTON' && text(n).includes('一键训练与独立评估'));
await train.listeners.click[0]();
assert.equal(imported['capture/export.json'],'{}');
assert.equal(trained,'abc');
assert.equal(folder.disabled,false);
assert.match(text(container),/训练与独立测试已完成/);
let localPaths = null;
const localFolder = nodes(container).find(n => n.getAttribute('id')==='human-local-folder');
localFolder.value = 'D:\\study\\capture';
api.importHumanLocalData = async paths => { assert.equal(localFolder.disabled,true); localPaths=paths; return {data:dataset}; };
api.importHumanData = async () => { throw new Error('Should use the explicit local directory'); };
await train.listeners.click[0]();
assert.deepEqual(localPaths,['D:\\study\\capture']);
assert.equal(localFolder.disabled,false);
''')
