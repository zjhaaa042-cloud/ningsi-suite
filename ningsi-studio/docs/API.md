# 凝思 Studio 接口契约（/api）

## 设备资料与开始前检查

| 方法与路径 | 请求 / 返回 |
|---|---|
| `GET /api/device-profile` | 查询参数 `device=lsl:<流名称>` 或 `sim-bsense`；返回该来源独立的 `profile`、`saved`、`missing_fields`、`updated_at` 与 `status=pending_verification`。离线也可查看；无档案不会套用其他来源或历史展示档案 |
| `POST /api/device-profile` | `{device,profile:{manufacturer,model,electrodes,reference,srate,source_unit,scale_to_uv,unit_basis,document_reference,notes?}}`；原子保存所选来源的操作员声明，不认证合规或标定。资料出处仅存文本，不读取路径 |
| `POST /api/sessions/preflight` | `{device,protocol?,participant?,eeg_calibration?}`；只读扫描服务端 LSL 元数据并核对配置，返回 `ready`、`checks[]`、`device_profile`、`calibration`、`descriptor`、`scoring.relative_indices` 与 `scoring.attention_model`。不创建会话或参与者，不建立 EEG inlet，不采集或保存脑电 |

`checks[].status` 为 `pass/warning/fail/pending`，并有 `key/label/message/action?`。`ready=true` 仅表示当前可见的真实 EEG 来源与单位配置可用于开始流程；实时信号质量、伪迹、新鲜度和个人基线仍为 `pending`，开始后才检查。模拟源、同名来源歧义、非 EEG、无设备、采样率不大于 90 Hz 或缺少单位确认时为 `false`。当前生产处理链使用 45 Hz 低通，检查只拒绝不支持的输入，不调整滤波参数。无有效 ADC 倍率或单位与设备声明冲突时返回 `checks.units=fail`。非法请求形状、来源键或协议为 422；显式传入的参与者不存在为 404。

`scoring.attention_model.status` 为 `none/compatible/incompatible/unavailable/simulation`，有模型时附 `model_id`。适配检查复用实时推断器的设备名称、来源标识、通道、电极标签、采样率、单位、时标与实现版本判定；离线原生 REC 时钟模型不能作为普通实时 LSL 模型。配置匹配仍不承诺能出分或识别有效。十分钟方案不含 SART 独立监测，额外标明模型适用范围。模型不兼容或文件损坏只发出明确警告，保留原有相对指数评估；模型结果与相对指数分别呈现，预警和反馈目标仍为原指数。

历史设备资料不会悄悄覆盖实时单位。`calibration` 只回传本次显式提交并核对后的确认值；设备未声明单位时应由操作者本次确认，或沿用已经通过严格同源匹配的启用模型配置。所有新增接口沿用现有 API 令牌、来源和请求体限制。

## 开发分析接口（历史展示模块）

下列接口保留供开发分析使用；历史 `#/showcase` 页面已退出正式产品路由。信号分析不修改源数据或既有会话，普通请求仍限制为 1 MiB。返回的实现状态、历史验证摘要和现场验收状态各自保留，不计算虚假的总体达标率。

| 方法与路径 | 请求 / 返回 |
|---|---|
| `GET /api/showcase/metadata` | 处理策略、频段、特征、输入规模和解释边界 |
| `POST /api/showcase/analyze` | `{mode:"demo",artifact:"clean"或"blink"/"muscle"/"movement"/"mixed",state:"focused"}`，或 `{mode:"raw",rate,scale_to_uv,samples:通道×时间,channel_labels?,timestamps?,baseline_samples?,baseline_timestamps?}`；返回来源、三路波形、伪迹掩码、前后 PSD、频带/时域特征、基线、逐窗质量与指数和计算耗时。输入错误 422；没有独立合格基线时指数为空 |
| `GET /api/showcase/evidence` | 固定来源的脱敏计数与校验值、逐条要求及缺口；打包快照明确来源和日期，不返回身份、原始 EEG 或任意本地文件 |
| `GET /api/showcase/device-profile` | 当前数据目录中的设备声明、缺失项和待核验状态 |
| `POST /api/showcase/device-profile` | `{manufacturer,model,electrodes,reference,srate,source_unit,scale_to_uv,unit_basis,document_reference,notes?}`；原子保存本地声明，永远不自动认定合规。资料出处是文本，不触发文件读取 |
| `POST /api/showcase/demo-session` | 空对象；明确创建双通道、250 Hz、完整协议、0.01 倍率的模拟会话并返回 `session`，201；已有同名演示运行时返回该会话，200。脑电、量表和行为全部标模拟，不改变真人启用模型 |

导入矩阵每通道至少六秒，采样率 100–1000 Hz，最多八通道和 240000 个数值；更大的记录应先截取用于展示的短段。基线与任务必须为同配置的独立输入。无源时间戳时仅按标称采样率回放，不代表设备时钟通过验证。

## 真人数据与模型

详细流程见 [真人模型训练说明](HUMAN_MODEL_TRAINING.md)。普通请求体上限1 MiB，特征导入端点上限16 MiB；其中解码后的特征文本上限12 MiB。原始脑电不上传。

| 方法与路径 | 请求 / 返回 |
|---|---|
| `GET /api/models/human` | 已导入数据集摘要、持久化任务和当前启用模型 |
| `POST /api/models/human/import` | `{files:{"相对目录/文件名":"原始UTF-8文本"}}`；返回数据集ID、人数、标签及排除计数，201 |
| `POST /api/models/human/import-local` | `{paths:["D:\\采集导出目录"]}`；仅允许本机回环地址调用，复用相同导入校验，201 |
| `POST /api/models/human/train` | `{dataset_id}`；返回异步任务，运行中202、已有完成结果200 |
| `GET /api/models/human/jobs/{run_id}` | 任务状态、阶段、进度、完成后的独立测试结果 |
| `GET /api/models/human/jobs/{run_id}/artifacts/{kind}` | `kind=model/bundle/report/markdown/metrics/predictions`；下载模型、ZIP、HTML、Markdown、JSON、CSV |
| `POST /api/models/human/activate` | `{run_id}` 或 `{model:模型JSON}` 二选一；验证规格和指纹，对新会话生效 |
| `DELETE /api/models/human/active` | 停用新会话模型，当前会话保留启动时版本 |

仅采集器特征CSV及配套清单可导入；拒绝路径穿越、重复探针、跨组参与者、校验失败和错误标签方向。相同研究目录锁定人员分组；同一数据集重复训练复用已完成结果。已有测试打开记录的参与者不能再次作为新开发轮次的独立测试组。

本机目录入口拒绝相对路径、网络共享、磁盘根目录及非本机客户端，只读取采集器规定的特征和清单文件。报告下载端点的 `report` 产物增加 `?preview=1` 可在浏览器内预览，默认仍为附件下载。

`window` / 十分钟协议 `feedback` 事件和 `GET /api/sessions/{uuid}/live` 增加 `attention_model`：`available`、`reason`、`model_id`，可用时带 `mind_wandering_probability`、`on_task_probability`、`prediction`、`threshold`、`contributions`、时间区间及适用范围提示。原有 `scores.focus` 含义不变。

模型启用会话另有 `attention_model` 和 `attention_predictions` 下载产物。`POST /api/models/train` 继续是仿真链路自检，输出不作为真人准确率证据。

- 基地址：与前端同端口（默认 `http://127.0.0.1:8765`）
- 全部接口在 `/api` 前缀下；`GET /api/openapi.json` 返回机器可读描述
- 请求/响应均为 `application/json; charset=utf-8`
- 请求体最多 1 MiB，读取总期限为 10 秒；API 令牌在读取请求体之前校验。不支持 `Transfer-Encoding`，重复或非法 `Content-Length` 会被拒绝并关闭连接。
- 时间倍率 `time_scale`：`1.0` 为真实节奏（每窗 2 秒），`0.05` 为 20 倍速演示；算法口径与窗口长度不变
- 运行中会话的实时数据通过 **SSE** 推送，`GET /api/sessions/{uuid}/live` 为轮询兜底

## 错误体

```json
{ "error": { "code": "conflict", "message": "当前会话不处于 SART 阶段或已结束", "detail": {} } }
```

| HTTP | code | 场景 |
|---|---|---|
| 400 | bad_request | 请求体不是合法 JSON，长度头非法或请求体不完整 |
| 401 | unauthorized | 配置了 `--token` 但 `/api` 请求未带（或带错）令牌；静态资源不受影响 |
| 403 | forbidden | 浏览器来源与服务地址不同，或 Host 不是当前服务监听的地址 |
| 404 | not_found | 被试/会话/产物不存在 |
| 405 | internal_error→405 | 方法不被支持（响应带 `detail.allow`） |
| 408 | bad_request | 请求体读取超过 10 秒 |
| 409 | conflict | 状态冲突：会话不在运行、任务未到该阶段、编号重复 |
| 413 | bad_request | 请求体超过 1 MiB |
| 422 | validation_failed | 字段缺失或取值越界 |
| 429 | too_many_requests | 并发运行会话超上限（默认 2） |
| 500 | internal_error | 未预期异常（已记录日志，服务不退出） |

---

## 基础

### `GET /api/health`
```json
{
  "status": "ok", "service": "ningsi-studio", "version": "0.1.0", "product": "0.1.0",
  "engine": { "product": "0.1.0", "spectrum": "welch-v1", "indicator": "indicator-v1",
              "baseline": "baseline-v2", "assessment": "joint-assessment-v3" },
  "engine_path": ".../ningsi/src/ningsi/__init__.py",
  "db": ".../var/studio/studio.sqlite3", "data_dir": ".../var/studio",
  "runs_root": ".../var/studio/runs",
  "active_sessions": ["<uuid>"], "max_active_sessions": 2,
  "overview": { "subjects": 1, "sessions": 3, "sessions_done": 2, "alerts": 5, "latest": [] }
}
```
`runs_root` 是会话运行目录的根：每个会话写在 `<runs_root>/<uuid>/`，其 JSONL 台账在
`<runs_root>/<uuid>/history/sessions.jsonl`（全局趋势的 `ledger_points` 就是从这里汇总的）。

### `GET /api/config`
返回 `version`、`specs`、`window_sec`(4.0)、`step_sec`(2.0)、`welch`、`bands`、`total_band`、
`indicators`、`score_gain`、`baseline_protocol`、`quality`、`alerts`、`assess`、`training`、
`heatmap_bands`、`scale_boundary`、`phases`、`privacy`（隐私与结论边界原文），以及：

- **`phases[]`**：11 个阶段定义。每个阶段除 `headline / details / duration_sec / advance / next_hint /
  auto_note` 外，还有 **`why`**——「为什么需要这一步 / 不做会丢什么」，会话流程页直接显示它
  （用户问过"会话流程中的操作都是必要的吗"，页面必须能自己回答）。
- **`behavior`**：行为任务与量表的协议参数，避免界面自己写死数字：
  `sart{practice_trials, trials, nogo_trials, nogo_digit, window{practice, main}}`、
  `pvt{duration_sec, isi_sec, lapse_sec}`、`scales{codes, items_per_scale}`。
  向导第 3 步据此算出"整场约 N 分钟、需要动手的是哪几步、各多少次"。

### `GET /api/devices?probe=1.0`
```json
{ "sources": [ { "key": "lsl:BioMultiLite EEG-00cde1", "kind": "lsl", "srate": 250.0,
                 "channels": 2, "device": "BioMultiLite EEG-00cde1", "real": true,
                 "simulated": false, "note": "真实 LSL 流：…（信号已按 acq-condition-v1 调理）",
                 "channel_labels": ["Fp1", "Fp2"], "source_id": "BioMulti_Lite_EEG-00cde1" },
               { "key": "sim-bsense", "kind": "sim", "srate": 250.0, "channels": 1,
                 "device": "sim-bsense", "real": false, "explicit_only": true,
                 "note": "仿真脑电源（仅用于演示与自动测试，必须在界面与报告中标注）" } ],
  "source": "lsl:BioMultiLite EEG-00cde1", "source_kind": "lsl", "source_note": "…",
  "has_real_source": true }
```

**顺序与缺省规则（2026-10-07 起）**：

- **真实 LSL 流排在前面**；仿真源 `sim-bsense` 永远排最后并带 `explicit_only: true`，
  只能被**显式**选择，不再作为缺省（原来它排第一，前端就把仿真当默认了）。
- `has_real_source` 直接回答"现在到底有没有脑机信号"；没有实时源时列表里只有仿真源，
  且它带 `hardware_note`（接入指引）。
- `simulated: true` 表示这条 `lsl:` 流是**本仓库内置的仿真 outlet**（`simulate-outlet`）：
  传输是真的、数据是算出来的，note 会写明"非真实设备"。

### `GET /api/devices/preview` · `POST /api/devices/preview` · `DELETE /api/devices/preview`
**无会话的设备实时预览**：不创建会话也能看当前脑电信号（读取检测到的实时流，
做与报告一致的 `acq-condition-v1` 真机调理，并给出这一窗的质检判定）。

```json
GET  /api/devices/preview            → { "active": false, "source": null, "no_signal": true,
                                         "note": "未开始预览：当前没有脑机信号" }
POST /api/devices/preview            → body { "source": "lsl:<流名>" 或 "sim-bsense"（可省略） }
     { "active": true, "source": "lsl:…", "kind": "lsl", "device": "…", "srate": 250.0,
       "channels": 2, "channel_labels": ["Fp1","Fp2"], "live": true,
       "seconds_since_last": 0.02, "buffered_samples": 4820,
       "conditioning": { "spec": "acq-condition-v1", … }, "no_signal": false }
GET  /api/devices/preview?window=1   → 上面字段 + { "seconds": 4.0, "samples": [[…], […]],
                                         "peak_uv": 42.1, "std_uv": 8.3,
                                         "quality": { "ok": true, "reasons": [], "metrics": {…} } }
DELETE /api/devices/preview          → { "active": false, "stopped": true }
```

- `source` 省略时用**当前检测到的实时源**；一个都没有 ⇒ `409 conflict`
  （「未检测到脑电信号…」），**不会**回落到仿真源。
- 预览**只读**：不建会话、不落库、不出指标；`quality` 用的是与报告同一份 `config.QUALITY` 门槛。
- `live=false` / `no_signal=true` 表示流还在但最近没有样本（设备停了/线掉了），界面据此显示"无信号"。

### `GET /api/devices/status`
正在运行的会话所用设备的**活体健康**：是否在收数、实测采样率、缓冲量、错误。
```json
{ "active_sessions": ["<uuid>"],
  "devices": [ { "session": "<uuid>", "key": "lsl:ningsi-sim-eeg", "kind": "lsl",
                 "device": "ningsi-sim-eeg", "srate": 250.0, "channels": 1,
                 "live": true, "seconds_since_last": 0.02, "observed_srate": 249.98,
                 "buffered_samples": 4820, "total_samples": 15230, "stream_errors": {},
                 "descriptor": { "kind": "eeg", "name": "ningsi-sim-eeg", "channel_labels": ["Fp1"] },
                 "note": "真实 LSL 流：ningsi-sim-eeg（1 通道，250 Hz，标签 ['Fp1']）" } ],
  "sources": [ { "key": "sim-bsense", "kind": "sim" } ] }
```
上例是 **LSL 源**的形状：`live`、`seconds_since_last`、`observed_srate`、`buffered_samples`、`total_samples`、
`stream_errors`、`descriptor` 这些**活体字段只有 LSL 源才有**（引擎能自查流状态）；仿真源
（`kind: "sim"`）只给 `session`/`key`/`kind`/`device`/`srate`/`channels`/`note`/`live`（`live` 恒为 `true`，
其余字段直接缺失，前端按「—」显示，不是 0）。
`live=false` 且 `seconds_since_last` 持续变大，说明设备已停/线掉了：此时会话会把窗判为不可用
（界面上显示为缺失色块），不会拿旧数据继续算指标。

### `GET /api/overview`
首页统计 + 最近会话 `latest_detail` + 最近被试 `subjects_recent` + `active_sessions` +
`source` / `source_kind` / `source_note`（**当前检测到的实时源**的键、类别与说明文案，供顶栏展示）
+ `has_real_source`。

**检测不到实时源时 `source`/`source_kind`/`source_note` 都是 `null`**，`has_real_source` 为 `false`
——顶栏据此如实写成「数据来源：无信号」。仿真源 `sim-bsense` **不参与**这个回退：它必须由调用方
显式指定（见 `POST /api/sessions` 的取值规则），没有脑机信号时不会自动使用仿真。

> `GET /api/health` 里的 `overview` 也带同样四个字段（LSL 扫描结果有 3 秒缓存，health 不会被拖慢）。

### `GET /api/sessions/{uuid}`

会话详情。除 `session`（含 `protocol` / `protocol_label`）、`runs[]`、`alerts`、`artifacts`、
`indicator_summary`、`runtime`（`alive` / `awaiting_input` / `source_kind`）之外，
**`phases[]` 按本次会话的协议档给出**：短协议返回 9 步（不含 `training` / `model`）。
前端据此渲染"第 x / y 步"，不能拿 `/api/config` 的完整 11 步去顶（否则短协议会话会显示错步数）。

### `GET /api/sessions` 与列表类响应的分页约定
`{"items": [...], "total": N, "limit": L, "page": P}`；`page` 从 1 开始，`limit` 上限 200。
会话、被试、被试的会话列表、`/api/scales`、`/api/sessions/{uuid}/artifacts` 全部遵守该信封，
并都接受 `?page=&limit=`（默认 `limit=50&page=1`）。

---

## 被试

### `GET /api/subjects?query=&page=1&limit=50`
```json
{ "items": [ { "public_id": "p01", "label": "演示被试", "age_band": null, "sex": null,
               "handedness": null, "consent_version": "consent-v1", "consent_at": "2026-01-01T00:00:00+00:00",
               "note": null, "created_at": "...", "updated_at": "...",
               "sessions": { "done": 1, "total": 1 } } ],
  "total": 1, "limit": 50, "page": 1 }
```

### `POST /api/subjects`
请求（`public_id` 省略且 `auto_id != false` 时自动生成 `p01`、`p02`…）：
```json
{ "auto_id": true, "label": "演示被试", "age_band": "18-25", "sex": "女",
  "handedness": "右", "consent_version": "consent-v1", "note": "仅用于演示" }
```
响应 `201` 同单个被试对象；编号重复返回 `409`。

### `PATCH /api/subjects/{public_id}`
可改 `label / age_band / sex / handedness / note / consent_version / consent_at`，返回更新后的对象。

### `GET /api/subjects/{public_id}/sessions?page=&limit=`
返回该被试的会话列表（结构同 `GET /api/sessions`）。

---

## 会话

### `POST /api/sessions`
请求：
```json
{ "participant": "p01", "device": "lsl:BioMultiLite EEG-00cde1", "time_scale": 1,
  "training_mode": "quick", "srate": 250.0, "channels": 1, "label": "第一次训练",
  "create_subject": true }
```
`time_scale` 取值 `[0.01, 1.0]`；**`time_scale < 0.2` 视为快速演示模式**，量表、SART、PVT 由服务端
生成确定性作答（无需前端交互），用于评审演示与自动化测试。

LSL 会话要求 `time_scale=1`。可省略 `eeg_calibration` 以使用流声明的电压单位。设备元数据缺失时，经操作者核对厂商配置，可发送 `{"eeg_calibration":{"source_unit":"mV"}}`。支持 nV、uV、mV、V。若设备发送 ADC 计数，需要用实测或厂商提供的倍率，如 `{"eeg_calibration":{"source_unit":"ADC","scale_to_uv":0.25}}`。示例倍率必须换成实际数值。接口将本人选择、来源和转换倍率保存在会话与报告中，只作用于 EEG 通道。与已声明单位冲突会返回 409，旧会话没有确认记录时返回 null。

**`device` 取值规则（2026-10-07 起收紧：不再有"缺省仿真"）**：

| `device` | 行为 |
|---|---|
| 省略 / `""` / `auto` | 用**当前检测到的实时源**（真机优先、脑电流优先于同机的 Metric/HeartRate 等，其次内置仿真 outlet）；**一个都没有 ⇒ `409`**，消息里给接入指引 |
| `lsl:<流名称>` | 先确认该流现在可见，否则 `409`（fail fast，不再"建好会话才发现没信号"）；连接后 `srate`/`channels` 按流描述符回写 |
| `sim-bsense` | **允许，但必须显式**：这是演示/自测入口，`source`/SSE 事件/报告都会标注为仿真 |
| 其它任意字符串 | `422 validation_failed`（不再被当成仿真源） |

**流类型守卫（2026-10-07 新增）**：一台 BioMultiLite 会同时推 6~7 条流
（`eeg` / `metric` / `fnirs` / `heart_rate` / `general_metric` / `motion` / `marker`），
**只有 `eeg` 那条能用于本产品**——质检、频谱、专注/放松/负荷指标与 SART/PVT 判定全部基于脑电通道。
选错流的后果不是报错而是"算出一堆看着很像的数字"，所以建会话时：

- 流类型属于 `fnirs` / `metric` / `motion` / `heart_rate` / `general_metric` / `marker`
  ⇒ **`409`**，消息里写明它是什么流、为什么不能用，并在 `detail.eeg_candidates` 里给出可用的 EEG 流键；
- 类型为空/未知的流**不拦**（可能只是采集端没写 `type` 字段）；
- 确实要用非脑电流采集时显式传 **`allow_non_eeg: true`**（放行后仍会因流打不开而如实失败）。

`GET /api/devices` 的每条 source 因此多了三个字段，界面据此把 EEG 单独标为推荐、其余折叠：
`stream_kind_label`（中文类型名，如 `近红外（FNIRS）`）、`is_eeg`（是否脑电）、`usable`（本产品能否用它跑检测）。

不再有"真实设备打不开就降级为仿真源"的静默替换：真实源不可用时会话启动失败并写明
`无信号：数据源 <device> 不可用（…）`。

**`protocol` 取值规则（2026-10-07 新增：短协议）**：

| `protocol` | 行为 |
|---|---|
| 省略 / `""` / `full` | **完整协议（默认）**：11 个阶段；SART 12 练习 + 180 正式（20 No-Go）、PVT 180 秒 |
| `short` | **短协议**：9 个阶段（**不含 `training` 与 `model`**）；SART 6 + 90（10 No-Go）、PVT 120 秒 |
| 其它任意字符串 | `422 validation_failed`（**绝不静默当成短协议**） |

短协议的代价必须留痕：会话详情与建会话响应都带 `protocol` / `protocol_label`，
`GET /api/sessions/{uuid}` 的 `runs[]` 里不会出现 `training` / `model`，
报告 `report_markdown` 表头会写 `- 检测协议：短协议（本次未跑：神经反馈训练、模型训练）`
与一行 `- ⚠ 协议差异提醒：…`，`report.extras` 里带 `protocol` / `protocol_label` /
`protocol_note` / `skipped_phases`。**两个协议的结果不应横向比较**（No-Go 数量不同）。

`GET /api/config` 的 `profiles[]` 给出两个档的 `phase_count` / `total_sec`（按实际行为任务秒数算，
不是阶段名义值）/ `saved_sec` / `dropped_phases` / `caveat`，界面据此显示负担，不写死数字。

响应 `201`：
```json
{ "session": { "uuid": "<32位>", "participant": "p01", "subject_id": 1, "label": "第一次训练",
                "status": "running", "phase": "qc",
                "phase_label": "设备质检", "progress": 0.0,
                "device": "lsl:BioMultiLite EEG-00cde1",
                "time_scale": 0.05, "training_mode": "quick",
                "engine_versions": { "spectrum": "welch-v1", "indicator": "indicator-v1",
                                     "baseline": "baseline-v2", "assessment": "joint-assessment-v3" },
                "source": "lsl:BioMultiLite EEG-00cde1", "srate": 250.0, "channels": 2,
                "started_at": "...", "ended_at": null, "created_at": "...", "error": null },
  "events_url": "/api/sessions/<uuid>/events",
  "phases": [ { "key": "qc", "label": "设备质检", "interactive": false, "weight": 0.04 } ],
  "runtime": { "alive": true, "source": "lsl:BioMultiLite EEG-00cde1" } }
```
`source` 是**实际数据源**（`lsl:<name>` 或显式的 `sim-bsense`），与 `runtime.source` /
`GET .../{uuid}` 的 `runtime.source_kind` 同源，不会出现"标着 lsl、实际跑 sim"的自相矛盾。

连上真实流时，`srate` 与 `channels` 用**流描述符里的真实值回写**（请求里不传时默认 250.0 / 1）：
`device="lsl:BioMulti Lite EEG-00cde1"` 的响应会是 `"channels": 2`，与 `runtime.source.channels`
以及报告 `extras.channels` 同源；流名按 `lsl:` 后的字符串**精确匹配**，同名多条流时只连这一条。
并发超限返回 `429`。

连上真实流时，`srate` 与 `channels` 用**流描述符里的真实值回写**（请求里不传时默认 250.0 / 1）：
`device="lsl:BioMulti Lite EEG-00cde1"` 的响应会是 `"channels": 2`，与 `runtime.source.channels`
以及报告 `extras.channels` 同源；流名按 `lsl:` 后的字符串**精确匹配**，同名多条流时只连这一条。
并发超限返回 `429`。

### `GET /api/sessions?participant=p01&status=done&page=1&limit=50`
`items[]` 为会话对象，额外含 `alert_count` 与 `phase_label`（已知阶段的**中文名**，与详情接口同一张
`core/phases.py` 映射表；终态 `done`/`failed`/`cancelled` 不是流程阶段，此字段为 `null`，由界面本地化）。
`phase` 运行中是流程阶段键（`qc`/`baseline`/`scale`/…），会话收尾后写的**就是终态本身**
（`done`/`failed`/`cancelled`，与 `status` 一致），所以取消掉的会话是「已取消」而不是「失败」。

### `GET /api/sessions/{uuid}`
会话对象 + `alerts`、`runs`（每阶段一行：`phase/status/duration_ms/error/payload`；`payload` 已解析为
对象，不是 JSON 字符串）、`artifacts`、`indicator_summary`、
`runtime{ alive, awaiting_input, source, source_kind }`。
`runtime.source_kind` 优先取运行期实际种类；**运行器已被回收的历史会话**按落库的 `source` 键反推
（`lsl:` ⇒ `lsl`，仿真键 ⇒ `sim`），避免会话结束后界面上"仿真"标注消失（`_source_kind_from_key`）。
会话仍在运行时，`source` 取运行时**实际**数据源，与 `runtime.source` / `runtime.source_kind`
一致（LSL 未就绪而内部降级为仿真时，这里会显示 `sim`，不会仍标 `lsl`）。

### `DELETE /api/sessions/{uuid}`
取消运行中的会话，返回 `{"cancelled": true, "uuid": "..."}`；不在运行中返回 `409`。

### `GET /api/sessions/{uuid}/events`（SSE）
`Content-Type: text/event-stream`。每条事件形如：

```
id: 12
event: window
data: {"id":12,"type":"window","session":"<uuid>","data":{"index":7,"t_end":14.0,"scores":{"focus":0.62}}, "phase":"monitor"}
```

> 事件负载约定：**业务字段一律包在 `data` 里**；`id` / `type` / `session` / `phase` 等为事件元数据。
> 下方的"data 关键字段"即指 `event.data` 的属性。

| event | data 关键字段 |
|---|---|
| `started` | `uuid, participant, device, source_kind, source_note, time_scale, auto, phases[], engine_versions` |
| `notice` | `level, message`；PVT 抢答结算另带 `task: "pvt", index, false_start: true`，前端据此取消同一试次的刺激和漏答计时 |
| `phase` | `key, label, state(running/done), progress` |
| `progress` | `key, progress, windows_done?, windows_total?, trials_done?, trials_total?, segments_done?, segments_total?` |
| `quality` | `windows, usable, valid_ratio, reasons{}, passed` |
| `baseline` | `key(baseline_open/baseline_closed), label, baseline{...}` |
| `window` | `index, t_end, state, usable, quality, scores{focus,relax,load}, index_z, band_z, reasons[], alerts[], heatmap_cell, signal{srate,channels,samples[],relative,time_domain}` |
| `scale_request` | `code, label, size, instruction` |
| `scale_scored` | `code, raw_score, standard_score, level` |
| `behavior_request` | `task, digits[], nogo_trials, practice[], sequence_set_id, seed, instruction`（PVT 为 `trials, onsets[], duration_sec, lapse_sec, instruction`） |
| `trial` | SART：`phase(practice/main/done), index, digit, total, response_window`；PVT：`index, onset, total, response_window` |
| `behavior` | `task, result{...}` |
| `monitor` | `summary, quality` |
| `feedback` | `segment, target, score, on_target, usable, t, heatmap_cell?`；实际训练窗附同口径时间线单元，休息提示不生成假窗口 |
| `segment` | `seq, index, target, target_after, hold_sec, hold_after, duration_sec, excluded_windows, stats{mean,on_target_ratio,volatility,n,achieved}, mean_score, on_target_ratio, volatility, n, achieved, samples[[t,score]]` |
| `training_start` | `mode, segments, segment_sec, target, rationale, initial_from, target_step, hold_sec` |
| `assessment` | 同 `GET .../assessment` |
| `scales` | `results[]`（SAS/SDS 计分结果汇总，量表阶段结束时下发一次） |
| `model` | `spec, features[], subject_split, train, validation, test, model_path` |
| `model_prediction` | SART期间的真人模型结果：`available, reason, model_id, mind_wandering_probability, prediction, contributions` |
| `artifacts` | `kind -> path` |
| `cancelled` | `message` |
| `error` | `message, detail` |
| `finished` | `status(done/failed/cancelled), error, summary` |

> **`trial.response_window`（秒，真实时间）**：本试次的作答窗口，是试次能否推进的关键——
> 试次是"服务端发一个、前端答一个"驱动的，而 **SART 的 No-Go 试次（显示 3）正确做法就是不按键**；
> 前端必须在窗口到点时自动补交一笔 `{responded:false, rt:null}`（Go 记漏报、No-Go 记正确抑制），
> 该提交表示页面仍在正常执行试次；不能仅依赖服务端超时来推进。
> 该值已按 `time_scale` 折算（演示模式整体节奏变快），且不小于 0.35 秒。
> 服务端自身只等 `response_window + 0.6s`，并校验提交里的 `index`（SART 另校验 `phase`）。
> 错位输入被拒绝或丢弃，当前试次继续等待原截止时间，不会被旧作答提前推进。
> 单次超时按未作答继续；连续 5 个试次都未收到匹配提交才中止并提示检查页面与连接。
> PVT 抢答结算后会发送带试次序号的 `notice`；前端必须清除该试次的计时器，避免重复提交。
>
> 注：本说明必须放在事件表**之外**——表格中间夹引用块或空行会把 Markdown 表格截断，
> `ningsi_studio check` 的"文档事件"统计会掉（本文件被踩过一次：23 → 11）。

重连：带上 `Last-Event-ID: <id>` 头（或 `?last_event_id=`）即可补发漏掉的事件；
不带 id 订阅时只补发"快照类"事件（阶段快照 `phase`，会话已结束时还有 `finished`），
随后若总线已关闭会立刻收到 `closed`，客户端据此结束连接。

### `GET /api/sessions/{uuid}/signal?hz=10&seconds=10&points=1200`
同一会话支持 Studio 与管理者平台同时订阅；每个连接独立保留 `hz`、`seconds`、`points` 显示参数。关闭一个连接不会中止会话或其他连接，频谱计算缓存仍在会话内共享。

**高频原始信号 SSE**（`text/event-stream`）：给"实时监测"页的波形与频谱供数，
与 4 秒分析窗**解耦**——按 `hz`（默认 10 帧/秒，上限 25）直接读采集缓冲的最新样本。

- `seconds`：每帧覆盖的时间跨度（2–30 秒，默认 10）；**真实 LSL 源会被压到 6 秒以内**
  （等于 `ManagedLslSource.condition_seconds`）：真机调理链是纯 Python 双二阶，成本与窗长线性相关
  （2 通道 10 秒窗约 30 ms/帧），10 FPS 的显示通道按长窗调理会把推帧线程吃满。
  这是"显示窗口"的上限，帧里回报的 `window_sec` 会同步变小（前端按它定标），**不改变** 4 秒分析窗口径。
- `points`：每通道最多返回多少个点（200–4000，默认 1200）；抽稀用 **min/max 保峰值**，
  尖峰/伪迹不会被均匀抽样抹掉
- 会话不在运行中时返回 `409`，走**统一错误体**（`{"error":{"code":"conflict","message":...}}`，
  不是自造的 `session_not_running`）；会话结束（`finished`/`cancelled`）后服务端会停止推帧并收流，
  客户端应以 `EventSource.onerror` + `GET .../live` 的 `runtime_alive` 作为降级依据，不必依赖连接 EOF

事件类型固定为 `signal`：
```json
{ "type": "signal", "session": "<uuid>", "seq": 42, "t": 1765000000.12,
  "window_sec": 10.0, "srate": 250.0, "channel_count": 2, "samples": 2500,
  "realtime": true, "refresh_hz": 10.0, "source_kind": "lsl", "device": "ningsi-sim-eeg",
  "channels": [ { "index": 0, "label": "Fp1", "unit": "uV", "peak": 78.5,
                  "points": 1200, "raw_points": 2500, "samples": [3.2, -4.1, "..."] } ],
  "spectrum": { "freqs": [0.0, 0.49, "..."], "power_db": [-18.5, 21.9, "..."],
                "segments": 3, "delta_f_hz": 0.4883 },
  "band_peaks": { "theta": 12.4, "alpha": 18.1 },
  "bands": { "theta": 0.011, "alpha": 0.415 } }
```
说明：`samples` 只用于**显示**，不参与指标与计时（报告口径仍以 4 秒窗的 `window` 事件为准）；
`spectrum` 用最近一个窗的 welch-v1 结果，1 秒缓存一次，避免与报告出现两套算法。

### `GET /api/sessions/{uuid}/live`
轮询兜底：`status, phase, progress, indicator_summary, alerts[], awaiting_input[], runtime_alive`。

另含 `checkins_open`、`checkins` 和 `affect_model`。监测与训练窗在适配电极、频率、电压单位且质量合格时可记录压力与情绪效价的连续研究分数。会话页面在受试者作答期间隐藏模型方向和分数。低质量、模拟或不匹配输入返回缺失和原因。

### `POST /api/sessions/{uuid}/state-checkins`

在运行中会话的量表、监测或训练阶段提交当下自评。请求体含客户端生成的 32 位十六进制 `request_id`，`stress`（0–10 压力）与 `valence`（0–10 愉悦程度）至少填写一个整数，另一项可为 null。相同请求可以安全重试。同一编号改填不同答案或在已关闭阶段新增作答将返回 409。压力自评达到 7 分即记录 `self_reported_high_stress` 提醒，阈值属于工程设置。

返回 `record` 和 `high_stress`，记录含阶段、真实接收时间、会话内时刻和来源。相同阶段最近四秒内有质量合格、已确认单位的 EEG 分析窗时，`eeg_context` 附该窗频带特征与模型输出，缺失保留 null。会话 JSON 报告 `extras.state_observations` 保留全部自评和汇总，Markdown 与离线 HTML 展示自评、提醒及监测窗模型覆盖。自发作答时间可能有选择偏倚。

### `GET /api/sessions/{uuid}/heatmap`
```json
{ "uuid": "...", "spec": "heatmap-availability-v1", "columns": 30,
  "cells": [ { "index": 2, "label": "灰绿", "color": "#6E9E8A", "score": 0.51,
               "low": 0.4, "high": 0.6, "t": 12.0, "window_id": "training:6",
               "status": "scored", "signal_usable": true, "phase": "training",
               "reason_codes": [], "reason_labels": [], "indicator_reason": "", "elapsed_sec": 340.2 },
             { "index": -1, "label": "信号可用 · 基线待确认", "color": "#79aaa2", "score": null,
               "low": null, "high": null, "t": 14.0, "window_id": "baseline_open:7",
               "status": "baseline_pending", "signal_usable": true, "phase": "baseline_open",
               "reason_codes": ["baseline_invalid"], "reason_labels": ["个人基线尚未合格"],
               "indicator_reason": "baseline_invalid", "elapsed_sec": 342.2 } ],
  "legend": [ { "label": "深蓝", "color": "#1B3B6F", "range": "0.00–0.20" } ],
  "availability_legend": [ { "status": "baseline_pending", "label": "信号可用 · 基线待确认", "color": "#79aaa2" } ],
  "missing_color": "#6b6b6b",
  "scored": 1, "missing": 1,
  "summary": { "total": 2, "signal_usable": 2, "scored": 1, "missing": 1,
               "signal_coverage": 1.0, "score_coverage": 0.5, "unknown_quality": 0, "coverage_unit": "windows" },
  "evidence_source": "heatmap_metrics",
  "note": "状态颜色仅用于已有合格分数；采集可用性分开呈现。重叠窗不是独立样本或连续有效时长。" }
```
示例省略了完整图例和统计。`index = -1` 表示**未评分**，其 `score/low/high` 为 `null`，不得补零、插值或解释为低专注。`status` 分别为：`scored`（已有合格分数）、`baseline_pending`（信号明确合格但基线尚未可用）、`signal_limited`（信号未通过质量检查）、`no_data`（缺样本或不足完整窗口）、`timing_issue`（时标问题）及 `unscored`（尚无指数或历史依据不足）。各非评分状态的颜色取 `availability_legend`；原五档颜色和旧字段保留兼容。

`signal_usable` 为 `true/false/null`，`null` 表示旧记录无法判断，不能当作合格或不合格。`summary` 另含 `counts{status:窗数}`、`reason_counts[{code,label,count}]`（同窗原因可重叠）和 `phase_counts[{phase,total,signal_usable,scored}]`。两种 coverage 均为对应窗数除以全部已记录窗数，无窗口时为 `null`；不以重叠窗数乘窗口长度报告有效秒数。

新会话将质检、基线、监测和实际训练窗写入独立 `metrics.kind="heatmap"` 序列，不使训练窗进入评估指标或趋势汇总。`window` 与实际训练 `feedback` 的 `heatmap_cell` 和 REST 使用同一结构及稳定 `window_id`，前端据此去重。`t` 保留原窗口步进时标，`elapsed_sec` 为有记录时的真实会话经过时间；未采集的行为任务间隔不补窗。监测阶段结束不会覆盖此前窗口。下载的 `heatmap_svg` 使用同源的状态／信号可用性双轨。

旧会话优先从已有运行器或报告 `signal_diagnostics.windows` 只读恢复阶段、质量和原分数，不修改原报告或重新评分；没有逐窗诊断时仅采用已存指标和 `valid_ratio`，不猜缺失原因。`evidence_source` 对应 `heatmap_metrics/runtime_diagnostics/report_diagnostics/legacy_metrics`。

### `GET /api/sessions/{uuid}/trend?field=focus&period=week`
`field ∈ {focus, relax, load, stress}`，`period ∈ {week, month}`；
`points[{period, mean, n, std}]`，另返回 `sessions / comparable / rejected / rejected_reasons`
（设备、采样率或通道数变化的历史记录会被排除在聚合之外）。

`field=stress` 使用已完成现场 LSL 会话的本人压力自评（0–10 分），先求各会话均值，再按周或月对会话等权平均。模拟记录与未作答会话不参与，`n` 为会话数，`ledger_points` 为空。同一统计适用于全局趋势端点，可用 `participant` 选择个体。

---

## 量表

- `GET /api/scales?page=1&limit=50` → `{ "items": [{ "code": "SAS", "name": "焦虑自评量表", "size": 20,
  "version": "zung-cn-v1", "estimate_minutes": 5 }], "total": 2, "limit": 50, "page": 1 }`
  （目录目前固定 SAS/SDS 两套；信封与分页参数见「列表类响应的分页约定」）
- `GET /api/scales/{code}`（`code` 取 `SAS` 或 `SDS`）→ `code, name, version, factor, size, options[{value,text}]`,
  `items[{index,text,reverse}]`, `reverse_items[]`, `standard_factor`, `boundaries`, `note`
- `POST /api/sessions/{uuid}/scales/{code}`
  ```json
  { "responses": [2,2,3,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2] }
  ```
  响应：`{ "code": "SAS", "delivered_to_runtime": true, "raw_score": 45,
            "standard_score": 56, "level": "轻度", "answered": 20, "missing": [] }`
  作答值必须是 1–4 的整数，长度 20；否则 `422`。仅当会话正等待该量表时
  `delivered_to_runtime` 为 `true`（也可在阶段开始前预提交）。

---

## 行为任务

### SART
- `GET /api/sessions/{uuid}/behaviors/sart/sequence`
  `{ "task": "sart", "trials": 180, "nogo_trials": 20, "practice_trials": 12,
     "sequence_set_id": 5, "seed": 4157852885, "digits": [1..9...], "practice": [7,2,...],
     "instruction": "看到 1–9 按空格；看到数字 3 不要按。…" }`
- `POST /api/sessions/{uuid}/behaviors/sart/trial`
  ```json
  { "phase": "main", "index": 0, "responded": true, "rt": 0.37 }
  ```
  响应 `202`-style：`{ "accepted": true }`；服务端按 `index` 校验顺序，顺序不符返回 `409`。
  未运行时（快速模式）该接口返回 `409`。

### PVT-B
- `GET /api/sessions/{uuid}/behaviors/pvt/sequence` → `{ "task": "pvt-b", "duration_sec": 180.0,
  "trials": 69, "onsets": [2.0, 4.13, ...], "lapse_sec": 0.5, "instruction": "..." }`
- `POST /api/sessions/{uuid}/behaviors/pvt/trial` → `{ "index": 0, "responded": true, "rt": 0.31,
  "false_start": false }`

### 结果
`GET /api/sessions/{uuid}/behaviors/{task}/result`（`task` 取 `sart` 或 `pvt`）
落库后返回 `{ "result": {...}, "trials": {...} }`；未完成时 `404`/`409`。
SART 结果字段：`trials, go_trials, nogo_trials, go_accuracy, commission_rate, omission_rate,
rt_mean, rt_sd, rt_p50, rt_variability, sequence_set_id, seed, spec`；
PVT 结果字段：`trials, responded, missed, rt_mean, rt_median, rt_sd, rt_p90, lapses,
lapse_rate, false_starts, valid`。

---

## 评估、训练、报告

- `GET /api/sessions/{uuid}/assessment` → 联合评估对象：
  `spec, conclusion, dimension_states{attention,stress}, consistency{eeg_vs_scale,eeg_vs_behavior},
  evidence[{code,source,dimension,summary,value,direction,available,ref}], advice[], boundary`。
  `consistency` 的取值：`一致`（两侧都有可用证据且同向提示偏离）、`不一致`、
  `无冲突`（两侧都有证据但无冲突），以及三种缺证据写法——`无法比较（脑电不可用）`、
  `无法比较（量表不可用）`、`无法比较（量表与脑电均无可用证据）`（行为任务同理）。
  **任一侧缺证据时不再报"一致"**（脑电 0 可用窗却写"脑电与量表一致"属于凭空断言一致性）。
  评估未完成时 `409`。
- `GET /api/sessions/{uuid}/training` → `{ "segments": [...], "summary": {...} }`（`segments[]` 的字段见
  上方 `segment` 事件）；`summary` 含 `mean_focus, on_target_ratio, first_to_last_change, baseline_before,
  baseline_after, target_rationale, baseline_comparable`，另含 `describe`（训练方案：
  `mode, segments, segment_sec, target, rationale, initial_from, target_step, hold_sec`，
  与 `training_start` 事件同一份）。
  `neurofeedback-v3` 的均值和波动对 `n > 0` 的轮次等权平均；全无有效数据时为 `null`。
  `on_target_ratio` 改为达标秒数/全部训练段时长（含缺失轮次）；
  `valid_time_ratio` 为有效秒数/全部训练段时长，`on_target_valid_ratio` 为达标秒数/有效秒数。
  无有效数据时两种达标率为 `null`，覆盖率为0。旧版占比按有效样本统计，页面标识旧版，不能直接比较。
  `first_to_last_change` 仅在至少两轮且首末轮均有有效数据时计算，否则为 `null`。
  空 `summary: {}` 不代表训练结束；实时反馈缺测时显示缺测，训练结束后才展示保存结果。
- `GET /api/sessions/{uuid}/report` → 报告 JSON（`spec/participant/indicators/quality/scales/
  behavior/assessment/extras`），并附 `report_markdown`（报告原文，Markdown 文本）与
  `report_markdown_path`；尚未生成时返回 `202` 且带 `partial: true` 与阶段进度。
  `extras.signal_conditioning` **只在真实 LSL 会话里出现**（仿真源不做调理），形状为
  `{spec, dc_removal, chain{spec,stages[]}, srate, observed_srate, context_samples, window_samples,
  expected_samples}`；它同时会渲染成报告正文表头的「采集调理」一行，便于复算与审计。
- `GET /api/sessions/{uuid}/artifacts?page=1&limit=50`
  → `{ "items": [{ kind, path, bytes, sha256, exists, download }], "total": 7, "limit": 50, "page": 1 }`
  同一信封与分页参数；单会话产物通常不足 10 条（`report_md/report_json/report_html/heatmap_svg/trend_svg/
  model/history/zip` 的子集），默认一页即可取全，`total` 为登记到 `artifacts` 表的产物总数。
- `GET /api/sessions/{uuid}/artifacts/{kind}` → 文件下载（`report_md/report_json/report_html/heatmap_svg/
  trend_svg/model/history/zip`），带 `Content-Disposition`。
- `GET /api/sessions/{uuid}/export.zip` → 打包全部产物 + `manifest.json`（含 sha256）。

## 模型与趋势

- `POST /api/models/train` → `{ "subjects": 6, "windows_per_state": 8 }`
  评估集只有一种标签时，`auc` 为 `null`（不可计算），不以 `0.5` 代替。
  返回 `spec, version`（引擎 `config.VERSION`）、`features[]`、`subject_split{train,validation,test}`、
  `samples`（= 被试数 × 状态数 × 每状态窗数）、`participants[]`（参与训练的被试编号）、
  `train, validation, test`（后两者在划分不足时可能为 `null`）、`model_path`。
- `GET /api/reports/trend?participant=p01&field=focus&period=week`
  → `points[]`（来自数据库）、`ledger_points[]`（来自 JSONL 台账）、`sessions`、
  `comparable` / `rejected` / `rejected_reasons`（可比性过滤结果）、`note`。
  `ledger_points` 汇总的是**每个会话自己的**台账 `<runs_root>/<uuid>/history/sessions.jsonl`
  （外加 `export-ledger` 导出的 `<data_dir>/history/sessions.jsonl`），按 `session_uuid` 去重；
  聚合口径与 `GET .../{uuid}/trend` 一致（`period/mean/n/std`，并带 `rejected` 计数）。
  两种来源均先按 `participant` 筛选，再按同一设备参照做可比性过滤。
  会话详情和数据库趋势只使用 `monitor` 阶段的汇总评分；该阶段无评分时返回空结果，不回退到基线。

---

## 认证（可选）

以 `serve --token <字符串>` 启动后，**只有 `/api` 请求**必须带令牌，两种方式任选：

- 请求头 `X-API-Token: <字符串>`（前端 `fetch` 用这种）；
- 查询参数 `?token=<字符串>`（SSE 与原生下载链接使用这种；前端自动对令牌做 URL 编码）。

**静态资源（`/`、`/index.html`、`/js/*`、`/css/*` 与 SPA 回退路径）不受令牌约束**：页面本身不含数据，
令牌是接口访问控制；否则带 `--token` 启动时连 SPA 入口都拿不到，页面无法 boot。缺少或错误的令牌返回
`401` + `error.code = "unauthorized"`。

未配置令牌时（默认）本机演示无需令牌。无论是否配置令牌，浏览器请求都必须与本服务同源；
跨域请求（包括 `Origin: null`、预检和简单表单提交）在执行业务前返回 403。
服务还会检查 Host，防止外部域名被当成本服务。无 Origin 的 CLI/API 客户端仍可使用实际服务地址访问。

同一个数据目录同一时间只允许一个服务或 CLI 演示占用。服务重新启动时，会将上次退出遗留的
`running/queued` 会话及未完成阶段收尾为 `failed`，写入原因和结束时间，保留已保存的结果。

1. 实时：优先 `EventSource('/api/sessions/<uuid>/events')`；断线后浏览器自动重连并带 `Last-Event-ID`。
2. 兜底：`setInterval` 轮询 `/live`、`/heatmap`（10 秒）。
3. 交互阶段：收到 `scale_request` / `behavior_request` / `trial` 事件后渲染对应界面，
   逐试次 `POST`；收到 `phase.done` 或 `finished` 后停止。
4. 所有数值均已做 NaN/Inf → `null` 处理；前端需容忍 `null`（表示该窗不可用）。
5. 颜色只用 `config.heatmap_bands` 与 `heatmap.legend` 给的值，不要自造状态色。

### NeuroFocus 接入补充（2026-10-16，向后兼容字段）

- `GET /api/config` 增加 `focus_score.range`、`algorithm_id`、`algorithm_version`、`definition`，来自当前引擎配置，前端不得猜测范围。
- `GET /api/sessions/{uuid}` 的 `runtime` 增加 `source_note`，用于区分模拟 LSL 与真实 LSL 来源。LSL 传输本身不能证明真实硬件或真人采集。
- `EventBus.publish` 产生的会话 SSE 事件增加顶层 `emitted_at_ms`，表示服务器发送的 Unix epoch 毫秒，重放保持原值。它不是设备逐样本时间戳；客户端重连后须检查新鲜度。
- 原 `/signal` 的 `t` 是服务器显示帧时间；只有未抽稀样点可按声明采样率显示窗内相对位置，不能把抽稀后的数组当作原始等间隔采样。
## 测量有效性字段更新

`GET /api/config` 增加 `validation_note`、`threshold_status=pending_validation`；报告和联合评估 JSON 同步携带这两个字段。现有数值阈值保留。基线版本为 `baseline-v2`，`quality` 包含 `usable_seconds`（窗口时间区间并集）、`valid_ratio`、`half_mean_difference`、`stability_status=not_validated`、`min_usable_seconds=30`。不可用原因新增 `insufficient_baseline_duration`、`insufficient_baseline_valid_ratio`、`inconsistent_baseline_data`。

窗口 `quality.metrics.channels` 为各通道的 `channel`（零基索引）、`emg_rel`、`eog_rel`；顶层两个频带代理值是通道最大值。任一通道不合格则整窗拒绝。该门控仍为伪迹粗筛，阈值待验证。

PVT `scoring_spec=pvt-scoring-v3`，保留主 `lapse_sec=0.5`、`lapse_operator=>`；增加 `comparison_lapse_sec=0.355`、`comparison_lapse_operator=>=`、`lapses_355ms`、`lapse_rate_355ms`、`omission_rate`、`invalid_responses`、`rate_denominators`、`unavailable_reasons`、`protocol_note`、`threshold_status`。Studio 另有 `completed`。已作答的 RT<100 ms 视为抢答，负数/缺失/非有限 RT 视为无效；抢答不进入反应时统计和漏答分母。正常漏答不再被零慢反应率掩盖，联合评估检查既有 10% 漏答率阈值（待验证）。调用方须先检查 `valid`，不可将不可用结果中的零统计解释成正常状态。

v3另检查有效反应≥5次、有效反应/全部试次≥60%、抢答/全部试次≤25%、无非法反应时；
Studio还要求任务完成。这些是待验证的工程完整性门槛，由 `/api/config.pvt_validity` 给出。
结果包含 `response_ratio`、`false_start_ratio`、`validity_thresholds`；失败原因包括
`insufficient_response_coverage`、`excessive_false_starts`。原始任务成绩仍保留，不可用任务不进入联合评定。

完整说明见 [测量有效性](../../ningsi/docs/MEASUREMENT_VALIDITY.md)。


### v0.1.2 十分钟协议扩展

`POST /api/sessions` 的 `protocol` 新增 `ten_min`；`GET /api/config` 返回三个协议档。新阶段为 `practice`、`baseline_after`。十分钟PVT作答须携带 `phase: "practice" | "main"`，旧协议不要求。`feedback` 增加 `feedback_mode`、`mode_label`、`grade`；非eeg模式 `score=null`、`usable=false`。报告 `extras.ten_min` 包含阶段时长、前后同通道描述性统计及分轮反馈时长；`signal_diagnostics.grades` 返回A/B/C窗数。旧字段和旧协议语义保持兼容。

十分钟完成页通过 `GET /api/sessions/{uuid}` 的 `result_overview` 读取训练指数：
`source_phase="training"`、`source_label`、`summary.{focus,relax,load}.{mean,std,n}`、
`quality.{windows_total,windows_usable,windows_scored,valid_ratio,scored_ratio}`、
`available`、`reason`、`note`、`excluded_reasons`。仅使用训练阶段严格合格、A级且已有完整有效指数的窗口，
按窗等权汇总未平滑分数；包含反馈升级等待窗，与平滑奖励分数口径不同。
`windows_usable` 是信号严格质检通过数；`windows_scored` 还要求A级与有效基线，两者不能混用。
新会话把同一对象保存至 `runs[training].payload.indicator_overview` 和 `extras.ten_min.indicator_overview`。
训练最终摘要也写入 `runs[training].payload.training`，可由 `/training` 读取。
旧会话可从已保存报告的 `signal_diagnostics.windows` 只读恢复；无逐窗记录时返回空 `summary` 与原因，绝不填零。
`indicator_summary` 继续仅代表独立监测阶段，训练指数不进入联合评估、历史趋势或跨协议对比。
完成页的启动质检窗数仅描述最初质检阶段；十分钟训练另显示可评分窗数。
完整窗口、基线、分级与评分公式见 [脑电窗口与评分口径](EEG_SCORING.md)。

### 2026-11-13 流程修复：时间、阶段边界与证据措辞

- `segment.stats`新增 `valid_seconds, on_target_seconds, valid_time_ratio, on_target_valid_ratio,
  on_target_sample_ratio, coverage_sufficient, timing_spec`。时间以段起点为零，每个有效反馈最多覆盖后续2秒，
  在下一事件、不可用标记或段末截断；长缺口不补齐。`interruptions[]`保存不可用时间点，明确打断连续保持。
  覆盖率不足60%时保持目标；提高目标还要求实际达到连续保持时长。
- 完整/短协议基线与十分钟协议均排除跨阶段滚动窗。普通采集在每个间隔末读取，
  完整协议训练后安排32秒，留出滚动窗建立余量；有效覆盖门槛仍为30秒。
- `/api/sessions`创建响应、详情与`started.phases`均按该会话的协议/训练模式提供阶段表。
  训练阶段有`training_plan{segments,segment_sec,rest_sec,baseline_after_sec,min_wall_segment_sec}`，
  `duration_sec`包含该阶段实际安排。十分钟训练240秒，普通quick/full训练含后基线分别272/512秒。
  协议档`total_sec_by_training_mode`供开始页显示时间，`saved_sec`按同一阶段表计算。
- 联合评定升级为`joint-assessment-v3`：维度状态为`证据不足`、`未触发工程阈值`、
  `单一来源提示偏离`、`部分提示偏离`、`一致提示偏离`。后两种要求至少两个不同来源。
  `dimension_states`增加`sources,source_count,within_source_mixed`；多个脑电指标属于同一来源。
  单独PVT提示明确标注仅有PVT行为证据及缺少其他来源交叉验证，不据此输出个人能力等级。
  未采集独立监测时，建议区说明训练记录单独查看；仅基线无效时才暂停脑电训练目标。
  网页优先显示服务器保存的`report_markdown`，与导出原文一致；只有旧版纯JSON报告使用兼容排版。
