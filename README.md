# 凝思（Ningsi）

**A09 赛题**「AI+便携脑电设备的专注力强化训练系统」。项目包含算法引擎、Studio 评估平台和独立实验采集器。

当前源码版本为引擎 0.1.2、Studio 0.1.13、Capture 0.4.1，以各包 `pyproject.toml` 为准。Capture 提供连续原始采集、因果调理与伪迹标记；Studio 提供联合评估、反馈、历史趋势及真人模型训练与部署。已有 Windows 包以包内构建记录为准。

Studio 按「评估与训练」「记录与结果」组织日常操作；设备连接、参与者和模型管理集中在默认折叠的「设置与管理」。赛事展示页已退出产品导航与路由，保留的开发分析模块与接口见[开发说明](ningsi-studio/docs/SHOWCASE.md)。

Studio 0.1.10 增加按数据源保存的设备档案，以及开始前的单位、采样率和模型适配检查；配置通过后仍需采集质检和个人基线。Capture 0.4.1 可选发布不含答案的 LSL 事件，供 LabRecorder 独立同步记录。原始信号复核、成熟工具依据和现场步骤见[现场验证说明](docs/EXISTING_SOLUTIONS_AND_FIELD_VALIDATION.md)及[验收清单](docs/FIELD_ACCEPTANCE_CHECKLIST.txt)。

Studio 0.1.12 新增“有效时长训练”：坏窗暂停累计、信号恢复后有限补采，前后基线各累计 60 秒、4 轮训练各累计 50 秒可用反馈；PVT 仍按固定时长运行。训练主要展示有效时间内达标率，并同时报告覆盖、有效秒数和用时；达到补采上限仍不足会明确标记。质量门槛保持原规则，原固定十分钟方案继续保留。

Studio 0.1.13 在设备页补入 BSense-R 手册参考、连接与推流步骤、两类质量指示和明确的资料草稿入口。CSV 单位与实际 LSL 配置分别核对，手册没有说明的参考／地及 LSL 通道映射继续待确认；详见[设备接入说明](docs/BSENSE_R_DEVICE_GUIDE.md)。

项目按“采集 → 预处理 → 特征 → 状态估计 → 联合评估 → 反馈”组织。开发先看[系统架构](docs/ARCHITECTURE.md)，技术验收看[验证计划](docs/VALIDATION_PLAN.md)。A09 没有给出准确率或毫秒时延门槛；公开数据研究尚未证实三类状态在目标设备上的稳定识别，最新个人校准量实验见本机 [研究结果](var/calibration-dose-20261116/RESULTS.md)。

Studio 和 Capture 分别使用同一仓库的算法引擎。Capture 通过研究数据导出与 Studio 训练入口衔接。

```
ningsi-studio/ ──┐  评估、监测、反馈、训练与持久化
                ├──▶ ningsi/  共用算法引擎与逐窗应用服务
ningsi-capture/ ─┘  原始记录、研究任务、探针与导出

Capture ──带版本与校验值的导出文件──▶ Studio 真人模型训练
```

| 目录 | 包名 / 命令 | 职责 | 独立运行 |
|---|---|---|---|
| [`ningsi-capture/`](ningsi-capture/README.md) | `ningsi-capture` | 连续 LSL 原始采集、任务事件、走神探针、质量筛查与训练数据导出 | 随 Windows 程序包含所需引擎 |
| [`ningsi/`](ningsi/README.md) | `ningsi` / `ningsi` | 算法引擎 + CLI + Tkinter 桌面版。输入便携脑电（LSL 真机或仿真源），输出专注/放松/负荷三指标、量表 + 行为 + 脑电联合评估、神经反馈训练、报告 / 热力图 / 趋势、逻辑回归基线模型 | ✅ 仅依赖 NumPy |
| [`ningsi-studio/`](ningsi-studio/README.md) | `ningsi-studio` / `ningsi-studio` | Web 产品层：HTTP/SSE、会话状态机、SQLite/JSONL、NeuroFocus、模型训练与报告 | 需要引擎；便携包包含所需运行资源 |

## 发行版下载与使用（Windows x64）

当前正式版为 **v0.1.12**，包含 Studio 0.1.12、Capture 0.4.1 和引擎 0.1.2，无需安装 Python。

- [发行说明与更新内容](https://github.com/zjhaaa042-cloud/ningsi-suite/releases/tag/v0.1.12)
- [下载 Windows x64 完整便携套件（约 49 MB）](https://github.com/zjhaaa042-cloud/ningsi-suite/releases/download/v0.1.12/ningsi-product-0.1.12-20261117.zip)
- [下载 SHA-256 校验文件](https://github.com/zjhaaa042-cloud/ningsi-suite/releases/download/v0.1.12/ningsi-product-0.1.12-20261117.zip.sha256)
- [中文使用说明](https://github.com/zjhaaa042-cloud/ningsi-suite/releases/download/v0.1.12/README.zh-CN.txt) · [构建记录](https://github.com/zjhaaa042-cloud/ningsi-suite/releases/download/v0.1.12/BUILD.json) · [EXE 验收记录](https://github.com/zjhaaa042-cloud/ningsi-suite/releases/download/v0.1.12/FROZEN_VERIFICATION.json)

完整解压到可写目录，保留整个 `_internal`，双击 `01-Start-Studio.bat` 打开平台，或 `02-Start-Capture.bat` 打开采集器。升级请使用新目录；需要原会话时先备份数据，再按包内说明指定旧数据目录。

本版增加有效时长训练、坏窗暂停与有限补采，分开显示有效时间内达标率、覆盖和用时，并保留热力图、设备档案及开始前配置检查。完整回归运行 619 项：616 项通过、3 项可选检查跳过，最终 EXE 通过 35 项验收。程序未签名；真人识别与训练效果仍需独立验证。套件不含私人脑电或会话数据库。

### v0.1.2 历史发行版

以下为 **v0.1.2** 历史发行版的使用说明。该版无需安装 Python，源码见 [v0.1.2 标签](https://github.com/zjhaaa042-cloud/ningsi-suite/tree/v0.1.2)，构建提交见其发行附件 `BUILD.json`。

- [发行说明与更新内容](https://github.com/zjhaaa042-cloud/ningsi-suite/releases/tag/v0.1.2)
- [下载 Windows x64 便携版 ZIP（约 22 MB）](https://github.com/zjhaaa042-cloud/ningsi-suite/releases/download/v0.1.2/ningsi-studio-0.1.2-win-x64.zip)
- [下载 SHA-256 校验文件](https://github.com/zjhaaa042-cloud/ningsi-suite/releases/download/v0.1.2/ningsi-studio-0.1.2-win-x64.zip.sha256)
- [中文使用说明](https://github.com/zjhaaa042-cloud/ningsi-suite/releases/download/v0.1.2/README.zh-CN.txt) · [构建环境与版本清单](https://github.com/zjhaaa042-cloud/ningsi-suite/releases/download/v0.1.2/BUILD.json)

### v0.1.2 更新

- 新增独立「十分钟测试＋训练」协议：30秒质检、60秒睁眼基线、30秒PVT练习、180秒正式PVT、四轮50秒训练＋10秒休息、60秒训练后采集。报告生成另计，设备阻塞时记录实际超时。
- A/B/C分级：A用于探索性脑电反馈；B仅保留可计算的同通道片段用于描述与质量反馈；C使用固定引导。严格评分门槛未放宽，不保证所有低质量数据可以恢复。
- 十分钟协议基线采用中位数/MAD，原有完整和短协议保持原口径；练习不计分，PVT练习/正式作答阶段隔离。
- 报告增加分级统计、同通道描述性前后变化及各轮实际反馈时长。工程阈值和新分级规则待设备验证，不作诊断或疗效证明。
- 本版验收使用仿真数据和打包程序；尚未完成v0.1.2的真人真机十分钟全流程。发行包不含私人脑电或量表数据。

### v0.1.1 历史更新

- 修复真实 LSL 启动时半窗即返回、质检连续重复读取缓冲的问题；真机质检按2秒步长取完整4秒窗。
- 新增质量窗口序列、排除原因、基线覆盖、分组频带、幅值阈值敏感性和 PVT 可视化；新增离线 HTML 报告，随会话 ZIP 导出。
- 个体脑电评分不可用时仍提供采集诊断与独立行为分析；修正无效基线下的训练建议，并保留阈值待验证说明。
- 诊断窗与任务态评分窗区分展示；旧会话缺失逐窗频带时明确提示，不事后补造。训练诊断不改变原有指标序列。

### 解压与运行

1. 下载 ZIP 和同名 `.sha256` 文件，按下方方法校验。
2. 将 ZIP **完整解压**到可写目录；进入解压后的文件夹，保留 EXE 旁的 `_internal` 文件夹。
3. 双击 `ningsi-studio-0.1.2.exe`，程序会启动本机服务并打开浏览器。默认地址为 `http://127.0.0.1:8765`；端口被占用时可能顺延，以控制台显示地址为准。若提示来源校验失败，请在浏览器地址栏手动输入该地址并回车。
4. 在「检测目的」选择「十分钟测试＋训练」，或选择原有评估／完整训练流程。
5. 无设备时可点击「运行模拟演示」；真实脑电设备需通过其采集软件发布 LSL EEG 流。
5. 使用结束后关闭控制台程序以停止服务。数据默认保存在程序旁的 `var/studio`；目录不可写时使用用户本地应用数据目录。

如需诊断，在解压后的程序目录打开 PowerShell，运行：

```powershell
.\ningsi-studio-0.1.2.exe doctor
```

该发行包已验证 EXE 启动、完整仿真会话、可视化/HTML 报告与导出、本机服务接口及打包页面。曾用 v0.1.0 完成真机流程并发现启动质检问题，本版据此修复；**v0.1.2 尚未重新完成真实硬件全流程验证，程序未做代码签名。** 脑电指数不是专注概率，评分与判定阈值仍待验证，详见[测量有效性说明](ningsi/docs/MEASUREMENT_VALIDITY.md)。

升级时解压到新目录，先保留旧程序及其 `var/studio` 数据；需要迁移时先停止旧程序并备份数据，再将 `var/studio` 复制到新程序旁，或通过 `NINGSI_STUDIO_DATA` 指定数据目录。不要把旧的 `_internal` 混入新目录。

### SHA-256 校验

在 ZIP 和 `.sha256` 文件所在目录打开 PowerShell，执行：

```powershell
$archive = 'ningsi-studio-0.1.2-win-x64.zip'
$expected = ((Get-Content -LiteralPath "$archive.sha256" -Raw).Trim() -split '\s+')[0]
$actual = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash
if ($actual -ne $expected) { throw 'SHA-256 不一致，请重新下载并核对文件。' }
'SHA-256 校验通过'
```

预期 SHA-256 以本版本同名 `.sha256` 附件为准；包内 `FILES.sha256` 提供逐文件校验。发行包不包含真实脑电、量表作答或测试数据库。

## 从源码运行

### 1. Web 应用（主界面，推荐）

```powershell
cd ningsi-studio
.\run.bat            # 双击亦可：建环境 → 启动服务 → 打开浏览器
# 或最省事、不经过 PowerShell：
start.bat
```

引擎源码就在同级 `..\ningsi`，启动脚本会自动定位（`..\ningsi\src`），**无需单独安装引擎**。
服务默认监听 `http://127.0.0.1:8765`。

无浏览器跑一次完整会话：

```powershell
.\run.ps1 -Action demo
```

### 2. 算法引擎（CLI / 桌面版 / 单独使用）

```powershell
# 不安装，直接指向 src 运行
$env:PYTHONPATH="ningsi\src"
python -m ningsi.app.cli demo --root var/demo --participant p01

# 或安装后使用命令名
pip install -e ningsi
ningsi demo --root var/demo --participant p01
ningsi ui                 # Tkinter 桌面界面（离线兜底）
ningsi self-test          # 运行引擎全部测试
```

依赖：Python 3.11+ 与 NumPy；接真实设备时可选 `pylsl==1.18.2`。

## 目录结构

```
ningsi-suite/
├── README.md                 本文件：整体说明与快速开始
├── LICENSE                   MIT（自研代码）
├── docs/                     当前系统架构与技术验证计划
├── scripts/                  整仓验证与架构依赖检查
├── ningsi/                   算法引擎包
│   ├── src/ningsi/           signal / scales / behavior / assessment / training /
│   │                         monitoring / models / acquisition / processing / adapters / app
│   ├── tests/                算法、共享逐窗服务与端到端测试
│   ├── examples/sample_run/  一次完整会话的示例产物（免运行查看）
│   ├── docs/                 产品使用说明、赛题覆盖对照、文档口径修订清单
│   ├── tools/                PPT / docx / 演示视频等交付物生成脚本
│   └── scripts/verify.ps1    一键验证：跑测试 + 端到端演示
├── ningsi-studio/            Web 应用包
│   ├── src/ningsi_studio/    http / api / core / domain / db / acquisition / settings
│   ├── web/                  原生 ES Module 前端（含 neurofocus/ 独立监测台）
│   ├── tests/                API、会话、数据库、协议、跨端一致性与前端契约测试
│   ├── docs/                 接口契约与 NeuroFocus 说明
│   └── run.bat / start.bat   启动器（另有 run.ps1 / run.sh）
└── ningsi-capture/           独立研究采集器
    ├── src/ningsi_capture/   acquisition / storage / protocol / export / gui
    └── tests/               采集、时标、质控、探针预测与导出测试
```

## 关系与口径

- **单向依赖**：Studio 和 Capture 分别通过 `bootstrap.py` 定位引擎，产品包之间不互相导入；引擎可独立安装与运行。
- **按用途固定口径**：相对指数参数在 `ningsi/src/ningsi/config.py`，调理和研究特征另有版本契约；模型核对训练时的来源与实现指纹。CLI 任务监测与 Studio 共用 `ningsi.processing.process_window`，相同输入、基线和质量证据得到相同逐窗指标。
- **分开解释输出**：相对基线指数、模型估计、自评和信号质量各自保存。无效信号保持缺失，不作为低专注分数。
- **数据一致**：SQLite 为权威库，另以与引擎同格式的 JSONL 台账双写（`history/sessions.jsonl`、`scales/*.jsonl`），可用 `python -m ningsi_studio export-ledger` 从库重放。
- **上游**：`bsense-lsl` / `bsense-dataset-studio` 保持只读，本仓库不复制其代码，只通过公开接口读取其数据产物。

## 测试

三个包使用标准库 `unittest`。在根目录运行整仓验证（Python 3.11+，优先使用工作区 `.venv`；部分测试需要本机回环 HTTP）：

```powershell
.\scripts\verify.ps1
.\scripts\verify.ps1 -ArchitectureOnly
```

也可分别运行：

```powershell
# 可选：安装三个包（也可不装，用 PYTHONPATH 指向 src）
pip install -e ningsi -e ningsi-studio -e ningsi-capture

cd ningsi
python -m unittest discover -s tests -t .

cd ..\ningsi-studio
python -m unittest discover -s tests -t .

cd ..\ningsi-capture
python -m unittest discover -s tests -t .
```

未安装时可在各包目录下 `$env:PYTHONPATH="src"` 后再执行同一条命令；
`ningsi-studio` 的测试会通过 `bootstrap.py` 自动从同级 `..\ningsi\src` 找到引擎。

各包自带的启动器也可用：`ningsi\scripts\verify.ps1`（测试 + 端到端演示）、
`ningsi-studio\run.bat test`、`ningsi-studio\run.ps1 -Action check`（静态自检）。

## 文档

- 当前开发：[系统架构](docs/ARCHITECTURE.md) · [技术验证计划](docs/VALIDATION_PLAN.md) · [真人模型训练](ningsi-studio/docs/HUMAN_MODEL_TRAINING.md) · [Capture 说明](ningsi-capture/README.md)
- 历史与研究：[产品使用说明初版](ningsi/docs/PRODUCT_GUIDE.md) · [赛题要求覆盖快照](ningsi/docs/REQUIREMENT_COVERAGE.md) · [文档口径修订清单](ningsi/docs/DOC_ALIGNMENT.md) · [新队列验证](ningsi/docs/NEW_COHORT_VALIDATION.md)
- Web：[使用说明（零基础）](ningsi-studio/使用说明.md) · [STUDIO-GUIDE](ningsi-studio/STUDIO-GUIDE.md) · [接入真实脑电数据](ningsi-studio/接入真实脑电数据.md) · [接口契约](ningsi-studio/docs/API.md)

## 许可证

MIT，见 [LICENSE](LICENSE)。仅覆盖代码本身，不构成对任何评估结论的医学背书；
系统用于研究、竞赛与自我调节训练，不提供医疗诊断。
