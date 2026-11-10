# 凝思（Ningsi）

**A09 赛题**「AI+便携脑电设备的专注力强化训练系统」的完整实现：算法引擎 + Web 应用，单仓库管理。

本 monorepo 由两个包组成，它们是同一个系统的上下两层，依赖严格单向：

```
ningsi-studio/   Web 产品层：浏览器 + HTTP 服务 + SQLite/JSONL 持久化
      │  单向依赖（64 处 import / 22 个文件）
      ▼
ningsi/          算法引擎：信号处理、量表、行为任务、联合评估、训练、预警、模型
```

| 目录 | 包名 / 命令 | 职责 | 独立运行 |
|---|---|---|---|
| [`ningsi/`](ningsi/README.md) | `ningsi` / `ningsi` | 算法引擎 + CLI + Tkinter 桌面版。输入便携脑电（LSL 真机或仿真源），输出专注/放松/负荷三指标、量表 + 行为 + 脑电联合评估、神经反馈训练、报告 / 热力图 / 趋势、逻辑回归基线模型 | ✅ 仅依赖 NumPy |
| [`ningsi-studio/`](ningsi-studio/README.md) | `ningsi-studio` / `ningsi-studio` | Web 产品层。自研标准库 HTTP 服务、SSE 实时流、SQLite(WAL) 权威库 + JSONL 审计台账、原生 ES Module 前端（10 个视图）、11 阶段会话状态机、NeuroFocus 监测台、可打包单文件 exe | ❌ 需要引擎 |

## 发行版下载与使用（Windows x64）

无需安装 Python。当前发行版为 **v0.1.0**，对应源码提交 [`c3756c3`](https://github.com/zjhaaa042-cloud/ningsi-suite/commit/c3756c352a6a2dd6787a83eea3868618191281b1)。

- [发行说明与更新内容](https://github.com/zjhaaa042-cloud/ningsi-suite/releases/tag/v0.1.0)
- [下载 Windows x64 便携版 ZIP（约 22 MB）](https://github.com/zjhaaa042-cloud/ningsi-suite/releases/download/v0.1.0/ningsi-studio-0.1.0-c3756c3-win-x64.zip)
- [下载 SHA-256 校验文件](https://github.com/zjhaaa042-cloud/ningsi-suite/releases/download/v0.1.0/ningsi-studio-0.1.0-c3756c3-win-x64.zip.sha256)
- [中文使用说明](https://github.com/zjhaaa042-cloud/ningsi-suite/releases/download/v0.1.0/README.zh-CN.txt) · [构建环境与版本清单](https://github.com/zjhaaa042-cloud/ningsi-suite/releases/download/v0.1.0/BUILD.json)

### 解压与运行

1. 下载 ZIP 和同名 `.sha256` 文件，按下方方法校验。
2. 将 ZIP **完整解压**到可写目录；进入解压后的文件夹，保留 EXE 旁的 `_internal` 文件夹。
3. 双击 `ningsi-studio-0.1.0-c3756c3.exe`，程序会启动本机服务并打开浏览器。默认地址为 `http://127.0.0.1:8765`；端口被占用时可能顺延，以控制台显示地址为准。
4. 无设备时可点击「运行模拟演示」；真实脑电设备需通过其采集软件发布 LSL EEG 流。
5. 使用结束后关闭控制台程序以停止服务。数据默认保存在程序旁的 `var/studio`；目录不可写时使用用户本地应用数据目录。

如需诊断，在解压后的程序目录打开 PowerShell，运行：

```powershell
.\ningsi-studio-0.1.0-c3756c3.exe doctor
```

该发行包已验证 EXE 启动、完整仿真会话与报告、本机服务接口、Studio/NeuroFocus 页面，以及解压后的启动诊断。**真实硬件完整采集流程尚未验证，程序未做代码签名。** 脑电指数不是专注概率，评分与判定阈值仍待验证，详见[测量有效性说明](ningsi/docs/MEASUREMENT_VALIDITY.md)。

### SHA-256 校验

在 ZIP 和 `.sha256` 文件所在目录打开 PowerShell，执行：

```powershell
$archive = 'ningsi-studio-0.1.0-c3756c3-win-x64.zip'
$expected = ((Get-Content -LiteralPath "$archive.sha256" -Raw).Trim() -split '\s+')[0]
$actual = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash
if ($actual -ne $expected) { throw 'SHA-256 不一致，请重新下载并核对文件。' }
'SHA-256 校验通过'
```

此版本 ZIP 的预期 SHA-256 为：

```text
5be88543c3a446843e25226eea4ac441fef17757d0cc77fed2bbee296c25781a
```

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
├── LICENSE                   MIT（覆盖两个包）
├── ningsi/                   算法引擎包
│   ├── src/ningsi/           signal / scales / behavior / assessment / training /
│   │                         monitoring / models / acquisition / adapters / app
│   ├── tests/                63 个用例（信号、量表、行为、评估、预警、热力图、趋势、训练、模型、端到端）
│   ├── examples/sample_run/  一次完整会话的示例产物（免运行查看）
│   ├── docs/                 产品使用说明、赛题覆盖对照、文档口径修订清单
│   ├── tools/                PPT / docx / 演示视频等交付物生成脚本
│   └── scripts/verify.ps1    一键验证：跑测试 + 端到端演示
└── ningsi-studio/            Web 应用包
    ├── src/ningsi_studio/    http / api / core / domain / db / acquisition / settings
    ├── web/                  原生 ES Module 前端（含 neurofocus/ 独立监测台）
    ├── tests/                187 个用例（API、会话并发、SSE、数据库、LSL、协议、前端契约）
    ├── docs/                 接口契约与 NeuroFocus 说明
    └── run.bat / start.bat   启动器（另有 run.ps1 / run.sh）
```

## 关系与口径

- **单向依赖**：`ningsi-studio` 通过 `bootstrap.py` 定位并 `import ningsi`；`ningsi` 不引用 Web 应用，可独立安装与运行。
- **口径单一来源**：所有可复算参数集中在 `ningsi/src/ningsi/config.py`（`welch-v1` / `indicator-v1` / `baseline-v2` / `joint-assessment-v2` / `neurofeedback-v2` 等），CLI、桌面版、Web 三端结论一致。
- **数据一致**：SQLite 为权威库，另以与引擎同格式的 JSONL 台账双写（`history/sessions.jsonl`、`scales/*.jsonl`），可用 `python -m ningsi_studio export-ledger` 从库重放。
- **上游**：`bsense-lsl` / `bsense-dataset-studio` 保持只读，本仓库不复制其代码，只通过公开接口读取其数据产物。

## 测试

两个包的测试都只用标准库 `unittest`，不需要浏览器、真实设备或外网：

```powershell
# 可选：先安装两个包（也可不装，用 PYTHONPATH 指向 src）
pip install -e ningsi -e ningsi-studio

cd ningsi
python -m unittest discover -s tests -t .        # 63 个用例

cd ..\ningsi-studio
python -m unittest discover -s tests -t .        # 187 个用例
```

未安装时可在各包目录下 `$env:PYTHONPATH="src"` 后再执行同一条命令；
`ningsi-studio` 的测试会通过 `bootstrap.py` 自动从同级 `..\ningsi\src` 找到引擎。

各包自带的启动器也可用：`ningsi\scripts\verify.ps1`（测试 + 端到端演示）、
`ningsi-studio\run.bat test`、`ningsi-studio\run.ps1 -Action check`（静态自检）。

## 文档

- 引擎：[产品使用说明（系统架构与流程）](ningsi/docs/PRODUCT_GUIDE.md) · [赛题要求覆盖对照](ningsi/docs/REQUIREMENT_COVERAGE.md) · [文档口径修订清单](ningsi/docs/DOC_ALIGNMENT.md)
- Web：[使用说明（零基础）](ningsi-studio/使用说明.md) · [STUDIO-GUIDE](ningsi-studio/STUDIO-GUIDE.md) · [接入真实脑电数据](ningsi-studio/接入真实脑电数据.md) · [接口契约](ningsi-studio/docs/API.md)

## 许可证

MIT，见 [LICENSE](LICENSE)。仅覆盖代码本身，不构成对任何评估结论的医学背书；
系统用于研究、竞赛与自我调节训练，不提供医疗诊断。
