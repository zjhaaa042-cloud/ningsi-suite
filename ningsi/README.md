# 凝思（Ningsi）

仓库：`ningsi`（https://github.com/zjhaaa042-cloud/ningsi ）｜Python 包与命令：`ningsi`｜版本：`0.1.0`

基于便携脑电设备的**专注力强化训练与心理状态评估系统**，对应 A09 赛题
「AI+便携脑电设备的专注力强化训练系统」（杭州金扬智能科技有限公司）。

凝思 = 上游采集工程 `bsense-suite`（BSense 采集套件 + Dataset Studio）之上的**产品层**：
补齐了上游只有采集与质检、而没有的量表、三指标、联合评估、神经反馈训练、报告、趋势、预警与热力图。

## 1. 能做什么

| 能力 | 实现位置 | 说明 |
|---|---|---|
| 真实硬件采集 | `acquisition/lsl_source.py` | 通过 LSL 接入任意厂商 EEG 流；未接设备时用 `simulate.py` 联调 |
| 四级信号处理链 | `signal/preprocess.py` | 0.5 Hz 基线漂移校正 → 50/60 Hz 工频陷波 → 45 Hz 低通（0.5–45 Hz 带通），零相位 |
| 特征提取 | `signal/spectrum.py`、`bands.py`、`window.py` | Welch PSD（4 s 窗 / 2 s 分段 / 50% 重叠 / 汉宁窗）、θ/α/β/γ 相对功率、波幅与通道相关系数 |
| 伪迹与质量门控 | `signal/artifacts.py` | 幅度、恒定/贴轨、通道跨度、肌电与眼电代理，逐窗记录不可用原因 |
| 三个可解释指标 | `signal/indicators.py` | 专注度 β/θ、放松度 α/β、认知负荷 θ/α，相对个体基线做 z 标准化后映射到 0–1 |
| 量表联合评估 | `scales/`、`assessment/joint.py` | SAS/SDS（20 题、含反向题、粗分×1.25）+ 脑电指标 + 行为任务三类证据 |
| 行为任务 | `behavior/sart.py`、`pvt.py` | SART 180 试次（20 No-Go、8 套可复现序列）、PVT-B 3 分钟 |
| 神经反馈训练 | `training/neurofeedback.py` | 专注度驱动反馈、分段记录、按表现自适应调整目标与保持时长 |
| 状态预警 | `monitoring/alerts.py` | 低专注 / 高负荷 / 质量不足，按连续越界时长触发，伪迹窗不计时 |
| 热力图与趋势 | `monitoring/heatmap.py`、`history.py` | 五档着色 + 低质量缺失色块、周/月趋势 SVG、跨设备不可比提示 |
| 可训练模型 | `models/logistic.py` | 逻辑回归基线模型、按被试划分、AUC/混淆矩阵、JSON 版本化保存 |
| 结构化报告 | `assessment/report.py` | 状态评分 / 问题分析 / 改善建议 / 证据回填 / 边界声明（Markdown + JSON） |
| 桌面界面 | `app/ui.py` | 六个页签：状态指标、实时波形与频谱、状态热力图、神经反馈训练、评估报告、历史趋势 |

## 2. 快速开始

```powershell
# 方式一：直接运行（无需安装）
$env:PYTHONPATH="src"
python -m ningsi.app.cli demo --root var/demo --participant p01

# 方式二：可编辑安装后使用命令名
pip install -e .
ningsi demo --root var/demo --participant p01
ningsi ui                     # 桌面界面
ningsi self-test              # 运行全部测试
ningsi import-studio --dataset-root <上游 dataset_root> --root var/demo
```

依赖：Python 3.11+ 与 NumPy（必需）；`pylsl==1.18.2`（接真实设备时可选）。

一次 `demo` 会完整跑通：设备质检 → 量表 → 静息基线 → SART + PVT-B → 指标与预警 →
联合评估报告 → 训练闭环（含训练后基线）→ 模型训练与被试级评估 → 历史与趋势，并写出：

```
var/demo/
├── reports/sub-p01_ses-01_run-001_report.md     评估报告（含证据回填）
├── reports/sub-p01_ses-01_run-001_report.json   机器可读版本
├── reports/sub-p01_ses-01_run-001_heatmap.svg   状态热力图
├── reports/trend.svg                            专注度周趋势
├── scales/sub-p01_ses-01_run-001_scales.jsonl   量表作答与计分
├── models/classifier.json                       基线模型权重与指标
└── history/sessions.jsonl                       历史记录（趋势来源）
```

## 3. 口径与版本

所有可复算参数集中在 `config.py`，报告会带出口径版本：

| 口径 | 版本号 | 关键参数 |
|---|---|---|
| 频谱 | `welch-v1` | 4 s 窗、2 s 分段、50% 重叠、汉宁窗、NFFT 512（Δf 0.488 Hz）、0–45 Hz；0.5 Hz 漂移校正 + 50/60 Hz 陷波 |
| 指标 | `indicator-v1` | 专注 β/θ、放松 α/β、负荷 θ/α；z 裁剪 ±3；逻辑映射斜率 1.1 |
| 基线 | `baseline-v1` | 睁眼 2 分钟 + 闭眼 2 分钟（各 60 窗），两段独立建基线，任务态以睁眼基线为参照；至少 5 个合格窗；离散度下限 0.15 nat 或均值的 10% |
| 联合评估 | `joint-assessment-v1` | 低专注 <0.40、高负荷 ≥0.65、量表提示 ≥50 分 |
| 训练 | `neurofeedback-v1` | 初始目标 = 基线中位数对应评分；步长 0.05；保持时长 6–20 s |
| 模型 | `logistic-v1` | 6 维频带特征；被试级 6:2:2 划分 |

## 4. 目录结构

```
ningsi/
├── src/ningsi/         38 个模块（signal / scales / behavior / assessment / training / monitoring / models / acquisition / adapters / app）
├── tests/              52 个用例（信号、量表、行为、评估、预警、热力图、趋势、训练、模型、端到端）
├── examples/sample_run/ 一次完整会话的示例产物（报告、热力图、趋势、量表、模型、历史）
├── tools/              revise_docs.py（docx 口径修订）、build_ppt.py（PPT 与配图）、relayout_figures.py（docx 插图版式优化）
├── docs/               产品使用说明、赛题覆盖对照、文档口径修订清单
├── examples/           示例运行产物（供评审直接查看，无需运行）
├── scripts/verify.ps1  一键验证：跑测试 + 跑端到端演示
└── var/                本地运行产物（已加入 .gitignore）
```

### 交付物生成顺序（改动文档或配图后按此重跑）

```powershell
python tools/revise_docs.py        # 1. 从原始 docx 生成 v2（封面、口径、表格）
python tools/relayout_figures.py   # 2. 优化 v2 内所有插图的版式（外边距 + 标题带留白）
python tools/build_ppt.py          # 3. 生成 PPT 骨架与 5 张配图
```

`relayout_figures.py` 的版式规则：检测顶部标题带 → 补上/左/右内边距 → 拉开标题与副标题行距 →
整图加白色外边距 → 补白还原原始宽高比（避免 Word 拉伸变形）；原文件自动备份到 `_analysis/backup/`。
若目标 docx 正在 Word 中打开，会改写到 `*-插图优化.docx` 而不覆盖。

## 5. 与上游工程的关系

- 直接复用：`bsense-dataset-studio` 的采集产物通过 `adapters/studio.py` 读入（`derived/features/records.csv` 的质量与行为列），用于真实试采数据的趋势与证据回填。
- 不修改上游：`bsense-suite/` 两个仓库保持只读；凝思以独立包形式追加，避免污染上游历史。
- 数据口径一致：4 s 窗 / 2 s 步长、被试级划分、匿名编号、`restricted/participants` 受限目录等约定与上游一致。

## 6. 边界与隐私

- 系统用于研究、竞赛与自我调节训练，**不提供医疗诊断**，量表结果只作提示，不作诊断依据。
- M6 类结论不得用于处罚、自动上岗决策或永久能力画像。
- 被试编号必须匿名且跨会话一致；姓名等直接身份信息不进入文件名、Marker 与报告。
- 未接设备时的全部输出来自仿真源，界面与报告中已显式标注数据来源。

## 7. 上游血缘、许可与复现

| 上游仓库 | 版本 | 克隆时 HEAD | 本项目如何使用 |
|---|---|---|---|
| https://github.com/shaun5297/bsense-lsl | 0.8.0 | `578de55`（2026-09-15） | 只读参照：采集协议、实时监测、质量门控的实现证据 |
| https://github.com/shaun5297/bsense-dataset-studio | 0.2.0 | `77669cf`（2026-09-11） | 只读参照 + `adapters/studio.py` 读取其 `derived/features/records.csv` |

- 本仓库不复制上游代码，以独立包形式追加产品层能力；上游的提交历史与远端配置保持不变。
- 许可：本项目自研代码以 **MIT** 许可证发布（见 [LICENSE](LICENSE)）；上游 `bsense-lsl` / `bsense-dataset-studio` 未附带许可证，本项目未复制其代码，只通过公开接口读取其数据产物。
- MIT 许可只覆盖代码本身，不构成对任何评估结论的医学背书；结论使用边界见第 6 节。
- 复现：`scripts/verify.ps1` 可一键复跑 52 个用例与一次端到端演示；示例产物见 `examples/sample_run/`。

---

## 8. Web 应用（主界面）：ningsi-studio

Web 应用独立成库：[`ningsi-studio`](https://github.com/zjhaaa042-cloud/ningsi-studio)（前端 + 后端 + 数据保存），以本仓库为算法引擎：

```powershell
git clone https://github.com/zjhaaa042-cloud/ningsi-studio.git
cd ningsi-studio
.\run.bat                  # 双击也可：建环境 → 启动服务 → 打开浏览器
.\run.ps1 -Action demo     # 无浏览器跑一次完整会话（快速模式）
```

- 能力分工：本仓库负责算法与口径，`ningsi-studio` 负责服务、持久化与界面；
  两者共用同一份 `config.py` 与同一批领域模块，因此三端结论一致。
- 界面：单页 Web（原生 ES Module，离线可用）七个视图：概览 / 被试管理 / 会话流程（十一阶段向导）/
  实时监测（指标 + 波形 + 热力图 + 预警时间轴）/ 训练中心 / 历史与趋势 / 报告。
- 数据：SQLite（WAL）为权威库，另以与本仓库同格式的 JSONL 台账双写（`history/sessions.jsonl`、
  `scales/*.jsonl`），可用 `python -m ningsi_studio export-ledger` 从库重放。
- 关系：`ningsi ui`（Tkinter 桌面界面）保留为**离线兜底**，命令行与桌面版行为不变；
  现场演示以 Web 版为主。

---

详细使用说明见 [产品使用说明（系统架构与流程）](docs/PRODUCT_GUIDE.md)，
赛题逐条对照见 [赛题要求覆盖对照](docs/REQUIREMENT_COVERAGE.md)，
与两份交付文档的口径差异见 [文档口径修订清单](docs/DOC_ALIGNMENT.md)。
