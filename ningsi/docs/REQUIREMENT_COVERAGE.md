# 赛题要求覆盖对照（A09）

对照对象：`bsense-suite/【A09】AI+便携脑电设备的专注力强化训练系统【金扬智能】.docx`。
状态含义：**已实现** = 代码存在且有自动化用例或可运行产物；**部分实现** = 主链路可用但仍有边界；
**待实测** = 需要真实设备或真实被试才能确认；**路线图** = 本版本不承诺。

## 一、技术要求与指标

| 赛题要求 | 状态 | 实现位置 | 验证方式 |
|---|---|---|---|
| 使用合规消费级/科研级脑电硬件 | 部分实现 | `acquisition/lsl_source.py` | LSL 抽象层与本机化告警；实际硬件链路需现场验证 |
| 支持单通道及以上实时采集 | 已实现 | `LslSource.pull_window`、`analyze_window` | 单/双通道均通过；`tests/test_signal.py` |
| 基线漂移校正 | 已实现 | `signal/preprocess.py`（0.5 Hz 零相位高通） | `tests/test_preprocess_model.py::test_drift_and_dc_removed` |
| 带通滤波（适配 θ/α/β） | 已实现 | 0.5–45 Hz（高通 + 低通） | `test_alpha_band_preserved` |
| 工频陷波 50 Hz / 60 Hz | 已实现 | RBJ notch，Q=30，零相位 | `test_notch_removes_mains` |
| 伪迹去除（眼电、肌电、运动） | 已实现 | `signal/artifacts.py`（幅度/恒定/跨度/EMG/EOG 代理） | `tests/test_signal.py::QualityTest` |
| 提取 θ/α/β 功率谱密度特征 | 已实现 | `signal/spectrum.py`（Welch 4 s/2 s/50%/汉宁）+ `bands.py` | `test_peak_lands_in_alpha_band`、`test_relative_powers_sum_to_one` |
| 时域特征（波幅、相关系数） | 已实现 | `signal/window.py::time_domain_features` | `WindowFeatures.as_dict()["time_domain"]` |
| 空域特征（脑地形图） | 路线图 | —— | 当前设备为前额 1–2 通道，地形图信息量有限，报告中不承诺 |
| 模型可训练与部署 | 已实现 | `models/logistic.py`（逻辑回归 + 被试级划分 + AUC + JSON 保存） | `tests/test_preprocess_model.py::ModelTest`；`examples/sample_run/models/classifier.json` |
| 深度学习（CNN/LSTM）选型 | 路线图 | 上游 `dataset/eegnet.py` 已能导出 `X[N,C,T]` 窗口 | 作为后续工作，不影响当前解释性 |
| 脑电 + 量表 + 专注力指标联合映射 | 已实现 | `assessment/joint.py`（六步流程 + 一致性 + 缺失分支） | `tests/test_assessment.py::JointAssessmentTest` |
| 结构化评估报告（评分/分析/建议） | 已实现 | `assessment/report.py`（Markdown + JSON，含证据回填） | `tests/test_assessment.py::ReportTest` |
| 实时脑电波形与频谱可视化 | 已实现 | `app/ui.py`「实时波形与频谱」页签；Web 版同功能见 `../ningsi-studio/web/`「实时监测」视图 | UI 冒烟测试 + 界面截图 |
| 状态热力图 | 已实现 | `monitoring/heatmap.py` + 桌面「状态热力图」页签 + Web `/api/sessions/{uuid}/heatmap` | `tests/test_assessment.py::HeatmapHistoryTest` |
| 历史记录查询与周/月趋势 | 已实现 | `monitoring/history.py`（含跨设备可比性保护） | `test_history_aggregation_and_comparability` |
| 状态实时预警（高压力/低专注） | 已实现 | `monitoring/alerts.py`（连续越界计时 + 伪迹窗停表） | `tests/test_assessment.py::AlertTest` |

## 二、任务清单（1）–（4）

| 任务清单条目 | 状态 | 实现与证据 |
|---|---|---|
| （1）真实脑电信号采集 | 部分实现 | LSL 源 + 仿真源；无设备时全链路仍可演示（界面与报告标注数据来源） |
| （2）信号预处理与特征提取 | 已实现 | 四级处理链 + Welch 频带特征 + 时域特征；参数逐窗留痕（`ChainLog`） |
| （3）专注力联合识别与评估 | 已实现 | SAS/SDS 量表 + SART/PVT-B + 三指标 + 联合评估 + 实时预警 |
| （4）可视化监测系统 | 已实现 | 桌面六页签界面：状态指标、实时波形与频谱、状态热力图、训练、报告、趋势；Web 版七个视图（`../ningsi-studio/`） |

## 三、赛题要求的四类核心挑战

| 挑战 | 本项目的处理方式 | 证据 |
|---|---|---|
| 真实硬件采集与信号质量保障 | 逐窗质检 + 不可用原因记录 + Run 级质量门槛；质量不合格窗不计入指标与预警计时 | `signal/artifacts.py`、`monitoring/alerts.py` |
| 特征提取与多状态识别 | 只用可解释的频带相对功率与比值；指标同时输出原始功率、基线统计量与 z 值 | `signal/indicators.py`、`assessment/report.py` |
| 专注力联合评估建模 | 三类证据独立计算后做一致性检查；不一致时明确提示复测，不强行合并成单一分数 | `assessment/joint.py`、`docs/PRODUCT_GUIDE.md` 第 5–6 节 |
| 实时性与个性化自适应 | 2 秒步进实时指标 + 相对个体基线归一化 + 训练目标随表现自适应 | `training/neurofeedback.py`、`tests/test_assessment.py::TrainingTest` |

## 四、提交材料对照

| 赛题要求的提交材料 | 状态 |
|---|---|
| （1）项目概要介绍 | 已有（`A09-凝思-项目概要介绍-最终版.docx`），口径修订见 `docs/DOC_ALIGNMENT.md` |
| （2）项目简介 PPT | 待制作 |
| （3）项目详细方案 | 已有（`A09-凝思-项目详细方案-最终版.docx`），口径修订见 `docs/DOC_ALIGNMENT.md` |
| （4）项目演示视频 | 待录制（可用 Web 版 `ningsi-studio` 与 `ningsi ui`、`ningsi demo` 作为录制脚本） |
| （5）① 产品使用说明文档（系统架构与流程说明） | 本仓库 `docs/PRODUCT_GUIDE.md` |
| （6）自愿补充材料 | `docs/REQUIREMENT_COVERAGE.md`、测试用例与运行产物 |

## 五、当前边界（主动披露，避免答辩被动）

1. 真实设备链路需要现场验证：本机未安装 `pylsl`，全部演示与测试使用仿真源；接入真实设备只需替换数据源。
2. 模型为逻辑回归基线：足以支撑"可训练、可部署、可解释"，深度模型与更大样本属下一阶段。
3. 量表只作提示：SAS/SDS 分级为常模提示，不构成诊断；`uncertain` 与缺失作答在报告中单独标注。
4. 结论的适用范围：单次会话结论只用于本次自我调节参考，趋势结论仅在设备与佩戴配置一致时可比较。
