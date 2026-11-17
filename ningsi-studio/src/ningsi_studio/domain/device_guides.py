"""Reviewed manual facts, separate from source identity and operator declarations.

This catalogue neither reads the private PDF at runtime nor changes acquisition,
unit conversion, channel labels, quality gates or source-scoped saved profiles.
Pages below always refer to PDF page numbers, including the cover.
"""

from copy import deepcopy


_BSENSE_R = {
    "schema": "ningsi-device-guide-v1",
    "id": "bsense-r",
    "title": "BSense-R 手册参考",
    "source": {
        "filename": "BSense-R 产品手册.pdf",
        "sha256": "f83664699b5986b012ea4f2b79eed6837f577dab793373cec54cf3b9ef3e7635",
        "page_count": 25,
        "revision": "未标明",
    },
    "overview": (
        "依据用户提供的产品手册整理。规格与操作说明不代表当前硬件、佩戴、导联、"
        "LSL 数值单位或信号质量已核实；BioMulti Lite 流名也不能单独证明硬件型号。"
        "以下页码均为 PDF 页码（含封面）。"
    ),
    "specs": [
        {"label": "手册所列 LSL EEG", "value": "2 通道 · 250 Hz · cf_float32",
         "note": "表列该通道配置，不表示两条独立 EEG 流；以实际流元数据核对。Metric 等 250 Hz 流不能当作原始 EEG。", "pages": [23]},
        {"label": "厂商 CSV 的 EEG 列", "value": "EEG.FP1 / EEG.FP2，单位 µV",
         "note": "列名说明仅约束该 CSV 格式，不自动证明当前 LSL 的电极顺序或数值单位。", "pages": [21]},
        {"label": "封面品牌", "value": "荣脑科技 · LOONG BRAIN",
         "note": "封面标识，不作为法人厂家全称或当前设备身份的确认。", "pages": [1]},
        {"label": "产品概述", "value": "EEG、fNIRS、PPG 与 Motion 多模态系统",
         "note": "简介称支持 2–4 个 EEG 通道；当前连接的实际通道数仍以实时元数据与硬件配置为准。", "pages": [2]},
        {"label": "手册所列 LSL Motion", "value": "6 通道 · 12.5 Hz",
         "note": "运动流独立于 EEG；没有同步采集时不能声称已用于纠伪。", "pages": [23]},
    ],
    "steps": [
        {"title": "先建立厂商设备连接",
         "text": "在 BioMultiLite 点击 Connect Device，选择 Biox_BO 开头的设备编号；核对 Connected 提示。",
         "pages": [4, 5]},
        {"title": "检查实际 EEG 波形与接触",
         "text": "打开 EEG 的 Time Series 和 Spectrum，按厂商说明检查接触与原始信号；蓝牙已连接不等于 EEG 合格。",
         "pages": [7, 12]},
        {"title": "单独开启 EEG 推流",
         "text": "进入 BioMultiLite 的 LSL，勾选 EEG 后点击 Start；在凝思重新扫描并选择原始 EEG 流，核对实际通道、采样率与单位。",
         "pages": [23, 24]},
        {"title": "另行启动录制并确认落盘",
         "text": "开启 LSL 传输不自动保存 BioMultiLite 本地数据。使用凝思采集器或独立 LabRecorder 录制；LabRecorder 需 Update、选流、Start，结束后 Stop 保存并核查文件。",
         "pages": [24, 25]},
    ],
    "quality_notes": [
        {"title": "蓝牙颜色与接触颜色分别判断",
         "text": "标题栏笑脸绿/红表示蓝牙连接优/弱；Time Series 的红/橙/绿表示接触不良/信号合格/优异，不能混为同一项质量检查。",
         "pages": [7, 12]},
        {"title": "厂商 Filters 改变显示，不证明已纠净原始数据",
         "text": "手册说明 Filters 优化实时数据显示，存储和导出仍为原始未滤波数据；还需检查原始工频、突跳及接触。手册未据此说明 LSL 输出经过同样滤波。",
         "pages": [7]},
        {"title": "厂商 CSV 不是连续 EEG 矩阵",
         "text": "TIMESTAMP 为毫秒，IS_LOSS=1 表示丢失；EEG、fNIRS、Motion 按时间交错，其他模态列会为空。解析时须识别模态与丢失，不能将空列补成零脑电。",
         "pages": [21, 22]},
    ],
    "unknowns": [
        {"label": "实际佩戴与电极位置", "text": "手册未给出佩戴位置图或当前连接的 EEG 电极、左右顺序；须按实际安装和厂商确认记录。"},
        {"label": "参考、地与耳夹映射", "text": "手册未说明参考/地电极位置或左右耳夹映射，不据产品名推定。"},
        {"label": "当前 LSL 数值单位与 ADC 标定", "text": "CSV 的 µV 说明不等于当前 LSL 单位声明；缺失时仍需本次明确确认，ADC 须有标定倍率。"},
        {"label": "实际时标与信号质量", "text": "手册同步能力介绍不代替当前机器的时间戳核查、掉包检查、基线和逐窗质检。"},
    ],
    "declaration_seed": {
        "model": "BSense-R",
        "document_reference": (
            "BSense-R 产品手册.pdf；25页，修订号未标明；"
            "SHA-256: f83664699b5986b012ea4f2b79eed6837f577dab793373cec54cf3b9ef3e7635；"
            "所引页码为PDF页码（含封面）。"
        ),
        "notes": (
            "使用 BSense-R 手册资料补充草稿，仍须确认当前硬件型号。手册 LSL 表列 EEG 2通道/250 Hz/cf_float32；"
            "CSV EEG.FP1/EEG.FP2 列为 µV，但不证明当前 LSL 的通道顺序、单位或实际佩戴。"
            "实际电极、参考/地、采样率及输入单位/倍率请按当前设备和软件配置核对后填写；"
            "蓝牙连接与电极接触质量分开检查。"
        ),
    },
}


def device_guides():
    """Return detached display data; callers cannot mutate the reviewed catalogue."""
    return [deepcopy(_BSENSE_R)]
