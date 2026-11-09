"""全局参数与口径。所有可复算参数集中在此，界面与报告引用同一份定义。"""

VERSION = "0.1.0"
SPECTRUM_SPEC = "welch-v1"
INDICATOR_SPEC = "indicator-v1"
BASELINE_SPEC = "baseline-v1"
LABEL_SPEC = "joint-assessment-v1"

# Welch 功率谱：4 秒分析窗、2 秒分段、50% 重叠、汉宁窗（对应文档 8.3.1）
WELCH = {
    "window_sec": 4.0,
    "segment_sec": 2.0,
    "overlap": 0.5,
    "taper": "hann",
    "detrend": "mean",
    "fmax": 45.0,
    "min_segments": 1,
    "nfft": "next_pow2",   # 不小于分段点数的 2 的整数次幂（250 Hz 时 512 点）
}

# 静息基线协议（文档 7.2.4 / 8.4.1）：睁眼与闭眼各自独立质检与建基线，任务态以睁眼基线为参照
BASELINE_PROTOCOL = {
    "eyes_open_sec": 120.0,
    "eyes_closed_sec": 120.0,
    "min_windows": 5,
    "task_reference": "eyes_open",
}

# 频带定义（Hz）
BANDS = {
    "low": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
TOTAL_BAND = (0.5, 45.0)

# 指标定义：以频带功率比值刻画，再相对个体基线做 z 标准化
# 专注度 = beta/theta（β 相对增强且 θ 受抑制）；放松度 = alpha/beta；认知负荷 = theta/alpha
# 用比值而不是"相对功率之差"，避免各频带相对功率此消彼长带来的串扰。
INDICATORS = {
    "focus": ("beta", "theta"),
    "relax": ("alpha", "beta"),
    "load": ("theta", "alpha"),
}
INDEX_FLOOR = 1e-6  # 比值分母下限
SCORE_GAIN = 1.1          # 逻辑函数斜率
SCORE_CENTER = 0.0        # z=0 时评分 0.5
SCORE_MIN, SCORE_MAX = 0.0, 1.0

# 伪迹与质量门槛（文档 7.2.15）
QUALITY = {
    "amp_max_uv": 150.0,
    "flat_std_uv": 2.0,
    "span_max_uv": 100.0,
    "emg_rel_max": 0.35,      # 30-45 Hz 相对功率上限
    "eog_rel_max": 0.55,      # 0.5-4 Hz 相对功率上限（前额通道）
    "valid_ratio_min": 0.60,  # Run 级可用窗比例下限
    "consecutive_invalid_max": 3,
}

# 实时预警：阈值 + 连续越界时长 + 解除条件（文档 7.2.17）
ALERTS = {
    "low_focus": {"threshold": 0.35, "sustain_sec": 20.0, "release": 0.45, "release_sec": 10.0},
    "high_load": {"threshold": 0.75, "sustain_sec": 30.0, "release": 0.65, "release_sec": 15.0},
    "poor_signal": {"valid_ratio_min": 0.60, "sustain_sec": 10.0, "release_sec": 10.0},
}

WINDOW_SEC = 4.0
STEP_SEC = 2.0

# 训练闭环（文档 7.2.18）
TRAINING = {
    "full": {"segments": 4, "segment_sec": 120.0},
    "quick": {"segments": 2, "segment_sec": 120.0},
    "initial_from": "baseline_median",  # 初始目标取自当次基线中位数对应的评分
    "target_step": 0.05,
    "target_min": 0.40,
    "target_max": 0.90,
    "hold_sec": 6.0,             # 需连续保持的时长要求
    "volatility_high": 0.15,     # 波动幅度阈值
    "on_target_raise": 0.70,     # 达标时间占比高 → 提高要求
    "on_target_lower": 0.40,     # 达标时间占比低 → 降低要求
}

# 状态热力图分档（文档 6.6.2）
HEATMAP_BANDS = [
    (0.00, 0.20, "深蓝", "#1B3B6F"),
    (0.20, 0.40, "浅蓝", "#2E6DA4"),
    (0.40, 0.60, "灰绿", "#6E9E8A"),
    (0.60, 0.80, "橙黄", "#E69F00"),
    (0.80, 1.00, "橙红", "#D55E00"),
]

# 联合评估门槛（文档 8.5.2 一致性判定）
ASSESS = {
    "focus_low": 0.40,
    "focus_high": 0.65,
    "load_high": 0.65,
    "relax_low": 0.35,
    "scale_elevated": 50,          # SAS/SDS 标准分 ≥ 50 视为量表提示
    "rt_variability_high": 0.30,   # SART 反应时变异系数上限
    "commission_rate_high": 0.10,
    "omission_rate_high": 0.10,
    "lapse_rate_high": 0.10,
    "min_valid_ratio": 0.60,
}

# 量表口径
SAS_STANDARD_FACTOR = 1.25
SDS_STANDARD_FACTOR = 1.25
SCALE_BOUNDARY = {"normal_max": 49, "mild_max": 59, "moderate_max": 69}
