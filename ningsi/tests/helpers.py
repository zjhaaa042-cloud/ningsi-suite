"""测试公共构件：仿真源、基线、临时数据根目录。"""

from __future__ import annotations

import shutil
from pathlib import Path

from ningsi.acquisition.simulate import SyntheticEEG
from ningsi.signal.baseline import build_baseline
from ningsi.signal.window import analyze_window


def make_baseline(seed: int = 11, count: int = 8, device: str = "sim"):
    sim = SyntheticEEG(seed=seed)
    windows = [analyze_window(sim.window("rest"), sim.srate, t_end=index * 2.0) for index in range(count)]
    return sim, build_baseline(windows, device=device, srate=sim.srate)


def normal_answers(code: str) -> list:
    """构造一份"常模范围内"的作答：正向题答 1（没有或很少时间），反向题答 4。"""
    from ningsi.scales.instruments import get_scale
    scale = get_scale(code)
    return [4 if reverse else 1 for _, reverse in scale.items]


def elevated_answers(code: str, value: int = 3) -> list:
    return [value] * get_scale_size(code)


def get_scale_size(code: str) -> int:
    from ningsi.scales.instruments import get_scale
    return get_scale(code).size


def temp_root(name: str) -> Path:
    root = Path("var") / f"test_{name}"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    return root
