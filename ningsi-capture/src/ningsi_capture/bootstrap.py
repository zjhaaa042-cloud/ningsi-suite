"""Locate the adjacent algorithm engine without importing the product service."""

from pathlib import Path
import sys


def ensure_engine():
    candidate = Path(__file__).resolve().parents[3] / "ningsi" / "src"
    if not getattr(sys, "frozen", False) and (candidate / "ningsi" / "__init__.py").is_file():
        sys.path.insert(0, str(candidate)) if str(candidate) not in sys.path else None
    try:
        import ningsi  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("未找到凝思算法引擎，请保留同级 ningsi 目录或安装 ningsi 包。") from exc
