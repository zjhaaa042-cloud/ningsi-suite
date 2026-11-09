"""测试包：把 src 加入导入路径，便于直接 python -m unittest discover -s tests。"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
