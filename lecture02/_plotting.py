"""本讲脚本共用：无窗口后端 + 输出目录。"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 不弹窗口，直接存图
import matplotlib.pyplot as plt  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output"


def save(fig, name):
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path
