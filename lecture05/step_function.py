"""阶跃函数曲线。改编自 oreilly-japan/deep-learning-from-scratch 的 ch03/step_function.py
（MIT, Copyright (c) 2016 Koki Saitoh）。函数本体在 common/functions.py。"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common.functions import step_function  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"


def main():
    x = np.arange(-5.0, 5.0, 0.1)
    y = step_function(x)
    OUT.mkdir(exist_ok=True)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    path = OUT / "step_function.png"
    plt.savefig(path, dpi=100)
    plt.close()
    print("采样: np.arange(-5.0, 5.0, 0.1)，共", len(x), "个点")
    print("图已保存:", path)


if __name__ == "__main__":
    main()
