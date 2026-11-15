"""ReLU 曲线。改编自 oreilly-japan/deep-learning-from-scratch 的 ch03/relu.py
（MIT, Copyright (c) 2016 Koki Saitoh）。函数本体在 common/functions.py。"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common.functions import relu  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"


def main():
    x = np.arange(-5.0, 5.0, 0.1)
    y = relu(x)
    OUT.mkdir(exist_ok=True)
    plt.plot(x, y)
    plt.ylim(-1.0, 5.5)
    path = OUT / "relu.png"
    plt.savefig(path, dpi=100)
    plt.close()
    print("relu(-1) =", float(relu(-1.0)), " relu(0) =", float(relu(0.0)), " relu(3) =", float(relu(3.0)))
    print("图已保存:", path)


if __name__ == "__main__":
    main()
