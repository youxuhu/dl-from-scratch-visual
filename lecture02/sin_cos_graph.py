"""sin 和 cos 曲线。改编自 oreilly-japan/deep-learning-from-scratch 的 ch01/sin_cos_graph.py
（MIT, Copyright (c) 2016 Koki Saitoh）。采样为 np.arange(0, 6, 0.1)，共 60 个点。

图存到 output/，不弹窗口。本讲另外还有 sin_cos.py，用的是更疏的 np.arange(0, 7, 0.2)。
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"


def main():
    x = np.arange(0, 6, 0.1)
    y1 = np.sin(x)
    y2 = np.cos(x)
    print("x 的采样: np.arange(0, 6, 0.1)，共", len(x), "个点")
    print("x 的前 4 个值:", x[:4])
    plt.plot(x, y1, label="sin")
    plt.plot(x, y2, linestyle="--", label="cos")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("sin & cos")
    plt.legend()
    path = OUT / "sin_cos_graph.png"
    path.parent.mkdir(exist_ok=True)
    plt.savefig(path, dpi=100)
    plt.close()
    print("图已保存:", path)


if __name__ == "__main__":
    main()
