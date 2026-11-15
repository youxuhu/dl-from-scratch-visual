"""数值微分（一维）。改编自 oreilly-japan/deep-learning-from-scratch 的 ch04/gradient_1d.py
（MIT, Copyright (c) 2016 Koki Saitoh）。f(x) = 0.01x² + 0.1x，在 x = 5 处导数为 0.2。

numerical_diff 从 common/gradient.py 导入。图存到 output/，不弹窗口。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common.gradient import numerical_diff  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"


def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x


def tangent_line(f, x):
    d = numerical_diff(f, x)
    y = f(x) - d * x
    return lambda t: d * t + y


def main():
    x0 = 5.0
    d = numerical_diff(function_1, x0)
    print("f(x) = 0.01x² + 0.1x")
    print("f(5) =", function_1(x0), " 数值导数 =", d, " 解析导数 = 0.2")
    x = np.arange(0.0, 20.0, 0.1)
    OUT.mkdir(exist_ok=True)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.plot(x, function_1(x))
    plt.plot(x, tangent_line(function_1, x0)(x))
    path = OUT / "gradient_1d.png"
    plt.savefig(path, dpi=100)
    plt.close()
    print("图已保存:", path)


if __name__ == "__main__":
    main()
