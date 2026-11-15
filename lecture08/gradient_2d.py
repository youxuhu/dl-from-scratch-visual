"""二维梯度场。改编自 oreilly-japan/deep-learning-from-scratch 的 ch04/gradient_2d.py
（MIT, Copyright (c) 2016 Koki Saitoh）。f(x0, x1) = x0² + x1²。

数值梯度用 common/gradient.py 的 numerical_gradient。图存到 output/。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common.gradient import numerical_gradient_2d as numerical_gradient  # noqa: E402
# 鱼书 ch04/gradient_2d.py 里的 numerical_gradient 是按行求梯度，
# 对应 common 里的 numerical_gradient_2d；common.numerical_gradient 要求 f 返回标量。

OUT = Path(__file__).resolve().parent / "output"


def function_2(x):
    if x.ndim == 1:
        return np.sum(x ** 2)
    return np.sum(x ** 2, axis=1)


def main():
    p = np.array([3.0, 4.0])
    print("f(x0, x1) = x0² + x1²")
    print("在 (3, 4) 处的梯度:", numerical_gradient(function_2, p.copy()))
    x0 = np.arange(-2, 2.5, 0.25)
    x1 = np.arange(-2, 2.5, 0.25)
    X, Y = np.meshgrid(x0, x1)
    grad = numerical_gradient(function_2, np.array([X.flatten(), Y.flatten()]).T).T
    OUT.mkdir(exist_ok=True)
    plt.figure()
    plt.quiver(X.flatten(), Y.flatten(), -grad[0], -grad[1], angles="xy", color="#666666")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.grid()
    path = OUT / "gradient_2d.png"
    plt.savefig(path, dpi=100)
    plt.close()
    print("图已保存:", path)


if __name__ == "__main__":
    main()
