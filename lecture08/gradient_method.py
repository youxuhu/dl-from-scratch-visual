"""梯度下降。改编自 oreilly-japan/deep-learning-from-scratch 的 ch04/gradient_method.py
（MIT, Copyright (c) 2016 Koki Saitoh）。起点 (-3, 4)，学习率 0.1，20 步。

gradient_descent 从 common/gradient.py 导入（路径含起点和终点，长度 step_num+1）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common.gradient import gradient_descent  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"


def function_2(x):
    return x[0] ** 2 + x[1] ** 2


def main():
    init_x = np.array([-3.0, 4.0])
    lr = 0.1
    step_num = 20
    x, path = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)
    print("起点", init_x, "lr =", lr, "步数 =", step_num)
    print("终点", x)
    print("第 1 步走到", path[1])
    OUT.mkdir(exist_ok=True)
    plt.plot([-5, 5], [0, 0], "--b")
    plt.plot([0, 0], [-5, 5], "--b")
    plt.plot(path[:, 0], path[:, 1], "o")
    plt.xlim(-3.5, 3.5)
    plt.ylim(-4.5, 4.5)
    plt.xlabel("X0")
    plt.ylabel("X1")
    path_png = OUT / "gradient_method.png"
    plt.savefig(path_png, dpi=100)
    plt.close()
    print("图已保存:", path_png)


if __name__ == "__main__":
    main()
