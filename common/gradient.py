"""数值梯度（中心差分）。

改编自 oreilly-japan/deep-learning-from-scratch 的 common/gradient.py
（MIT, Copyright (c) 2016 Koki Saitoh）。保留 numerical_gradient 等原书接口，
并保留本仓库的 numerical_diff、gradient_descent（鱼书里写在 ch04 各脚本中）。
"""
import numpy as np


def numerical_diff(f, x, h=1e-4):
    """一元函数在 x 处的数值导数：(f(x+h) - f(x-h)) / 2h。鱼书 ch04/gradient_1d.py。"""
    return (f(x + h) - f(x - h)) / (2 * h)


def _numerical_gradient_1d(f, x, h=1e-4):
    """一维数组的中心差分。鱼书 common/gradient.py 中的同名函数。"""
    grad = np.zeros_like(x, dtype=float)
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val
    return grad


# 鱼书 gradient_2d.py 里的名字
_numerical_gradient_no_batch = _numerical_gradient_1d


def numerical_gradient_2d(f, X, h=1e-4):
    """对一维向量或逐行对二维数组求数值梯度。鱼书函数名。"""
    if X.ndim == 1:
        return _numerical_gradient_1d(f, X, h=h)
    grad = np.zeros_like(X, dtype=float)
    for idx, x in enumerate(X):
        grad[idx] = _numerical_gradient_1d(f, x, h=h)
    return grad


def numerical_gradient(f, x, h=1e-4):
    """f 对数组 x 每个元素的偏导，返回和 x 同形状的数组。

    x 可以是任意形状，必须是浮点数组。做法是原地给某一格加 h、减 h，
    各算一次 f，算完立刻把这一格改回原值。f 通常写成 lambda W: loss()，
    这样 f 内部读到的就是被原地修改过的参数。可选参数 h 默认 1e-4，和鱼书一致。
    """
    grad = np.zeros_like(x, dtype=float)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val
        it.iternext()
    return grad


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    """梯度下降：重复 x ← x - lr·∇f(x)。返回 (终点, 每一步位置组成的数组)。

    路径包含起点和每一步更新后的位置，长度为 step_num + 1。
    鱼书 ch04/gradient_method.py 的同名函数只记录更新前的位置（长度 step_num）。
    """
    x = np.array(init_x, dtype=float)
    path = [x.copy()]
    for _ in range(step_num):
        x -= lr * numerical_gradient(f, x)
        path.append(x.copy())
    return x, np.array(path)
