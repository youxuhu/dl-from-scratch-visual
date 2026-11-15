"""数值梯度（中心差分，独立实现）。"""
import numpy as np


def numerical_diff(f, x, h=1e-4):
    """一元函数在 x 处的数值导数：(f(x+h) - f(x-h)) / 2h。"""
    return (f(x + h) - f(x - h)) / (2 * h)


def numerical_gradient(f, x, h=1e-4):
    """f 对数组 x 每个元素的偏导，返回和 x 同形状的数组。

    x 可以是任意形状，必须是浮点数组。做法是原地给某一格加 h、减 h，
    各算一次 f，算完立刻把这一格改回原值。f 通常写成 lambda _: loss()，
    这样 f 内部读到的就是被原地修改过的参数。
    """
    grad = np.zeros_like(x, dtype=float)
    for idx in np.ndindex(x.shape):
        keep = x[idx]
        x[idx] = keep + h
        up = f(x)
        x[idx] = keep - h
        down = f(x)
        x[idx] = keep
        grad[idx] = (up - down) / (2 * h)
    return grad


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    """梯度下降：重复 x ← x - lr·∇f(x)。返回 (终点, 每一步位置组成的数组)。"""
    x = np.array(init_x, dtype=float)
    path = [x.copy()]
    for _ in range(step_num):
        x -= lr * numerical_gradient(f, x)
        path.append(x.copy())
    return x, np.array(path)
