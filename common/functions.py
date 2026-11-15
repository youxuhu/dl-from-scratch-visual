"""激活函数、输出层函数和损失函数。

改编自 oreilly-japan/deep-learning-from-scratch 的 common/functions.py
（MIT, Copyright (c) 2016 Koki Saitoh）。保留原书函数名和签名，并加上
中文注释、输入裁剪，以及向后兼容的别名。

约定：
- 输入可以是 Python 数、一维数组（一个样本）或二维数组（一批样本，每行一个样本）。
- softmax 沿最后一个轴计算。
- cross_entropy_error 对一批样本返回平均值。
"""
import numpy as np


def identity_function(x):
    """恒等函数：原样输出，回归问题的输出层用它。鱼书函数名。"""
    return x


identity = identity_function  # 向后兼容别名


def step_function(x):
    """阶跃函数：大于 0 输出 1，否则输出 0。返回 int 数组。"""
    return np.array(x > 0, dtype=int)


def sigmoid(x):
    """sigmoid：1 / (1 + exp(-x))。先裁剪输入，避免 exp 溢出警告。"""
    z = np.clip(x, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_grad(x):
    """sigmoid 对输入 x 的导数：s(x) * (1 - s(x))。注意参数是 x，不是 s(x)。"""
    return (1.0 - sigmoid(x)) * sigmoid(x)


def relu(x):
    """ReLU：小于 0 的变成 0，其余原样保留。"""
    return np.maximum(0, x)


def relu_grad(x):
    """ReLU 对输入 x 的导数：x >= 0 处为 1，否则为 0。"""
    grad = np.zeros_like(x)
    grad[x >= 0] = 1
    return grad


def softmax(x):
    """softmax，沿最后一个轴。先减去每行最大值，防止 exp 溢出。"""
    x = np.asarray(x, dtype=float)
    x = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


def sum_squared_error(y, t):
    """平方和误差 0.5 * Σ(y - t)²。鱼书函数名，不对 batch 取平均。"""
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    return 0.5 * np.sum((y - t) ** 2)


def _as_batch(y, t):
    y = np.asarray(y, dtype=float)
    t = np.asarray(t)
    if y.ndim == 1:
        y = y[np.newaxis, :]
        t = t.reshape(1, -1) if t.ndim == 1 and t.size == y.shape[1] else t.reshape(1)
    return y, t


def mean_squared_error(y, t):
    """均方误差 0.5 * Σ(y - t)² / N，一批样本时对样本取平均。

    这是本仓库的兼容接口；鱼书对应的是不对 batch 取平均的 sum_squared_error。
    t 可以是 one hot，也可以是标签下标。
    """
    y, t = _as_batch(y, t)
    if t.shape != y.shape:
        t = _one_hot(t, y.shape[1])
    return float(0.5 * np.sum((y - t) ** 2) / y.shape[0])


def cross_entropy_error(y, t, eps=1e-7):
    """交叉熵误差 -Σ t·log(y + eps)，对一批样本取平均。

    t 可以是 one hot（和 y 同形状），也可以是标签下标（长度等于样本数）。
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(t)
    if y.ndim == 1:
        t = t.reshape(1, t.size) if np.asarray(t).ndim != 0 else np.array([t])
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    t = t.astype(int).reshape(batch_size)
    return float(-np.sum(np.log(y[np.arange(batch_size), t] + eps)) / batch_size)


def softmax_loss(X, t):
    """先 softmax 再交叉熵，鱼书 common/functions.py 中的组合函数。"""
    y = softmax(X)
    return cross_entropy_error(y, t)


def _one_hot(labels, num_classes):
    labels = np.asarray(labels).astype(int).reshape(-1)
    out = np.zeros((labels.size, num_classes))
    out[np.arange(labels.size), labels] = 1.0
    return out
