"""激活函数、输出层函数和损失函数（独立实现）。

约定：
- 输入可以是 Python 数、一维数组（一个样本）或二维数组（一批样本，每行一个样本）。
- softmax 沿最后一个轴计算。
- 损失函数对一批样本返回平均值。
"""
import numpy as np


def identity(x):
    """恒等函数：原样输出，回归问题的输出层用它。"""
    return x


def step_function(x):
    """阶跃函数：大于 0 输出 1，否则输出 0。返回 int 数组。"""
    return (np.asarray(x) > 0).astype(int)


def sigmoid(x):
    """sigmoid：1 / (1 + exp(-x))。先裁剪输入，避免 exp 溢出警告。"""
    z = np.clip(x, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_grad(x):
    """sigmoid 对输入 x 的导数：s(x) * (1 - s(x))。注意参数是 x，不是 s(x)。"""
    s = sigmoid(x)
    return s * (1.0 - s)


def relu(x):
    """ReLU：小于 0 的变成 0，其余原样保留。"""
    return np.maximum(0, x)


def softmax(x):
    """softmax，沿最后一个轴。先减去每行最大值，防止 exp 溢出。"""
    z = np.asarray(x, dtype=float)
    shifted = z - z.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)


def _as_batch(y, t):
    y = np.asarray(y, dtype=float)
    t = np.asarray(t)
    if y.ndim == 1:
        y = y[np.newaxis, :]
        t = t.reshape(1, -1) if t.ndim == 1 and t.size == y.shape[1] else t.reshape(1)
    return y, t


def mean_squared_error(y, t):
    """均方误差 0.5 * Σ(y - t)²，t 是 one hot。一批样本时对样本取平均。"""
    y, t = _as_batch(y, t)
    if t.shape != y.shape:
        t = _one_hot(t, y.shape[1])
    return float(0.5 * np.sum((y - t) ** 2) / y.shape[0])


def cross_entropy_error(y, t, eps=1e-7):
    """交叉熵误差 -Σ t·log(y + eps)，对一批样本取平均。

    t 可以是 one hot（和 y 同形状），也可以是标签下标（长度等于样本数）。
    """
    y, t = _as_batch(y, t)
    n = y.shape[0]
    if t.shape == y.shape:
        labels = t.argmax(axis=1)
    else:
        labels = t.astype(int).reshape(n)
    picked = y[np.arange(n), labels]
    return float(-np.sum(np.log(picked + eps)) / n)


def _one_hot(labels, num_classes):
    labels = np.asarray(labels).astype(int).reshape(-1)
    out = np.zeros((labels.size, num_classes))
    out[np.arange(labels.size), labels] = 1.0
    return out
