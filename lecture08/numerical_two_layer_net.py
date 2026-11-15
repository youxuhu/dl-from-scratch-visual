"""用数值梯度学习的 2 层神经网络（第8讲）。第10讲会把梯度换成反向传播，和它对比。

接口（和第10讲约定好的，不要随便改）：
    net = NumericalTwoLayerNet(input_size, hidden_size, output_size, weight_init_std=0.01, seed=None)
    net.params          # {"W1", "b1", "W2", "b2"}
    net.predict(x)      # softmax 之前的分数，形状 (N, output_size)
    net.loss(x, t)      # 平均交叉熵，t 可以是标签下标或 one hot
    net.accuracy(x, t)
    net.numerical_gradient(x, t)  # {"W1", "b1", "W2", "b2"}，和 params 同形状
"""
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

from common.functions import cross_entropy_error, sigmoid, softmax  # noqa: E402
from common.gradient import numerical_gradient as _numerical_gradient  # noqa: E402


class NumericalTwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01, seed=None):
        rng = np.random.default_rng(seed)
        self.params = {
            "W1": weight_init_std * rng.standard_normal((input_size, hidden_size)),
            "b1": np.zeros(hidden_size),
            "W2": weight_init_std * rng.standard_normal((hidden_size, output_size)),
            "b2": np.zeros(output_size),
        }

    def predict(self, x):
        p = self.params
        hidden = sigmoid(np.dot(x, p["W1"]) + p["b1"])
        return np.dot(hidden, p["W2"]) + p["b2"]

    def loss(self, x, t):
        return cross_entropy_error(softmax(self.predict(x)), t)

    def accuracy(self, x, t):
        t = np.asarray(t)
        labels = t.argmax(axis=1) if t.ndim == 2 else t
        return float(np.mean(self.predict(x).argmax(axis=1) == labels))

    def numerical_gradient(self, x, t):
        objective = lambda _: self.loss(x, t)  # noqa: E731  参数被原地修改，f 里直接重算损失
        return {k: _numerical_gradient(objective, v) for k, v in self.params.items()}
