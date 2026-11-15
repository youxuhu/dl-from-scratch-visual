"""第10讲：用反向传播求梯度的 2 层网络 TwoLayerNet。

层按顺序放进 OrderedDict：Affine1 → Relu1 → Affine2，最后接 SoftmaxWithLoss。
gradient 正向一次，再把层倒过来逐个 backward；numerical_gradient 只用来做梯度确认。
"""
import sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from common.gradient import numerical_gradient  # noqa: E402
from common.layers import Affine, Relu, SoftmaxWithLoss  # noqa: E402


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01, seed=None):
        rng = np.random.default_rng(seed)
        self.params = {
            "W1": weight_init_std * rng.standard_normal((input_size, hidden_size)),
            "b1": np.zeros(hidden_size),
            "W2": weight_init_std * rng.standard_normal((hidden_size, output_size)),
            "b2": np.zeros(output_size),
        }
        self.layers = OrderedDict()
        self.layers["Affine1"] = Affine(self.params["W1"], self.params["b1"])
        self.layers["Relu1"] = Relu()
        self.layers["Affine2"] = Affine(self.params["W2"], self.params["b2"])
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        return self.last_layer.forward(self.predict(x), t)

    def accuracy(self, x, t):
        guess = self.predict(x).argmax(axis=1)
        truth = t.argmax(axis=1) if t.ndim != 1 else t
        return float(np.mean(guess == truth))

    def numerical_gradient(self, x, t):
        return {k: numerical_gradient(lambda _: self.loss(x, t), v) for k, v in self.params.items()}

    def gradient(self, x, t):
        self.loss(x, t)
        dout = self.last_layer.backward(1)
        for layer in reversed(list(self.layers.values())):
            dout = layer.backward(dout)
        first, second = self.layers["Affine1"], self.layers["Affine2"]
        return {"W1": first.dW, "b1": first.db, "W2": second.dW, "b2": second.db}


if __name__ == "__main__":
    net = TwoLayerNet(784, 50, 10, seed=0)
    print("层的顺序：", " → ".join(net.layers.keys()), "→ SoftmaxWithLoss")
    for k, v in net.params.items():
        print(k, v.shape)
