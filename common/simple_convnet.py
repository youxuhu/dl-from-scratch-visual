"""简单卷积网络 SimpleConvNet（独立实现），第14讲用。

结构：Conv → ReLU → Pool(2×2) → Affine → ReLU → Affine → SoftmaxWithLoss。
默认 MNIST 下各层输出形状：1×28×28 → 30×24×24 → 30×12×12 → 100 → 10。
"""
import pickle

import numpy as np

from common.gradient import numerical_gradient
from common.layers import Affine, Convolution, Pooling, Relu, SoftmaxWithLoss
from common.util import conv_output_size


class SimpleConvNet:
    def __init__(self, input_dim=(1, 28, 28),
                 conv_param=None, hidden_size=100, output_size=10,
                 weight_init_std=0.01, seed=None):
        cp = {"filter_num": 30, "filter_size": 5, "pad": 0, "stride": 1}
        cp.update(conv_param or {})
        fn, fs, pad, stride = cp["filter_num"], cp["filter_size"], cp["pad"], cp["stride"]
        ch, side = input_dim[0], input_dim[1]
        conv_side = conv_output_size(side, fs, stride, pad)
        pool_side = conv_side // 2
        flat = fn * pool_side * pool_side
        self.shapes = [input_dim, (fn, conv_side, conv_side), (fn, pool_side, pool_side),
                       (hidden_size,), (output_size,)]
        rng = np.random.default_rng(seed)
        s = weight_init_std
        self.params = {
            "W1": s * rng.standard_normal((fn, ch, fs, fs)), "b1": np.zeros(fn),
            "W2": s * rng.standard_normal((flat, hidden_size)), "b2": np.zeros(hidden_size),
            "W3": s * rng.standard_normal((hidden_size, output_size)), "b3": np.zeros(output_size),
        }
        self._build(stride, pad)

    def _build(self, stride, pad):
        p = self.params
        self.conv = Convolution(p["W1"], p["b1"], stride, pad)
        self.fc1 = Affine(p["W2"], p["b2"])
        self.fc2 = Affine(p["W3"], p["b3"])
        self.layers = [self.conv, Relu(), Pooling(2, 2, stride=2), self.fc1, Relu(), self.fc2]
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        return self.last_layer.forward(self.predict(x), t)

    def accuracy(self, x, t, batch_size=100):
        truth = np.argmax(t, axis=1) if t.ndim != 1 else t
        hits = 0
        for start in range(0, x.shape[0], batch_size):
            scores = self.predict(x[start:start + batch_size])
            hits += int(np.sum(scores.argmax(axis=1) == truth[start:start + batch_size]))
        return hits / x.shape[0]

    def numerical_gradient(self, x, t):
        return {name: numerical_gradient(lambda _: self.loss(x, t), p)
                for name, p in self.params.items()}

    def gradient(self, x, t):
        self.loss(x, t)
        g = self.last_layer.backward(1)
        for layer in reversed(self.layers):
            g = layer.backward(g)
        return {"W1": self.conv.dW, "b1": self.conv.db,
                "W2": self.fc1.dW, "b2": self.fc1.db,
                "W3": self.fc2.dW, "b3": self.fc2.db}

    def save_params(self, file_name="params.pkl"):
        with open(file_name, "wb") as fh:
            pickle.dump({k: v.copy() for k, v in self.params.items()}, fh)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, "rb") as fh:
            saved = pickle.load(fh)
        for k, v in saved.items():
            self.params[k][...] = v   # 原地写入，层里引用的还是同一个数组
