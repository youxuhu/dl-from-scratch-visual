"""简单卷积网络 SimpleConvNet，第14讲用。

改编自《深度学习入门：基于Python的理论与实现》官方源码 ch07/simple_convnet.py
（MIT 许可，Copyright (c) 2016 Koki Saitoh）。构造参数、层的命名（Conv1、Relu1、Pool1、Affine1、Relu2、Affine2）
和方法名与原书一致；注释改写成中文，另加了可选的随机种子 seed 和各层输出形状 shapes。

结构：Conv → ReLU → Pool(2×2) → Affine → ReLU → Affine → SoftmaxWithLoss。
默认 MNIST 下各层输出形状：1×28×28 → 30×24×24 → 30×12×12 → 100 → 10。
"""
import pickle
from collections import OrderedDict

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
        self.layers = OrderedDict()
        self.layers["Conv1"] = Convolution(p["W1"], p["b1"], stride, pad)
        self.layers["Relu1"] = Relu()
        self.layers["Pool1"] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers["Affine1"] = Affine(p["W2"], p["b2"])
        self.layers["Relu2"] = Relu()
        self.layers["Affine2"] = Affine(p["W3"], p["b3"])
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
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
        for layer in reversed(list(self.layers.values())):
            g = layer.backward(g)
        L = self.layers
        return {"W1": L["Conv1"].dW, "b1": L["Conv1"].db,
                "W2": L["Affine1"].dW, "b2": L["Affine1"].db,
                "W3": L["Affine2"].dW, "b3": L["Affine2"].db}

    def save_params(self, file_name="params.pkl"):
        with open(file_name, "wb") as fh:
            pickle.dump({k: v.copy() for k, v in self.params.items()}, fh)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, "rb") as fh:
            saved = pickle.load(fh)
        for k, v in saved.items():
            self.params[k][...] = v   # 原地写入，层里引用的还是同一个数组
