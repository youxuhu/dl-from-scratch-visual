"""用反向传播求梯度的 2 层网络 TwoLayerNet（第10讲）。

改编自《深度学习入门：基于Python的理论与实现》官方源码 ch05/two_layer_net.py
（MIT 许可，Copyright (c) 2016 Koki Saitoh）。为了让第11讲以后也能导入，放在 common/ 里；
类名、构造参数、层的命名（Affine1、Relu1、Affine2）和方法名与原书一致，注释改写成中文。

层按顺序放进 OrderedDict：Affine1 → Relu1 → Affine2，最后接 SoftmaxWithLoss。
构造参数和 params 键名与第8讲的数值梯度版 NumericalTwoLayerNet 相同，方便两者对照。
第8讲的隐藏层用 sigmoid，所以这里多一个可选参数 activation：默认 'relu'（本讲视频的结构），
和第8讲对照时传 'sigmoid'，这时两者的正向计算完全一样。
"""
from collections import OrderedDict

import numpy as np

from common.gradient import numerical_gradient
from common.layers import Affine, Relu, Sigmoid, SoftmaxWithLoss


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01, seed=None,
                 activation="relu"):
        rng = np.random.default_rng(seed)
        self.params = {
            "W1": weight_init_std * rng.standard_normal((input_size, hidden_size)),
            "b1": np.zeros(hidden_size),
            "W2": weight_init_std * rng.standard_normal((hidden_size, output_size)),
            "b2": np.zeros(output_size),
        }
        self.layers = OrderedDict()
        self.layers["Affine1"] = Affine(self.params["W1"], self.params["b1"])
        if activation == "sigmoid":
            self.layers["Sigmoid1"] = Sigmoid()
        else:
            self.layers["Relu1"] = Relu()
        self.layers["Affine2"] = Affine(self.params["W2"], self.params["b2"])
        self.last_layer = SoftmaxWithLoss()

    def load_from(self, other_params):
        """把别的网络（例如第8讲的数值梯度版）的参数原地拷进来，层里引用的数组不变。"""
        for k in self.params:
            self.params[k][...] = other_params[k]

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
