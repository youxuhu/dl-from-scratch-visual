"""全连接的多层网络 MultiLayerNet，第11讲、第12讲用。

改编自《深度学习入门：基于Python的理论与实现》官方源码 common/multi_layer_net.py
（MIT 许可，Copyright (c) 2016 Koki Saitoh）。构造参数、层的命名（Affine1、Activation_function1……）
和方法名与原书一致；注释改写成中文，另加了可选的随机种子 seed。

结构：(Affine, 激活) 重复若干次，最后 Affine 接 SoftmaxWithLoss。参数键名 W1、b1、W2、b2……
"""
from collections import OrderedDict

import numpy as np

from common.gradient import numerical_gradient
from common.layers import Affine, Relu, Sigmoid, SoftmaxWithLoss

ACTIVATIONS = {"relu": Relu, "sigmoid": Sigmoid}


def init_scale(weight_init_std, fan_in):
    """权重初始值的标准差。

    'relu' 或 'he'：sqrt(2 / 前一层节点数)，适合 ReLU；
    'sigmoid' 或 'xavier'：sqrt(1 / 前一层节点数)，适合 sigmoid、tanh；
    直接给数字就用这个数字。
    """
    if isinstance(weight_init_std, str):
        key = weight_init_std.lower()
        if key in ("relu", "he"):
            return np.sqrt(2.0 / fan_in)
        if key in ("sigmoid", "xavier"):
            return np.sqrt(1.0 / fan_in)
        raise ValueError(f"不认识的初始值写法：{weight_init_std}")
    return float(weight_init_std)


class MultiLayerNet:
    """input_size → hidden_size_list 里的各隐藏层 → output_size。

    activation: 'relu' 或 'sigmoid'
    weight_init_std: 见 init_scale
    weight_decay_lambda: 权值衰减系数 λ，损失里加 0.5·λ·ΣW²
    """

    def __init__(self, input_size, hidden_size_list, output_size,
                 activation="relu", weight_init_std="relu", weight_decay_lambda=0,
                 seed=None):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = list(hidden_size_list)
        self.hidden_layer_num = len(self.hidden_size_list)
        self.weight_decay_lambda = weight_decay_lambda
        self.params = {}
        self.__init_weight(weight_init_std, np.random.default_rng(seed))

        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num + 1):
            self.layers[f"Affine{idx}"] = Affine(self.params[f"W{idx}"], self.params[f"b{idx}"])
            self.layers[f"Activation_function{idx}"] = ACTIVATIONS[activation]()
        idx = self.hidden_layer_num + 1
        self.layers[f"Affine{idx}"] = Affine(self.params[f"W{idx}"], self.params[f"b{idx}"])
        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self, weight_init_std, rng):
        sizes = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(sizes)):
            scale = init_scale(weight_init_std, sizes[idx - 1])
            self.params[f"W{idx}"] = scale * rng.standard_normal((sizes[idx - 1], sizes[idx]))
            self.params[f"b{idx}"] = np.zeros(sizes[idx])

    @property
    def depth(self):
        """Affine 层的个数。"""
        return self.hidden_layer_num + 1

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        """交叉熵误差，再加上权值衰减项 0.5·λ·ΣW²。"""
        y = self.predict(x)
        weight_decay = 0.0
        for idx in range(1, self.depth + 1):
            W = self.params[f"W{idx}"]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)
        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t):
        y = np.argmax(self.predict(x), axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        return float(np.sum(y == t) / x.shape[0])

    def numerical_gradient(self, x, t):
        """数值梯度，只用来和 gradient 对照（梯度确认），很慢。"""
        loss_W = lambda _: self.loss(x, t)  # noqa: E731
        grads = {}
        for idx in range(1, self.depth + 1):
            grads[f"W{idx}"] = numerical_gradient(loss_W, self.params[f"W{idx}"])
            grads[f"b{idx}"] = numerical_gradient(loss_W, self.params[f"b{idx}"])
        return grads

    def gradient(self, x, t):
        """反向传播求梯度：正向一次，再把层倒过来逐个 backward。"""
        self.loss(x, t)
        dout = self.last_layer.backward(1)
        for layer in reversed(list(self.layers.values())):
            dout = layer.backward(dout)
        grads = {}
        for idx in range(1, self.depth + 1):
            affine = self.layers[f"Affine{idx}"]
            grads[f"W{idx}"] = affine.dW + self.weight_decay_lambda * affine.W
            grads[f"b{idx}"] = affine.db
        return grads
