"""带 Batch Normalization 和 Dropout 开关的多层网络 MultiLayerNetExtend，第12讲用。

改编自《深度学习入门：基于Python的理论与实现》官方源码 common/multi_layer_net_extend.py
（MIT 许可，Copyright (c) 2016 Koki Saitoh）。构造参数（包括原书的拼写 dropout_ration）、
层的命名（Affine1、BatchNorm1、Activation_function1、Dropout1……）和方法名与原书一致；注释改写成中文。

结构：每个隐藏层是 Affine →（可选 BatchNorm）→ 激活 →（可选 Dropout），最后 Affine 接 SoftmaxWithLoss。
参数键名 W1、b1、gamma1、beta1……
"""
from collections import OrderedDict

import numpy as np

from common.gradient import numerical_gradient
from common.layers import Affine, BatchNormalization, Dropout, SoftmaxWithLoss
from common.multi_layer_net import ACTIVATIONS, init_scale


class MultiLayerNetExtend:
    def __init__(self, input_size, hidden_size_list, output_size,
                 activation="relu", weight_init_std="relu", weight_decay_lambda=0,
                 use_dropout=False, dropout_ration=0.5, use_batchnorm=False, seed=None):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = list(hidden_size_list)
        self.hidden_layer_num = len(self.hidden_size_list)
        self.use_dropout = use_dropout
        self.weight_decay_lambda = weight_decay_lambda
        self.use_batchnorm = use_batchnorm
        rng = np.random.default_rng(seed)
        self.params = {}
        sizes = [input_size] + self.hidden_size_list + [output_size]
        for idx in range(1, len(sizes)):
            scale = init_scale(weight_init_std, sizes[idx - 1])
            self.params[f"W{idx}"] = scale * rng.standard_normal((sizes[idx - 1], sizes[idx]))
            self.params[f"b{idx}"] = np.zeros(sizes[idx])

        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num + 1):
            self.layers[f"Affine{idx}"] = Affine(self.params[f"W{idx}"], self.params[f"b{idx}"])
            if use_batchnorm:
                self.params[f"gamma{idx}"] = np.ones(sizes[idx])
                self.params[f"beta{idx}"] = np.zeros(sizes[idx])
                self.layers[f"BatchNorm{idx}"] = BatchNormalization(self.params[f"gamma{idx}"],
                                                                    self.params[f"beta{idx}"])
            self.layers[f"Activation_function{idx}"] = ACTIVATIONS[activation]()
            if use_dropout:
                self.layers[f"Dropout{idx}"] = Dropout(dropout_ration, rng=rng)
        idx = self.hidden_layer_num + 1
        self.layers[f"Affine{idx}"] = Affine(self.params[f"W{idx}"], self.params[f"b{idx}"])
        self.last_layer = SoftmaxWithLoss()

    @property
    def depth(self):
        return self.hidden_layer_num + 1

    def predict(self, x, train_flg=False):
        for key, layer in self.layers.items():
            if "Dropout" in key or "BatchNorm" in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t, train_flg=False):
        y = self.predict(x, train_flg)
        weight_decay = 0.0
        for idx in range(1, self.depth + 1):
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(self.params[f"W{idx}"] ** 2)
        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t):
        y = np.argmax(self.predict(x, train_flg=False), axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        return float(np.sum(y == t) / x.shape[0])

    def numerical_gradient(self, x, t):
        """数值梯度（训练模式下的损失）。用了 Dropout 时每次 mask 不同，不能拿来做梯度确认。"""
        loss_W = lambda _: self.loss(x, t, train_flg=True)  # noqa: E731
        return {name: numerical_gradient(loss_W, p) for name, p in self.params.items()}

    def gradient(self, x, t):
        self.loss(x, t, train_flg=True)
        dout = self.last_layer.backward(1)
        for layer in reversed(list(self.layers.values())):
            dout = layer.backward(dout)
        grads = {}
        for idx in range(1, self.depth + 1):
            affine = self.layers[f"Affine{idx}"]
            grads[f"W{idx}"] = affine.dW + self.weight_decay_lambda * self.params[f"W{idx}"]
            grads[f"b{idx}"] = affine.db
            if self.use_batchnorm and idx != self.depth:
                grads[f"gamma{idx}"] = self.layers[f"BatchNorm{idx}"].dgamma
                grads[f"beta{idx}"] = self.layers[f"BatchNorm{idx}"].dbeta
        return grads
