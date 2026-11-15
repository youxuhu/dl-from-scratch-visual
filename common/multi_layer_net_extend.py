"""带 Batch Normalization 和 Dropout 开关的多层网络 MultiLayerNetExtend（独立实现），第12讲用。

结构：每个隐藏层是 Affine →（可选 BatchNorm）→ 激活 →（可选 Dropout），
最后 Affine 接 SoftmaxWithLoss。参数键名 W1、b1、gamma1、beta1……
注意：为了和原书接口对照，构造参数沿用 dropout_ration 这个拼写。
"""
import numpy as np

from common.gradient import numerical_gradient
from common.layers import Affine, BatchNormalization, Dropout, SoftmaxWithLoss
from common.multi_layer_net import ACTIVATIONS, init_scale


class MultiLayerNetExtend:
    def __init__(self, input_size, hidden_size_list, output_size,
                 activation="relu", weight_init_std="relu", weight_decay_lambda=0,
                 use_dropout=False, dropout_ration=0.5, use_batchnorm=False, seed=None):
        self.sizes = [input_size] + list(hidden_size_list) + [output_size]
        self.depth = len(self.sizes) - 1
        self.weight_decay_lambda = weight_decay_lambda
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm
        rng = np.random.default_rng(seed)
        self.params = {}
        self.layers = []
        for k in range(1, self.depth + 1):
            fan_in, fan_out = self.sizes[k - 1], self.sizes[k]
            W = init_scale(weight_init_std, fan_in) * rng.standard_normal((fan_in, fan_out))
            self.params[f"W{k}"] = W
            self.params[f"b{k}"] = np.zeros(fan_out)
            self.layers.append(Affine(self.params[f"W{k}"], self.params[f"b{k}"]))
            if k == self.depth:
                break
            if use_batchnorm:
                self.params[f"gamma{k}"] = np.ones(fan_out)
                self.params[f"beta{k}"] = np.zeros(fan_out)
                self.layers.append(BatchNormalization(self.params[f"gamma{k}"], self.params[f"beta{k}"]))
            self.layers.append(ACTIVATIONS[activation]())
            if use_dropout:
                self.layers.append(Dropout(dropout_ration, rng=rng))
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x, train_flg=False):
        out = x
        for layer in self.layers:
            if isinstance(layer, (Dropout, BatchNormalization)):
                out = layer.forward(out, train_flg)
            else:
                out = layer.forward(out)
        return out

    def loss(self, x, t, train_flg=False):
        decay = 0.0
        if self.weight_decay_lambda:
            decay = 0.5 * self.weight_decay_lambda * sum(
                np.sum(self.params[f"W{k}"] ** 2) for k in range(1, self.depth + 1))
        return self.last_layer.forward(self.predict(x, train_flg), t) + decay

    def accuracy(self, x, t):
        guess = np.argmax(self.predict(x, train_flg=False), axis=1)
        truth = np.argmax(t, axis=1) if t.ndim != 1 else t
        return float(np.mean(guess == truth))

    def numerical_gradient(self, x, t):
        """数值梯度（训练模式下的损失）。用了 Dropout 时每次 mask 不同，不能拿来做梯度确认。"""
        return {name: numerical_gradient(lambda _: self.loss(x, t, train_flg=True), p)
                for name, p in self.params.items()}

    def gradient(self, x, t):
        self.loss(x, t, train_flg=True)
        g = self.last_layer.backward(1)
        for layer in reversed(self.layers):
            g = layer.backward(g)
        grads = {}
        k_aff = k_bn = 0
        for layer in self.layers:
            if isinstance(layer, Affine):
                k_aff += 1
                grads[f"W{k_aff}"] = layer.dW + self.weight_decay_lambda * self.params[f"W{k_aff}"]
                grads[f"b{k_aff}"] = layer.db
            elif isinstance(layer, BatchNormalization):
                k_bn += 1
                grads[f"gamma{k_bn}"] = layer.dgamma
                grads[f"beta{k_bn}"] = layer.dbeta
        return grads
