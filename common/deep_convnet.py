"""加深的卷积网络 DeepConvNet，第15讲用。

改编自《深度学习入门：基于Python的理论与实现》官方源码 ch08/deep_convnet.py
（MIT 许可，Copyright (c) 2016 Koki Saitoh）。构造参数 conv_param_1 到 conv_param_6、hidden_size、output_size
和方法名与原书一致；注释改写成中文，另加了可选的 conv_plan（简写通道和填充，测试里用来搭小网络）、seed 和 shapes。

默认结构：6 个 3×3 卷积层，通道数依次 16、16、32、32、64、64（第 4 个卷积层填充 2，其余填充 1），
每两个卷积层后接 2×2 Max 池化；然后 Affine(64×4×4 → 50)、ReLU、Dropout(0.5)、
Affine(50 → 10)、Dropout(0.5)、SoftmaxWithLoss。权重用 He 初始值。
"""
import pickle

import numpy as np

from common.gradient import numerical_gradient
from common.layers import Affine, Convolution, Dropout, Pooling, Relu, SoftmaxWithLoss
from common.util import conv_output_size

CONV_PLAN = [  # (输出通道, 填充)，和下面 6 个默认 conv_param 一致
    (16, 1), (16, 1), (32, 1), (32, 2), (64, 1), (64, 1),
]


def _param(filter_num, pad):
    return {"filter_num": filter_num, "filter_size": 3, "pad": pad, "stride": 1}


class DeepConvNet:
    def __init__(self, input_dim=(1, 28, 28),
                 conv_param_1=_param(16, 1), conv_param_2=_param(16, 1),
                 conv_param_3=_param(32, 1), conv_param_4=_param(32, 2),
                 conv_param_5=_param(64, 1), conv_param_6=_param(64, 1),
                 hidden_size=50, output_size=10, conv_plan=None, seed=None):
        if conv_plan is not None:
            conv_params = [_param(c, p) for c, p in conv_plan]
        else:
            conv_params = [conv_param_1, conv_param_2, conv_param_3,
                           conv_param_4, conv_param_5, conv_param_6]
        rng = np.random.default_rng(seed)
        self.params = {}
        self.layers = []
        self.trainable = []        # (编号, 层)，算梯度时按这个顺序取 dW、db
        ch, side = input_dim[0], input_dim[1]
        self.shapes = [tuple(input_dim)]
        for i, cp in enumerate(conv_params, start=1):
            fn, fs, pad, stride = cp["filter_num"], cp["filter_size"], cp["pad"], cp["stride"]
            fan_in = ch * fs * fs
            W = np.sqrt(2.0 / fan_in) * rng.standard_normal((fn, ch, fs, fs))
            self.params[f"W{i}"], self.params[f"b{i}"] = W, np.zeros(fn)
            conv = Convolution(self.params[f"W{i}"], self.params[f"b{i}"], stride, pad)
            self.layers += [conv, Relu()]
            self.trainable.append((i, conv))
            ch, side = fn, conv_output_size(side, fs, stride, pad)
            self.shapes.append((ch, side, side))
            if i % 2 == 0:
                self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
                side //= 2
                self.shapes.append((ch, side, side))
        flat = ch * side * side
        k = len(conv_params)
        for j, (n_in, n_out) in enumerate([(flat, hidden_size), (hidden_size, output_size)], start=k + 1):
            W = np.sqrt(2.0 / n_in) * rng.standard_normal((n_in, n_out))
            self.params[f"W{j}"], self.params[f"b{j}"] = W, np.zeros(n_out)
            fc = Affine(self.params[f"W{j}"], self.params[f"b{j}"])
            self.trainable.append((j, fc))
            self.layers.append(fc)
            if j == k + 1:
                self.layers.append(Relu())
            self.layers.append(Dropout(0.5, rng=rng))
            self.shapes.append((n_out,))
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x, train_flg=False):
        for layer in self.layers:
            x = layer.forward(x, train_flg) if isinstance(layer, Dropout) else layer.forward(x)
        return x

    def loss(self, x, t, train_flg=True):
        return self.last_layer.forward(self.predict(x, train_flg), t)

    def accuracy(self, x, t, batch_size=100):
        truth = np.argmax(t, axis=1) if t.ndim != 1 else t
        hits = 0
        for start in range(0, x.shape[0], batch_size):
            scores = self.predict(x[start:start + batch_size], train_flg=False)
            hits += int(np.sum(scores.argmax(axis=1) == truth[start:start + batch_size]))
        return hits / x.shape[0]

    def numerical_gradient(self, x, t):
        """数值梯度，只用于梯度确认。用推理模式的损失，所以对照时要先把 Dropout 的比例设成 0。"""
        return {k: numerical_gradient(lambda _: self.loss(x, t, train_flg=False), v)
                for k, v in self.params.items()}

    def gradient(self, x, t):
        self.loss(x, t, train_flg=True)
        g = self.last_layer.backward(1)
        for layer in reversed(self.layers):
            g = layer.backward(g)
        grads = {}
        for i, layer in self.trainable:
            grads[f"W{i}"], grads[f"b{i}"] = layer.dW, layer.db
        return grads

    def save_params(self, file_name="deep_convnet_params.pkl"):
        with open(file_name, "wb") as fh:
            pickle.dump({k: v.copy() for k, v in self.params.items()}, fh)

    def load_params(self, file_name="deep_convnet_params.pkl"):
        with open(file_name, "rb") as fh:
            saved = pickle.load(fh)
        for k, v in saved.items():
            self.params[k][...] = v
