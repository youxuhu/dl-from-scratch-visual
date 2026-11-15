"""加深的卷积网络 DeepConvNet（独立实现），第15讲用。

6 个 3×3 卷积层，通道数依次 16、16、32、32、64、64（第 4 个卷积层填充 2，其余填充 1），
每两个卷积层后接 2×2 Max 池化；然后 Affine(64×4×4 → 50)、ReLU、Dropout(0.5)、
Affine(50 → 10)、Dropout(0.5)、SoftmaxWithLoss。权重用 He 初始值。
"""
import pickle

import numpy as np

from common.gradient import numerical_gradient
from common.layers import Affine, Convolution, Dropout, Pooling, Relu, SoftmaxWithLoss
from common.util import conv_output_size

CONV_PLAN = [  # (输出通道, 填充)
    (16, 1), (16, 1), (32, 1), (32, 2), (64, 1), (64, 1),
]


class DeepConvNet:
    def __init__(self, input_dim=(1, 28, 28), conv_plan=None,
                 hidden_size=50, output_size=10, seed=None):
        plan = conv_plan or CONV_PLAN
        rng = np.random.default_rng(seed)
        self.params = {}
        self.layers = []
        self.trainable = []        # (编号, 层)，算梯度时按这个顺序取 dW、db
        ch, side = input_dim[0], input_dim[1]
        self.shapes = [tuple(input_dim)]
        for i, (out_ch, pad) in enumerate(plan, start=1):
            fan_in = ch * 3 * 3
            W = np.sqrt(2.0 / fan_in) * rng.standard_normal((out_ch, ch, 3, 3))
            self.params[f"W{i}"], self.params[f"b{i}"] = W, np.zeros(out_ch)
            conv = Convolution(self.params[f"W{i}"], self.params[f"b{i}"], 1, pad)
            self.layers += [conv, Relu()]
            self.trainable.append((i, conv))
            ch, side = out_ch, conv_output_size(side, 3, 1, pad)
            self.shapes.append((ch, side, side))
            if i % 2 == 0:
                self.layers.append(Pooling(2, 2, stride=2))
                side //= 2
                self.shapes.append((ch, side, side))
        flat = ch * side * side
        k = len(plan)
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
