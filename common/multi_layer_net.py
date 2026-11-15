"""全连接的多层网络 MultiLayerNet（独立实现），第11讲、第12讲用。

结构：(Affine, 激活) 重复若干次，最后 Affine 接 SoftmaxWithLoss。
参数键名 W1、b1、W2、b2……，从输入往输出数。
"""
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
        self.sizes = [input_size] + list(hidden_size_list) + [output_size]
        self.depth = len(self.sizes) - 1          # Affine 层的个数
        self.weight_decay_lambda = weight_decay_lambda
        rng = np.random.default_rng(seed)
        self.params = {}
        for k in range(1, self.depth + 1):
            fan_in, fan_out = self.sizes[k - 1], self.sizes[k]
            scale = init_scale(weight_init_std, fan_in)
            self.params[f"W{k}"] = scale * rng.standard_normal((fan_in, fan_out))
            self.params[f"b{k}"] = np.zeros(fan_out)
        self.layers = []
        for k in range(1, self.depth + 1):
            self.layers.append(Affine(self.params[f"W{k}"], self.params[f"b{k}"]))
            if k < self.depth:
                self.layers.append(ACTIVATIONS[activation]())
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def _decay_term(self):
        lam = self.weight_decay_lambda
        if not lam:
            return 0.0
        return 0.5 * lam * sum(np.sum(self.params[f"W{k}"] ** 2) for k in range(1, self.depth + 1))

    def loss(self, x, t):
        return self.last_layer.forward(self.predict(x), t) + self._decay_term()

    def accuracy(self, x, t):
        guess = np.argmax(self.predict(x), axis=1)
        truth = np.argmax(t, axis=1) if t.ndim != 1 else t
        return float(np.mean(guess == truth))

    def numerical_gradient(self, x, t):
        """数值梯度，只用来和 gradient 对照（梯度确认），很慢。"""
        return {name: numerical_gradient(lambda _: self.loss(x, t), p)
                for name, p in self.params.items()}

    def gradient(self, x, t):
        """反向传播求梯度：正向一次，再把层倒过来逐个 backward。"""
        self.loss(x, t)
        g = self.last_layer.backward(1)
        for layer in reversed(self.layers):
            g = layer.backward(g)
        grads = {}
        affines = [layer for layer in self.layers if isinstance(layer, Affine)]
        for k, layer in enumerate(affines, start=1):
            grads[f"W{k}"] = layer.dW + self.weight_decay_lambda * self.params[f"W{k}"]
            grads[f"b{k}"] = layer.db
        return grads
