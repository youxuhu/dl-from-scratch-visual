"""第4章的 2 层网络。改编自 oreilly-japan/deep-learning-from-scratch 的 ch04/two_layer_net.py
（MIT, Copyright (c) 2016 Koki Saitoh）。

类名 TwoLayerNet，方法 predict / loss / accuracy / numerical_gradient / gradient
与书中一致。predict 返回 softmax 概率（和书中一样）。

注意：第10讲使用的是 common/two_layer_net.py（反向传播版，执行员C 维护）。
本讲用数值梯度做演示的类是 NumericalTwoLayerNet（lecture08/numerical_two_layer_net.py），
它的 predict 返回 softmax 之前的分数，接口已和第10讲约定，不要改。
可选参数 seed 是本仓库加的，便于测试复现。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from common.functions import cross_entropy_error, sigmoid, sigmoid_grad, softmax  # noqa: E402
from common.gradient import numerical_gradient  # noqa: E402


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01, seed=None):
        rng = np.random.RandomState(seed) if seed is not None else np.random
        self.params = {
            "W1": weight_init_std * rng.randn(input_size, hidden_size),
            "b1": np.zeros(hidden_size),
            "W2": weight_init_std * rng.randn(hidden_size, output_size),
            "b2": np.zeros(output_size),
        }

    def predict(self, x):
        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        return softmax(a2)

    def loss(self, x, t):
        return cross_entropy_error(self.predict(x), t)

    def accuracy(self, x, t):
        y = np.argmax(self.predict(x), axis=1)
        t = np.asarray(t)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        return float(np.sum(y == t) / float(x.shape[0]))

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        return {
            "W1": numerical_gradient(loss_W, self.params["W1"]),
            "b1": numerical_gradient(loss_W, self.params["b1"]),
            "W2": numerical_gradient(loss_W, self.params["W2"]),
            "b2": numerical_gradient(loss_W, self.params["b2"]),
        }

    def gradient(self, x, t):
        """解析梯度。t 可以是 one hot（和书中一样），也可以是标签下标。"""
        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]
        batch_num = x.shape[0]
        t = np.asarray(t)
        if t.ndim == 1 or t.size != batch_num * W2.shape[1]:
            labels = t.reshape(batch_num)
            t_one = np.zeros((batch_num, W2.shape[1]))
            t_one[np.arange(batch_num), labels.astype(int)] = 1.0
            t = t_one

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        dy = (y - t) / batch_num
        grads = {
            "W2": np.dot(z1.T, dy),
            "b2": np.sum(dy, axis=0),
        }
        da1 = sigmoid_grad(a1) * np.dot(dy, W2.T)
        grads["W1"] = np.dot(x.T, da1)
        grads["b1"] = np.sum(da1, axis=0)
        return grads
