"""参数更新方法。

改编自《深度学习入门：基于Python的理论与实现》官方源码 common/optimizer.py
（MIT 许可，Copyright (c) 2016 Koki Saitoh）。类名、参数名和更新公式与原书一致，注释改写成中文。

统一接口：opt.update(params, grads)，params 和 grads 是同样键名的字典，
update 原地修改 params 里的数组。
"""
import numpy as np


class SGD:
    """随机梯度下降：W ← W - lr * dW。"""

    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for name, g in grads.items():
            params[name] -= self.lr * g


class Momentum:
    """动量法：速度 v ← momentum * v - lr * dW，然后 W ← W + v。像小球在碗里滚。"""

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {name: np.zeros_like(p) for name, p in params.items()}
        for name, g in grads.items():
            self.v[name] = self.momentum * self.v[name] - self.lr * g
            params[name] += self.v[name]


class Nesterov:
    """Nesterov 加速梯度（写法与原书相同）：先按动量往前看一步，再用梯度修正。"""

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {name: np.zeros_like(p) for name, p in params.items()}
        mu = self.momentum
        for name, g in grads.items():
            self.v[name] *= mu
            self.v[name] -= self.lr * g
            params[name] += mu * mu * self.v[name]
            params[name] -= (1 + mu) * self.lr * g


class AdaGrad:
    """AdaGrad：h 累加梯度的平方，每个参数的步长按 1/sqrt(h) 自动变小。"""

    def __init__(self, lr=0.01, eps=1e-7):
        self.lr = lr
        self.eps = eps
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {name: np.zeros_like(p) for name, p in params.items()}
        for name, g in grads.items():
            self.h[name] += g * g
            params[name] -= self.lr * g / (np.sqrt(self.h[name]) + self.eps)


class RMSprop:
    """RMSprop：和 AdaGrad 类似，但 h 是梯度平方的指数滑动平均，旧的梯度会被逐渐忘掉。"""

    def __init__(self, lr=0.01, decay_rate=0.99, eps=1e-7):
        self.lr = lr
        self.decay_rate = decay_rate
        self.eps = eps
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {name: np.zeros_like(p) for name, p in params.items()}
        d = self.decay_rate
        for name, g in grads.items():
            self.h[name] = d * self.h[name] + (1 - d) * g * g
            params[name] -= self.lr * g / (np.sqrt(self.h[name]) + self.eps)


class Adam:
    """Adam：一阶矩 m 像 Momentum，二阶矩 v 像 AdaGrad，偏置修正合并进学习率 lr_t（与原书写法相同）。"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m = {name: np.zeros_like(p) for name, p in params.items()}
            self.v = {name: np.zeros_like(p) for name, p in params.items()}
        self.iter += 1
        b1, b2 = self.beta1, self.beta2
        # 偏置修正：前几步 m、v 偏向 0，用 sqrt(1 - b2^t) / (1 - b1^t) 放大学习率补回来
        lr_t = self.lr * np.sqrt(1.0 - b2 ** self.iter) / (1.0 - b1 ** self.iter)
        for name, g in grads.items():
            self.m[name] += (1 - b1) * (g - self.m[name])
            self.v[name] += (1 - b2) * (g ** 2 - self.v[name])
            params[name] -= lr_t * self.m[name] / (np.sqrt(self.v[name]) + 1e-7)


OPTIMIZERS = {
    "sgd": SGD, "momentum": Momentum, "nesterov": Nesterov,
    "adagrad": AdaGrad, "rmsprop": RMSprop, "adam": Adam,
}


def make_optimizer(name, **kwargs):
    """按名字（不分大小写）创建优化器，例如 make_optimizer('Adam', lr=0.001)。"""
    return OPTIMIZERS[name.lower()](**kwargs)
