"""参数更新方法（独立实现）。

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
    """Nesterov 加速梯度：先按速度往前看一步，再用那里的梯度修正（这里用等价的改写式）。"""

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {name: np.zeros_like(p) for name, p in params.items()}
        mu = self.momentum
        for name, g in grads.items():
            old_v = self.v[name].copy()
            self.v[name] = mu * old_v - self.lr * g
            params[name] += -mu * old_v + (1 + mu) * self.v[name]


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
    """Adam：一阶矩 m 像 Momentum，二阶矩 v 像 AdaGrad，再做偏置修正。"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-7):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.step = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m = {name: np.zeros_like(p) for name, p in params.items()}
            self.v = {name: np.zeros_like(p) for name, p in params.items()}
        self.step += 1
        b1, b2 = self.beta1, self.beta2
        fix1 = 1 - b1 ** self.step
        fix2 = 1 - b2 ** self.step
        for name, g in grads.items():
            self.m[name] = b1 * self.m[name] + (1 - b1) * g
            self.v[name] = b2 * self.v[name] + (1 - b2) * g * g
            m_hat = self.m[name] / fix1
            v_hat = self.v[name] / fix2
            params[name] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


OPTIMIZERS = {
    "sgd": SGD, "momentum": Momentum, "nesterov": Nesterov,
    "adagrad": AdaGrad, "rmsprop": RMSprop, "adam": Adam,
}


def make_optimizer(name, **kwargs):
    """按名字（不分大小写）创建优化器，例如 make_optimizer('Adam', lr=0.001)。"""
    return OPTIMIZERS[name.lower()](**kwargs)
