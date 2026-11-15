"""一个只有权重矩阵的小网络。改编自鱼书 ch04/gradient_simplenet.py
（MIT, Copyright (c) 2016 Koki Saitoh）。类名 simpleNet 与书中一致。

为了结果可复现，这里给了一个固定种子；书中源码没有设种子。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from common.functions import cross_entropy_error, softmax  # noqa: E402
from common.gradient import numerical_gradient  # noqa: E402


class simpleNet:
    def __init__(self, seed=None):
        if seed is None:
            self.W = np.random.randn(2, 3)
        else:
            self.W = np.random.default_rng(seed).standard_normal((2, 3))

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        return cross_entropy_error(y, t)


def main():
    x = np.array([0.6, 0.9])
    t = np.array([0, 0, 1])
    net = simpleNet(seed=0)
    print("W =")
    print(net.W)
    print("predict(x) =", net.predict(x))
    print("loss =", net.loss(x, t))
    f = lambda w: net.loss(x, t)
    dW = numerical_gradient(f, net.W)
    print("dW =")
    print(dW)


if __name__ == "__main__":
    main()
