"""第11讲：权重全部相同（比如全 0 或全 0.5）为什么不行。

两个隐藏神经元拿到一模一样的权重：正向输出一样，反向梯度一样，更新后还是一样，
像一对永远分不开的双胞胎。这里训练 20 步后看两个神经元的输入权重是否仍然相同。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from common.multi_layer_net import MultiLayerNet  # noqa: E402

rng = np.random.default_rng(0)
x = rng.random((8, 3))
t = rng.integers(0, 2, size=8)


def train_twins(value, steps=20, lr=0.5):
    net = MultiLayerNet(3, [2], 2, activation="sigmoid", weight_init_std=0.0)
    net.params["W1"][...] = value
    net.params["W2"][...] = value
    for _ in range(steps):
        grads = net.gradient(x, t)
        for k in net.params:
            net.params[k] -= lr * grads[k]
    return net.params["W1"].copy()


W1_same = train_twins(0.5)


def main():
    print("初始权重全是 0.5，训练 20 步后 W1 =\n", np.round(W1_same, 6))
    print("两个隐藏神经元的输入权重（W1 的两列）是否完全相同：",
          bool(np.allclose(W1_same[:, 0], W1_same[:, 1])))


if __name__ == "__main__":
    main()
