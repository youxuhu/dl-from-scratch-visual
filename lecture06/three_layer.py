"""3 层神经网络的前向传播：2 输入 → 3 隐藏 → 2 隐藏 → 2 输出。权重是自己定的一组小数。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from common.functions import identity, sigmoid  # noqa: E402


def init_small_network():
    net = {}
    net["W1"] = np.array([[0.2, -0.3, 0.5], [0.4, 0.1, -0.2]])
    net["b1"] = np.array([0.1, 0.2, -0.1])
    net["W2"] = np.array([[0.3, -0.6], [-0.2, 0.4], [0.5, 0.1]])
    net["b2"] = np.array([0.0, 0.1])
    net["W3"] = np.array([[0.7, -0.5], [-0.4, 0.6]])
    net["b3"] = np.array([0.05, -0.05])
    return net


def forward(net, X):
    """返回每一层的中间结果，方便画面逐层展示。"""
    A1 = np.dot(X, net["W1"]) + net["b1"]
    Z1 = sigmoid(A1)
    A2 = np.dot(Z1, net["W2"]) + net["b2"]
    Z2 = sigmoid(A2)
    A3 = np.dot(Z2, net["W3"]) + net["b3"]
    Y = identity(A3)
    return {"A1": A1, "Z1": Z1, "A2": A2, "Z2": Z2, "A3": A3, "Y": Y}


X = np.array([1.0, -0.5])
network = init_small_network()
steps = forward(network, X)
shape_chain = [X.shape, network["W1"].shape, steps["Z1"].shape, network["W2"].shape,
               steps["Z2"].shape, network["W3"].shape, steps["Y"].shape]


def main():
    for k, v in steps.items():
        print(f"{k} =", np.round(v, 3))
    print("形状链:", " → ".join(str(s) for s in shape_chain))


if __name__ == "__main__":
    main()
