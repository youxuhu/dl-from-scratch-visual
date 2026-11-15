"""第6讲的 3 层推理网络（784 → 50 → 100 → 10），沉淀到 common 给后面各讲复用。

权重文件是 lecture06/weights/mnist_784_50_100_10.npz（由 lecture06/train_weights.py 训练得到）。
"""
from pathlib import Path

import numpy as np

from .functions import sigmoid, softmax

DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "lecture06" / "weights" / "mnist_784_50_100_10.npz"


def init_network(path=DEFAULT_WEIGHTS):
    """读取权重，返回 {"W1", "b1", "W2", "b2", "W3", "b3"}，都转成 float64。"""
    with np.load(path) as z:
        return {k: z[k].astype(np.float64) for k in z.files}


def predict(net, x):
    """x 可以是一张图 (784,)，也可以是一批 (N, 784)；返回 softmax 概率。"""
    z1 = sigmoid(np.dot(x, net["W1"]) + net["b1"])
    z2 = sigmoid(np.dot(z1, net["W2"]) + net["b2"])
    return softmax(np.dot(z2, net["W3"]) + net["b3"])
