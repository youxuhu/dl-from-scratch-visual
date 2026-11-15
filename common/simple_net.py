"""第6讲的 3 层推理网络（784 → 50 → 100 → 10），沉淀到 common 给后面各讲复用。

函数名 init_network、predict 对齐鱼书 ch03/neuralnet_mnist.py
（MIT, Copyright (c) 2016 Koki Saitoh）。权重用本仓库训练的
lecture06/weights/mnist_784_50_100_10.npz，不使用鱼书的 sample_weight.pkl。
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
