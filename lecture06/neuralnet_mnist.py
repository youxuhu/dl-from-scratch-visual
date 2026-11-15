"""一张一张识别 MNIST 测试集。

改编自 oreilly-japan/deep-learning-from-scratch 的 ch03/neuralnet_mnist.py
（MIT, Copyright (c) 2016 Koki Saitoh）。get_data / init_network / predict
与书中同名；权重用本仓库的 lecture06/weights/mnist_784_50_100_10.npz，
不使用鱼书的 sample_weight.pkl。

QUICK=1 时只逐张计算前 1000 张。
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from common.mnist import load_mnist  # noqa: E402
from common.simple_net import init_network, predict  # noqa: E402


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def main():
    x, t = get_data()
    network = init_network()
    accuracy_cnt = 0
    n = 1000 if os.environ.get("QUICK") == "1" else len(x)
    for i in range(n):
        y = predict(network, x[i])
        p = np.argmax(y)
        if p == t[i]:
            accuracy_cnt += 1
    print("Accuracy:" + str(float(accuracy_cnt) / n))
    if n < len(x):
        print(f"（QUICK=1，只算了前 {n} 张）")


if __name__ == "__main__":
    main()
