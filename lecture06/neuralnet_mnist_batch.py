"""按 batch 识别 MNIST 测试集。

改编自 oreilly-japan/deep-learning-from-scratch 的 ch03/neuralnet_mnist_batch.py
（MIT, Copyright (c) 2016 Koki Saitoh）。权重同样使用本仓库训练好的 npz，
不使用鱼书的 sample_weight.pkl。
"""
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
    batch_size = 100
    accuracy_cnt = 0
    for i in range(0, len(x), batch_size):
        x_batch = x[i:i + batch_size]
        y_batch = predict(network, x_batch)
        p = np.argmax(y_batch, axis=1)
        accuracy_cnt += np.sum(p == t[i:i + batch_size])
    print("Accuracy:" + str(float(accuracy_cnt) / len(x)))


if __name__ == "__main__":
    main()
