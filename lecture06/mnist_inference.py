"""用训练好的权重识别 MNIST 测试集：先一张一张地算，再 100 张一批地算。"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from common.functions import sigmoid, softmax  # noqa: E402
from common.mnist import load_mnist  # noqa: E402

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE / "weights" / "mnist_784_50_100_10.npz"


def get_data():
    _, (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network(path=WEIGHTS):
    with np.load(path) as z:
        return {k: z[k].astype(np.float64) for k in z.files}


def predict(net, x):
    """x 可以是一张图 (784,)，也可以是一批 (N, 784)。"""
    z1 = sigmoid(np.dot(x, net["W1"]) + net["b1"])
    z2 = sigmoid(np.dot(z1, net["W2"]) + net["b2"])
    return softmax(np.dot(z2, net["W3"]) + net["b3"])


def accuracy_one_by_one(net, x, t):
    hits = 0
    for i in range(len(x)):
        y = predict(net, x[i])
        hits += int(np.argmax(y) == t[i])
    return hits / len(x)


def accuracy_batched(net, x, t, batch_size=100):
    hits = 0
    for start in range(0, len(x), batch_size):
        y_batch = predict(net, x[start:start + batch_size])
        pred = np.argmax(y_batch, axis=1)
        hits += int(np.sum(pred == t[start:start + batch_size]))
    return hits / len(x)


def main():
    x, t = get_data()
    net = init_network()
    print("权重形状:", {k: v.shape for k, v in net.items()})
    y0 = predict(net, x[0])
    print("第 0 张图：标签", int(t[0]), " 预测", int(np.argmax(y0)), " 概率", np.round(y0, 3))
    y_first = predict(net, x[:100])
    print("第一批 100 张：输入", x[:100].shape, " 输出", y_first.shape,
          " 这一批对了", int(np.sum(np.argmax(y_first, axis=1) == t[:100])), "张")
    n = 1000 if os.environ.get("QUICK") == "1" else len(x)
    print(f"一张一张算（前 {n} 张）精度:", accuracy_one_by_one(net, x[:n], t[:n]))
    print("100 张一批（全部 10000 张）精度:", accuracy_batched(net, x, t))


if __name__ == "__main__":
    main()
