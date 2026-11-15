"""用训练好的权重识别 MNIST 测试集：先一张一张地算，再 100 张一批地算。

init_network 和 predict 是本讲讲解的内容，因为第7讲还要用，代码放在 common/simple_net.py。
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from common.mnist import load_mnist  # noqa: E402
from common.simple_net import init_network, predict  # noqa: E402  第6讲讲的网络，代码放在 common 里


def get_data():
    _, (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


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
