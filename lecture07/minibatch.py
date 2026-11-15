"""mini-batch：从 60000 张训练图里随机抽 100 张，算这一批的平均交叉熵。

网络用第6讲训练好的权重（common/simple_net.py 的 predict）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from common.functions import cross_entropy_error  # noqa: E402
from common.mnist import load_mnist  # noqa: E402
from common.simple_net import init_network, predict  # noqa: E402

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
train_size = x_train.shape[0]
batch_size = 100

rng = np.random.default_rng(7)
batch_mask = rng.choice(train_size, batch_size, replace=False)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

net = init_network()
y_batch = predict(net, x_batch)
loss_onehot = cross_entropy_error(y_batch, t_batch)
loss_labels = cross_entropy_error(y_batch, t_batch.argmax(axis=1))


def picked_probs():
    """每一行只取正确类别那一格的概率（画面里高亮的那一格）。"""
    labels = t_batch.argmax(axis=1)
    return y_batch[np.arange(batch_size), labels]


def main():
    print("训练集", x_train.shape, " 测试集", x_test.shape, " 标签", t_train.shape)
    print("抽到的前 5 个下标:", batch_mask[:5])
    print("x_batch", x_batch.shape, " t_batch", t_batch.shape)
    p = picked_probs()
    print("前 3 行正确类别的概率:", np.round(p[:3], 4))
    print("这一批的平均交叉熵（one hot 标签）:", round(loss_onehot, 4))
    print("这一批的平均交叉熵（标签下标）:", round(loss_labels, 4))


if __name__ == "__main__":
    main()
