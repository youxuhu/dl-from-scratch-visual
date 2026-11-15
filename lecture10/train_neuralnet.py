"""第10讲：用反向传播训练 TwoLayerNet（784 → 50 → 10）。

改编自《深度学习入门：基于Python的理论与实现》官方源码 ch05/train_neuralnet.py
（MIT 许可，Copyright (c) 2016 Koki Saitoh）。变量名（iters_num、train_size、batch_size、learning_rate、
train_loss_list、train_acc_list、test_acc_list、iter_per_epoch）和训练流程与原书一致；
注释改写成中文，另加了 QUICK 模式、固定随机种子，每批不重复抽样。

完整训练：10000 次迭代，批大小 100，学习率 0.1，每个 epoch 记录训练精度和测试精度。
QUICK=1：只用 2000 张训练图、300 次迭代，几秒钟跑完。
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # 小矩阵用多线程反而慢，机器忙时差几十倍
import numpy as np  # noqa: E402

from common.two_layer_net import TwoLayerNet  # noqa: E402
from common.util import get_mnist, is_quick  # noqa: E402


def train(iters_num, train_limit=None, test_limit=None, batch_size=100, learning_rate=0.1,
          seed=0, verbose=True):
    (x_train, t_train), (x_test, t_test), source = get_mnist(
        train_limit=train_limit, test_limit=test_limit)
    rng = np.random.default_rng(seed)
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10, seed=seed)

    train_size = x_train.shape[0]
    iter_per_epoch = max(train_size // batch_size, 1)
    train_loss_list, train_acc_list, test_acc_list = [], [], []

    for i in range(1, iters_num + 1):
        batch_mask = rng.choice(train_size, batch_size, replace=False)
        x_batch, t_batch = x_train[batch_mask], t_train[batch_mask]

        grad = network.gradient(x_batch, t_batch)  # 反向传播求梯度
        for key in ("W1", "b1", "W2", "b2"):
            network.params[key] -= learning_rate * grad[key]

        train_loss_list.append(network.loss(x_batch, t_batch))
        if i % iter_per_epoch == 0 or i == iters_num:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            if verbose:
                print(f"第 {i} 步：训练精度 {train_acc:.4f}，测试精度 {test_acc:.4f}")
    history = {"loss": train_loss_list, "train_acc": train_acc_list, "test_acc": test_acc_list}
    return network, history, source


def main():
    if is_quick():
        network, hist, source = train(300, train_limit=2000, test_limit=1000)
    else:
        network, hist, source = train(10000)
    print(f"数据：{source}；损失从 {hist['loss'][0]:.4f} 降到 {hist['loss'][-1]:.4f}")
    print(f"最终测试精度 {hist['test_acc'][-1]:.4f}")


if __name__ == "__main__":
    main()
