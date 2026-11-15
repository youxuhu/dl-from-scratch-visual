"""用解析梯度训练 TwoLayerNet（784 → 50 → 10）。

改编自 oreilly-japan/deep-learning-from-scratch 的 ch04/train_neuralnet.py
（MIT, Copyright (c) 2016 Koki Saitoh）。完整训练 10000 步；QUICK=1 时只跑 20 步。
图存到 output/，不覆盖第8讲用数值梯度训练的 results/train_log.json。
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common.mnist import load_mnist  # noqa: E402
from lecture08.two_layer_net import TwoLayerNet  # noqa: E402

HERE = Path(__file__).resolve().parent
QUICK = os.environ.get("QUICK") == "1"


def main():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10, seed=0)
    iters_num = 20 if QUICK else 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1
    iter_per_epoch = max(int(train_size / batch_size), 1)

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    rng = np.random.default_rng(0)

    for i in range(iters_num):
        batch_mask = rng.choice(train_size, batch_size, replace=False)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        grad = network.gradient(x_batch, t_batch)
        for key in ("W1", "b1", "W2", "b2"):
            network.params[key] -= learning_rate * grad[key]
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)
        # 完整训练时每个 epoch 评估一次；QUICK 时最后一步评估一次
        if (not QUICK and i % iter_per_epoch == 0) or (QUICK and i == iters_num - 1):
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

    print("最后一步 loss:", round(float(train_loss_list[-1]), 4))
    out = HERE / "output"
    out.mkdir(exist_ok=True)
    if train_acc_list:
        plt.plot(train_acc_list, label="train acc")
        plt.plot(test_acc_list, label="test acc", linestyle="--")
        plt.xlabel("epochs" if not QUICK else "eval")
        plt.ylabel("accuracy")
        plt.ylim(0, 1.0)
        plt.legend(loc="lower right")
        plt.savefig(out / "train_neuralnet.png", dpi=100)
        plt.close()
        print("图已保存:", out / "train_neuralnet.png")


if __name__ == "__main__":
    main()
