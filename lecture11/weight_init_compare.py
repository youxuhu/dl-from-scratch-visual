"""第11讲：MNIST 上比较三种权重初始值：标准差 0.01、Xavier、He。

网络 MultiLayerNet(784, [100, 100, 100, 100], 10)，激活 ReLU，SGD 学习率 0.01，批大小 128，2000 次迭代。
QUICK=1：2000 张训练图、150 次迭代。

改编自《深度学习入门：基于Python的理论与实现》官方源码 ch06/weight_init_compare.py
（MIT 许可，Copyright (c) 2016 Koki Saitoh）。变量名（weight_init_types、optimizer、networks、train_loss、
max_iterations、batch_mask）和实验设置与原书一致；注释改写成中文，另加 QUICK 模式、固定随机种子。
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # 小矩阵用多线程反而慢，机器忙时差几十倍
import numpy as np  # noqa: E402

from common.multi_layer_net import MultiLayerNet  # noqa: E402
from common.optimizer import SGD  # noqa: E402
from common.util import get_mnist, is_quick, smooth_curve  # noqa: E402

weight_init_types = {"std=0.01": 0.01, "Xavier": "sigmoid", "He": "relu"}  # 和原书相同的写法


def compare(x_train, t_train, max_iterations, batch_size=128, seed=0):
    train_size = x_train.shape[0]
    optimizer = SGD(lr=0.01)
    networks = {}
    train_loss = {}
    for key, weight_type in weight_init_types.items():
        networks[key] = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100],
                                      output_size=10, weight_init_std=weight_type, seed=seed)
        train_loss[key] = []
    rng = np.random.default_rng(seed)
    for _ in range(max_iterations):
        batch_mask = rng.choice(train_size, batch_size, replace=False)
        x_batch, t_batch = x_train[batch_mask], t_train[batch_mask]
        for key in weight_init_types:
            grads = networks[key].gradient(x_batch, t_batch)
            optimizer.update(networks[key].params, grads)
            train_loss[key].append(networks[key].loss(x_batch, t_batch))
    return train_loss, networks


def main():
    quick = is_quick()
    (x_train, t_train), (x_test, t_test), source = get_mnist(
        train_limit=2000 if quick else None, test_limit=1000 if quick else None)
    iters = 150 if quick else 2000
    losses, nets = compare(x_train, t_train, iters)
    print(f"数据：{source}，{iters} 次迭代")
    for k in weight_init_types:
        print(f"{k:>8}: 起始损失 {losses[k][0]:.4f}，平滑后最终损失 {smooth_curve(losses[k])[-1]:.4f}，"
              f"测试精度 {nets[k].accuracy(x_test, t_test):.4f}")


if __name__ == "__main__":
    main()
