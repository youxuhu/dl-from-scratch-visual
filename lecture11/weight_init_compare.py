"""第11讲：MNIST 上比较三种权重初始值：标准差 0.01、Xavier、He。

网络 MultiLayerNet(784, [100, 100, 100, 100], 10)，激活 ReLU，SGD 学习率 0.01，批大小 128，2000 次迭代。
QUICK=1：2000 张训练图、150 次迭代。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from common.multi_layer_net import MultiLayerNet  # noqa: E402
from common.optimizer import SGD  # noqa: E402
from common.util import get_mnist, is_quick, smooth_curve  # noqa: E402

INITS = {"std=0.01": 0.01, "Xavier": "xavier", "He": "he"}


def compare(x_train, t_train, iters, batch_size=128, seed=0):
    nets = {k: MultiLayerNet(784, [100] * 4, 10, weight_init_std=v, seed=seed) for k, v in INITS.items()}
    opt = SGD(lr=0.01)
    losses = {k: [] for k in INITS}
    rng = np.random.default_rng(seed)
    for _ in range(iters):
        pick = rng.choice(x_train.shape[0], batch_size, replace=False)
        xb, tb = x_train[pick], t_train[pick]
        for k, net in nets.items():
            opt.update(net.params, net.gradient(xb, tb))
            losses[k].append(net.loss(xb, tb))
    return losses, nets


def main():
    quick = is_quick()
    (x_train, t_train), (x_test, t_test), source = get_mnist(
        train_limit=2000 if quick else None, test_limit=1000 if quick else None)
    iters = 150 if quick else 2000
    losses, nets = compare(x_train, t_train, iters)
    print(f"数据：{source}，{iters} 次迭代")
    for k in INITS:
        print(f"{k:>8}: 起始损失 {losses[k][0]:.4f}，平滑后最终损失 {smooth_curve(losses[k])[-1]:.4f}，"
              f"测试精度 {nets[k].accuracy(x_test, t_test):.4f}")


if __name__ == "__main__":
    main()
