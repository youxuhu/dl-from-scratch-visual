"""第12讲：有无 Batch Normalization，在 16 种权重初始值标准差下的学习情况。

1000 个训练样本，20 个 epoch，网络 784 → 100×5 → 10，SGD 学习率 0.01，批大小 100，
标准差取 np.logspace(0, -4, num=16)，对应视频里 4×4 的小倍数图。
QUICK=1：只取其中 4 个标准差、300 个样本、4 个 epoch。
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # 小矩阵用多线程反而慢，机器忙时差几十倍
import numpy as np  # noqa: E402

from common.multi_layer_net_extend import MultiLayerNetExtend  # noqa: E402
from common.optimizer import SGD  # noqa: E402
from common.util import get_mnist, is_quick  # noqa: E402


def run(x, t, std, epochs, batch_size=100, seed=0):
    """同一个初始化种子，分别训练带 BN 和不带 BN 的网络，返回两条每个 epoch 的训练精度曲线。"""
    nets = {
        "BN": MultiLayerNetExtend(784, [100] * 5, 10, weight_init_std=std, use_batchnorm=True, seed=seed),
        "无BN": MultiLayerNetExtend(784, [100] * 5, 10, weight_init_std=std, seed=seed),
    }
    opt = SGD(lr=0.01)
    curves = {k: [] for k in nets}
    rng = np.random.default_rng(seed)
    per_epoch = max(x.shape[0] // batch_size, 1)
    for _ in range(epochs):
        for _ in range(per_epoch):
            pick = rng.choice(x.shape[0], batch_size, replace=False)
            for net in nets.values():
                opt.update(net.params, net.gradient(x[pick], t[pick]))
        for k, net in nets.items():
            curves[k].append(net.accuracy(x, t))
    return curves


def main():
    quick = is_quick()
    stds = np.logspace(0, -4, num=16)
    if quick:
        stds = stds[[0, 5, 10, 15]]
    n = 300 if quick else 1000
    (x, t), _, source = get_mnist(train_limit=n, test_limit=10)
    epochs = 4 if quick else 20
    print(f"数据：{source} 前 {n} 张，{epochs} 个 epoch")
    for i, std in enumerate(stds, start=1):
        # 标准差 1 且不带 BN 时激活值会溢出成 nan，这正是要展示的「学不起来」，所以不打印警告
        with np.errstate(over="ignore", invalid="ignore"):
            c = run(x, t, std, epochs)
        print(f"{i:2d}/{len(stds)} 标准差 {std:.2e}：BN 最终训练精度 {c['BN'][-1]:.3f}，无BN {c['无BN'][-1]:.3f}")


if __name__ == "__main__":
    main()
