"""第12讲：用验证数据做超参数的随机搜索。

500 个训练样本，其中 20% 切出来当验证数据（测试数据一直不碰）。
学习率在 10^-6 到 10^-2、权值衰减在 10^-8 到 10^-4 之间按对数均匀随机采样，各试 100 组，每组 50 个 epoch；
第二轮在第一轮前 5 名的范围内再撒点。QUICK=1：每轮 8 组、每组 30 个 epoch。

改编自《深度学习入门：基于Python的理论与实现》官方源码 ch06/hyperparameter_optimization.py
（MIT 许可，Copyright (c) 2016 Koki Saitoh）。实验设置（500 个样本、验证数据 20%、100 组、每组 50 个 epoch、搜索范围）与原书一致；另加了第二轮缩小范围的搜索，注释改写成中文，保留中间结果打印和 QUICK 模式。
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # 小矩阵用多线程反而慢，机器忙时差几十倍
import numpy as np  # noqa: E402

from common.multi_layer_net import MultiLayerNet  # noqa: E402
from common.trainer import Trainer  # noqa: E402
from common.util import get_mnist, is_quick, shuffle_dataset  # noqa: E402


def split_validation(x, t, rate=0.2, seed=0):
    x, t = shuffle_dataset(x, t, rng=np.random.default_rng(seed))
    n_val = int(x.shape[0] * rate)
    return (x[n_val:], t[n_val:]), (x[:n_val], t[:n_val])


def try_one(train, val, lr, decay, epochs, seed=0):
    net = MultiLayerNet(784, [100] * 6, 10, weight_decay_lambda=decay, seed=seed)
    trainer = Trainer(net, train[0], train[1], val[0], val[1], epochs=epochs, mini_batch_size=100,
                      optimizer="SGD", optimizer_param={"lr": lr}, verbose=False, seed=seed)
    trainer.train()
    return trainer.test_acc_list[-1]


def search(train, val, trials, epochs, lr_range=(-6, -2), wd_range=(-8, -4), seed=0):
    rng = np.random.default_rng(seed)
    records = []
    for _ in range(trials):
        lr = 10 ** rng.uniform(*lr_range)
        wd = 10 ** rng.uniform(*wd_range)
        records.append((try_one(train, val, lr, wd, epochs), lr, wd))
    records.sort(key=lambda r: -r[0])
    return records


def narrowed(records, top=5):
    """用前 top 名的取值范围（对数）作为下一轮的搜索范围。"""
    best = records[:top]
    lr_log = np.log10([r[1] for r in best])
    wd_log = np.log10([r[2] for r in best])
    return (lr_log.min(), lr_log.max()), (wd_log.min(), wd_log.max())


def main():
    quick = is_quick()
    (x, t), _, source = get_mnist(train_limit=500, test_limit=10)
    train, val = split_validation(x, t)
    trials, epochs = (8, 30) if quick else (100, 50)
    print(f"数据：{source}，训练 {train[0].shape[0]} 张，验证 {val[0].shape[0]} 张")
    first = search(train, val, trials, epochs)
    print("第一轮前 5 名：")
    for acc, lr, wd in first[:5]:
        print(f"  验证精度 {acc:.3f}  学习率 {lr:.2e}  权值衰减 {wd:.2e}")
    lr_r, wd_r = narrowed(first)
    print(f"缩小范围：学习率 10^{lr_r[0]:.2f} 到 10^{lr_r[1]:.2f}，权值衰减 10^{wd_r[0]:.2f} 到 10^{wd_r[1]:.2f}")
    second = search(train, val, trials, epochs, lr_r, wd_r, seed=1)
    acc, lr, wd = second[0]
    print(f"第二轮最好：验证精度 {acc:.3f}  学习率 {lr:.2e}  权值衰减 {wd:.2e}")


if __name__ == "__main__":
    main()
