"""第12讲：故意制造过拟合，再用权值衰减缓解。

只用 300 个训练样本，7 层网络 784 → 100×6 → 10，SGD 学习率 0.01，批大小 100，200 个 epoch。
分别跑 λ = 0 和 λ = 0.1，对比训练精度和测试精度的差距。
QUICK=1：60 个 epoch，测试集只取 1000 张。
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # 小矩阵用多线程反而慢，机器忙时差几十倍
import numpy as np  # noqa: E402

from common.multi_layer_net import MultiLayerNet  # noqa: E402
from common.optimizer import SGD  # noqa: E402
from common.util import get_mnist, is_quick  # noqa: E402


def run(x_train, t_train, x_test, t_test, lam, epochs, batch_size=100, seed=0):
    net = MultiLayerNet(784, [100] * 6, 10, weight_decay_lambda=lam, seed=seed)
    opt = SGD(lr=0.01)
    rng = np.random.default_rng(seed)
    train_acc, test_acc = [], []
    per_epoch = max(x_train.shape[0] // batch_size, 1)
    for _ in range(epochs):
        for _ in range(per_epoch):
            pick = rng.choice(x_train.shape[0], batch_size, replace=False)
            opt.update(net.params, net.gradient(x_train[pick], t_train[pick]))
        train_acc.append(net.accuracy(x_train, t_train))
        test_acc.append(net.accuracy(x_test, t_test))
    weight_size = float(sum(np.sum(net.params[f"W{k}"] ** 2) for k in range(1, net.depth + 1)))
    return train_acc, test_acc, weight_size


def main():
    quick = is_quick()
    (x_train, t_train), (x_test, t_test), source = get_mnist(
        train_limit=300, test_limit=1000 if quick else None)
    epochs = 60 if quick else 200
    print(f"数据：{source}，训练 300 张，{epochs} 个 epoch")
    for lam in (0.0, 0.1):
        tr, te, wsize = run(x_train, t_train, x_test, t_test, lam, epochs)
        print(f"λ = {lam}：训练精度 {tr[-1]:.3f}，测试精度 {te[-1]:.3f}，差距 {tr[-1] - te[-1]:.3f}，ΣW² = {wsize:.1f}")


if __name__ == "__main__":
    main()
