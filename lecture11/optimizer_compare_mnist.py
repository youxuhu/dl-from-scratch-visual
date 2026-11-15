"""第11讲：在 MNIST 上比较 SGD、Momentum、AdaGrad、Adam。

网络 MultiLayerNet(784, [100, 100, 100, 100], 10)，批大小 128，2000 次迭代。
QUICK=1：只用 2000 张训练图、120 次迭代。四种方法用同一个初始权重（同一个 seed）。

改编自《深度学习入门：基于Python的理论与实现》官方源码 ch06/optimizer_compare_mnist.py
（MIT 许可，Copyright (c) 2016 Koki Saitoh）。变量名（optimizers、networks、train_loss、max_iterations、batch_mask、x_batch、t_batch）和实验设置与原书一致；四种方法共用同一初始权重、每批不重复抽样。注释改写成中文，保留中间结果打印和 QUICK 模式。
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # 小矩阵用多线程反而慢，机器忙时差几十倍
import numpy as np  # noqa: E402

from common.multi_layer_net import MultiLayerNet  # noqa: E402
from common.optimizer import SGD, AdaGrad, Adam, Momentum  # noqa: E402
from common.util import get_mnist, get_pyplot, is_quick, smooth_curve  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"

OPTIMIZERS = {  # 和原书一样都用默认学习率
    "SGD": lambda: SGD(lr=0.01),
    "Momentum": lambda: Momentum(lr=0.01),
    "AdaGrad": lambda: AdaGrad(lr=0.01),
    "Adam": lambda: Adam(lr=0.001),
}


def compare(x_train, t_train, max_iterations, batch_size=128, seed=0, report_every=100):
    train_size = x_train.shape[0]
    optimizers = {key: make() for key, make in OPTIMIZERS.items()}
    networks = {}
    train_loss = {}
    for key in optimizers:
        networks[key] = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100],
                                      output_size=10, seed=seed)
        train_loss[key] = []
    rng = np.random.default_rng(seed)
    for i in range(max_iterations):
        batch_mask = rng.choice(train_size, batch_size, replace=False)
        x_batch, t_batch = x_train[batch_mask], t_train[batch_mask]
        for key in optimizers:
            grads = networks[key].gradient(x_batch, t_batch)
            optimizers[key].update(networks[key].params, grads)
            train_loss[key].append(networks[key].loss(x_batch, t_batch))
        if report_every and (i % report_every == 0 or i == max_iterations - 1):
            row = "  ".join(f"{key} {train_loss[key][-1]:.4f}" for key in optimizers)
            print(f"第 {i} 次：{row}")
    return train_loss, networks


def main():
    quick = is_quick()
    (x_train, t_train), (x_test, t_test), source = get_mnist(
        train_limit=2000 if quick else None, test_limit=1000 if quick else None)
    iters = 120 if quick else 2000
    losses, nets = compare(x_train, t_train, iters, report_every=20 if quick else 100)
    print(f"数据：{source}，{iters} 次迭代")
    for name in OPTIMIZERS:
        tail = smooth_curve(losses[name])[-1]
        acc = nets[name].accuracy(x_test, t_test)
        print(f"{name:>8}: 平滑后最终损失 {tail:.4f}，测试精度 {acc:.4f}")
    plt = get_pyplot(OUT)
    if plt is not None:
        fig, ax = plt.subplots(figsize=(7, 4))
        for name in OPTIMIZERS:
            ax.plot(smooth_curve(losses[name]), label=name)
        ax.set_xlabel("iteration")
        ax.set_ylabel("loss")
        ax.legend()
        fig.savefig(OUT / "optimizer_mnist.png")
        plt.close(fig)


if __name__ == "__main__":
    main()
