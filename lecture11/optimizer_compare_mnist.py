"""第11讲：在 MNIST 上比较 SGD、Momentum、AdaGrad、Adam。

网络 MultiLayerNet(784, [100, 100, 100, 100], 10)，批大小 128，2000 次迭代。
QUICK=1：只用 2000 张训练图、120 次迭代。四种方法用同一个初始权重（同一个 seed）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from common.multi_layer_net import MultiLayerNet  # noqa: E402
from common.optimizer import SGD, AdaGrad, Adam, Momentum  # noqa: E402
from common.util import get_mnist, is_quick, smooth_curve  # noqa: E402

OPTS = {
    "SGD": lambda: SGD(lr=0.01),
    "Momentum": lambda: Momentum(lr=0.01),
    "AdaGrad": lambda: AdaGrad(lr=0.01),
    "Adam": lambda: Adam(lr=0.001),
}


def compare(x_train, t_train, iters, batch_size=128, seed=0, report_every=100):
    losses = {name: [] for name in OPTS}
    nets = {name: MultiLayerNet(784, [100, 100, 100, 100], 10, seed=seed) for name in OPTS}
    opts = {name: make() for name, make in OPTS.items()}
    rng = np.random.default_rng(seed)
    for i in range(iters):
        pick = rng.choice(x_train.shape[0], batch_size, replace=False)
        xb, tb = x_train[pick], t_train[pick]
        for name in OPTS:
            grads = nets[name].gradient(xb, tb)
            opts[name].update(nets[name].params, grads)
            losses[name].append(nets[name].loss(xb, tb))
        if report_every and (i % report_every == 0 or i == iters - 1):
            row = "  ".join(f"{n} {losses[n][-1]:.4f}" for n in OPTS)
            print(f"第 {i} 次：{row}")
    return losses, nets


def main():
    quick = is_quick()
    (x_train, t_train), (x_test, t_test), source = get_mnist(
        train_limit=2000 if quick else None, test_limit=1000 if quick else None)
    iters = 120 if quick else 2000
    losses, nets = compare(x_train, t_train, iters, report_every=20 if quick else 100)
    print(f"数据：{source}，{iters} 次迭代")
    for name in OPTS:
        tail = smooth_curve(losses[name])[-1]
        acc = nets[name].accuracy(x_test, t_test)
        print(f"{name:>8}: 平滑后最终损失 {tail:.4f}，测试精度 {acc:.4f}")
    plt = __import__("_plot").get_pyplot()
    if plt is not None:
        fig, ax = plt.subplots(figsize=(7, 4))
        for name in OPTS:
            ax.plot(smooth_curve(losses[name]), label=name)
        ax.set_xlabel("iteration")
        ax.set_ylabel("loss")
        ax.legend()
        fig.savefig(__import__("_plot").OUT / "optimizer_mnist.png")
        plt.close(fig)


if __name__ == "__main__":
    main()
