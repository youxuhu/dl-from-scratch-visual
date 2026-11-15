"""第10讲：用反向传播训练 TwoLayerNet（784 → 50 → 10）。

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


def train(iters, train_limit=None, test_limit=None, batch_size=100, lr=0.1, seed=0, verbose=True):
    (x_train, t_train), (x_test, t_test), source = get_mnist(
        train_limit=train_limit, test_limit=test_limit)
    rng = np.random.default_rng(seed)
    net = TwoLayerNet(784, 50, 10, seed=seed)
    per_epoch = max(x_train.shape[0] // batch_size, 1)
    history = {"loss": [], "train_acc": [], "test_acc": []}
    for i in range(1, iters + 1):
        pick = rng.choice(x_train.shape[0], batch_size, replace=False)
        xb, tb = x_train[pick], t_train[pick]
        grads = net.gradient(xb, tb)
        for k in net.params:
            net.params[k] -= lr * grads[k]
        history["loss"].append(net.loss(xb, tb))
        if i % per_epoch == 0 or i == iters:
            tr, te = net.accuracy(x_train, t_train), net.accuracy(x_test, t_test)
            history["train_acc"].append(tr)
            history["test_acc"].append(te)
            if verbose:
                print(f"第 {i} 步：训练精度 {tr:.4f}，测试精度 {te:.4f}")
    return net, history, source


def main():
    if is_quick():
        net, hist, source = train(300, train_limit=2000, test_limit=1000)
    else:
        net, hist, source = train(10000)
    print(f"数据：{source}；损失从 {hist['loss'][0]:.4f} 降到 {hist['loss'][-1]:.4f}")
    print(f"最终测试精度 {hist['test_acc'][-1]:.4f}")


if __name__ == "__main__":
    main()
