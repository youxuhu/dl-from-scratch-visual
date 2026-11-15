"""用数值梯度真的训练一次 NumericalTwoLayerNet。

数值梯度非常慢：每个参数都要算两次损失。所以这里把 MNIST 缩小成 7×7（每 4×4 个像素取平均），
网络是 49 → 20 → 10，一共 1210 个参数；训练集用前 10000 张，测试集用全部 10000 张。
另外实测「原尺寸 784 → 50 → 10 的网络算一次数值梯度要多久」，作为下一讲反向传播的伏笔。

QUICK=1：只训练 20 步，不计时原尺寸网络的完整梯度。
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from common.mnist import load_mnist  # noqa: E402
from lecture08.numerical_two_layer_net import NumericalTwoLayerNet  # noqa: E402

HERE = Path(__file__).resolve().parent
QUICK = os.environ.get("QUICK") == "1"

TRAIN_N = 10000
BATCH = 100
HIDDEN = 20
LR = 2.0
ITERS = 20 if QUICK else 1500
ITER_PER_EPOCH = TRAIN_N // BATCH       # 100 步算一个 epoch


def shrink_to_7x7(x):
    """(N, 784) → (N, 49)：28×28 的图按 4×4 的小块取平均。"""
    return x.reshape(-1, 7, 4, 7, 4).mean(axis=(2, 4)).reshape(-1, 49)


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True)
    return shrink_to_7x7(x_train[:TRAIN_N]), t_train[:TRAIN_N], shrink_to_7x7(x_test), t_test


def train(iters=ITERS, seed=8):
    x_train, t_train, x_test, t_test = get_data()
    net = NumericalTwoLayerNet(49, HIDDEN, 10, weight_init_std=0.1, seed=seed)
    rng = np.random.default_rng(seed)
    log = {"loss": [], "epoch": [], "train_acc": [], "test_acc": []}
    start = time.time()
    for i in range(1, iters + 1):
        pick = rng.choice(TRAIN_N, BATCH, replace=False)
        grads = net.numerical_gradient(x_train[pick], t_train[pick])
        for k in net.params:
            net.params[k] -= LR * grads[k]
        log["loss"].append(net.loss(x_train[pick], t_train[pick]))
        if i % ITER_PER_EPOCH == 0:
            log["epoch"].append(i // ITER_PER_EPOCH)
            log["train_acc"].append(net.accuracy(x_train, t_train))
            log["test_acc"].append(net.accuracy(x_test, t_test))
    log["seconds"] = time.time() - start
    log["final_train_acc"] = net.accuracy(x_train, t_train)
    log["final_test_acc"] = net.accuracy(x_test, t_test)
    return net, log


def time_full_size_gradient(sample_params=2000):
    """784 → 50 → 10 的网络有 39760 个参数。实测前 sample_params 个参数的数值梯度耗时，再按比例估算整次。"""
    from common.functions import cross_entropy_error, sigmoid, softmax
    (x_train, t_train), _ = load_mnist(normalize=True)
    net = NumericalTwoLayerNet(784, 50, 10, seed=0)
    x, t = x_train[:BATCH], t_train[:BATCH]
    W1 = net.params["W1"]
    n_total = sum(v.size for v in net.params.values())
    flat = W1.reshape(-1)
    start = time.time()
    for k in range(sample_params):
        keep = flat[k]
        flat[k] = keep + 1e-4
        net.loss(x, t)
        flat[k] = keep - 1e-4
        net.loss(x, t)
        flat[k] = keep
    per_param = (time.time() - start) / sample_params
    return n_total, per_param * n_total


def main():
    net, log = train()
    print(f"训练 {ITERS} 步（49 → {HIDDEN} → 10，batch {BATCH}，lr {LR}），用时 {log['seconds']:.1f} 秒，"
          f"平均每步 {log['seconds'] / ITERS:.3f} 秒")
    print("前 5 步 loss:", np.round(log["loss"][:5], 4))
    print("最后 5 步 loss:", np.round(log["loss"][-5:], 4))
    for e, tr, te in zip(log["epoch"], log["train_acc"], log["test_acc"]):
        print(f"epoch {e:2d}: 训练精度 {tr:.4f}  测试精度 {te:.4f}")
    print("最终：训练精度", round(log["final_train_acc"], 4), " 测试精度", round(log["final_test_acc"], 4))
    out = HERE / "output"
    out.mkdir(exist_ok=True)
    name = "train_log_quick.json" if QUICK else "train_log.json"
    (out / name).write_text(json.dumps(log, ensure_ascii=False, indent=1))
    print("训练记录已保存:", out / name)
    if not QUICK:
        n_total, est = time_full_size_gradient()
        print(f"原尺寸 784 → 50 → 10 有 {n_total} 个参数，算一次数值梯度估计要 {est:.1f} 秒")


if __name__ == "__main__":
    main()
