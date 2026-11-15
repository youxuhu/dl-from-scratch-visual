"""第10讲：梯度确认。同一批数据，数值梯度和反向传播梯度逐格做差，看平均绝对误差。

同时计时，对比两种方法求一次梯度的耗时。每次运行随机初始化不同，误差在 1e-10 到 1e-7 的量级。
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from common.util import get_mnist  # noqa: E402
from two_layer_net_bp import TwoLayerNet  # noqa: E402


def check(x_batch, t_batch, seed=None):
    net = TwoLayerNet(784, 50, 10, seed=seed)
    start = time.perf_counter()
    numeric = net.numerical_gradient(x_batch, t_batch)
    t_num = time.perf_counter() - start
    start = time.perf_counter()
    backprop = net.gradient(x_batch, t_batch)
    t_bp = time.perf_counter() - start
    diffs = {k: float(np.mean(np.abs(backprop[k] - numeric[k]))) for k in numeric}
    return diffs, t_num, t_bp


def main():
    (x_train, t_train), _, source = get_mnist(train_limit=3)
    diffs, t_num, t_bp = check(x_train[:3].astype(float), t_train[:3])
    print(f"数据：{source} 前 3 张")
    for k, v in diffs.items():
        print(f"{k}: 平均绝对误差 {v:.1e}")
    print(f"数值梯度耗时 {t_num:.3f} 秒，反向传播耗时 {t_bp * 1000:.3f} 毫秒，约快 {t_num / t_bp:.0f} 倍")


if __name__ == "__main__":
    main()
