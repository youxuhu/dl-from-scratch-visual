"""第10讲：梯度确认。

改编自《深度学习入门：基于Python的理论与实现》官方源码 ch05/gradient_check.py
（MIT 许可，Copyright (c) 2016 Koki Saitoh）。变量名 x_batch、t_batch、grad_numerical、grad_backprop 和
「逐个参数打印平均绝对误差」的做法与原书一致；另加了耗时对比，并改用第8讲的数值梯度版网络。
第8讲的数值梯度版和本讲的反向传播版，用同一组参数、同一批数据求梯度，逐格比较。

做法：同一个 seed 构造第8讲的 NumericalTwoLayerNet 和 common 里的 TwoLayerNet（隐藏层和第8讲一样用 sigmoid），
把前者的参数拷给后者，
前者用 numerical_gradient、后者用 gradient，比较平均绝对误差，并对比耗时。
第8讲的文件不在时，退回用 TwoLayerNet 自带的 numerical_gradient（同样调用 common/gradient.py）。
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # 小矩阵用多线程反而慢，机器忙时差几十倍
import numpy as np  # noqa: E402

from common.two_layer_net import TwoLayerNet  # noqa: E402
from common.util import get_mnist  # noqa: E402


def make_pair(seed=0, sizes=(784, 50, 10)):
    """返回 (数值梯度用的网络, 反向传播网络, 数值网络来源说明)，两者参数完全相同。"""
    try:
        from lecture08.numerical_two_layer_net import NumericalTwoLayerNet
        num_net = NumericalTwoLayerNet(*sizes, seed=seed)
        bp_net = TwoLayerNet(*sizes, seed=seed, activation="sigmoid")
        bp_net.load_from(num_net.params)
        source = "第8讲 NumericalTwoLayerNet"
    except ImportError:
        bp_net = TwoLayerNet(*sizes, seed=seed)
        num_net = bp_net
        source = "TwoLayerNet.numerical_gradient（第8讲文件未找到）"
    return num_net, bp_net, source


def check(x_batch, t_batch, seed=0, sizes=(784, 50, 10)):
    num_net, bp_net, source = make_pair(seed, sizes)
    start = time.perf_counter()
    grad_numerical = num_net.numerical_gradient(x_batch, t_batch)
    t_num = time.perf_counter() - start
    start = time.perf_counter()
    grad_backprop = bp_net.gradient(x_batch, t_batch)
    t_bp = time.perf_counter() - start
    # 各参数的平均绝对误差
    diffs = {key: float(np.average(np.abs(grad_backprop[key] - grad_numerical[key])))
             for key in grad_numerical}
    return diffs, t_num, t_bp, source


def main():
    (x_train, t_train), _, data_source = get_mnist(train_limit=3)
    x_batch = x_train[:3].astype(float)
    t_batch = t_train[:3]
    diffs, t_num, t_bp, source = check(x_batch, t_batch)
    print(f"数据：{data_source} 前 3 张；数值梯度来自 {source}")
    for key, diff in diffs.items():
        print(f"{key}: 平均绝对误差 {diff:.1e}")
    print(f"数值梯度耗时 {t_num:.3f} 秒，反向传播耗时 {t_bp * 1000:.3f} 毫秒，约快 {t_num / t_bp:.0f} 倍")


if __name__ == "__main__":
    main()
