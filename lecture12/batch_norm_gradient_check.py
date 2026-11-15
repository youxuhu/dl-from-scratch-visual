"""第12讲：带 Batch Normalization 的网络做梯度确认。

改编自《深度学习入门：基于Python的理论与实现》官方源码 ch06/batch_norm_gradient_check.py
（MIT 许可，Copyright (c) 2016 Koki Saitoh）。网络 MultiLayerNetExtend(784, [100, 100], 10, use_batchnorm=True)、
变量名 x_batch、t_batch、grad_backprop、grad_numerical 和逐参数打印平均绝对误差的做法与原书一致；注释改写成中文。

和原书的一处不同：原书只取 1 张图（x_train[:1]），批大小为 1 时 BN 减去均值后全是 0，
W1、b1 的梯度都是 0，比较不出东西。这里取前 3 张，让每个参数都有非零梯度。
QUICK=1：隐藏层缩小到 [20, 20]，几秒钟跑完；完整设置 [100, 100] 要数值微分约 9 万个参数，慢一些。
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # 小矩阵用多线程反而慢，机器忙时差几十倍
import numpy as np  # noqa: E402

from common.multi_layer_net_extend import MultiLayerNetExtend  # noqa: E402
from common.util import get_mnist, is_quick  # noqa: E402


def check(x_batch, t_batch, hidden_size_list=(100, 100), seed=0):
    network = MultiLayerNetExtend(input_size=x_batch.shape[1], hidden_size_list=list(hidden_size_list),
                                  output_size=10, use_batchnorm=True, seed=seed)
    grad_backprop = network.gradient(x_batch, t_batch)
    grad_numerical = network.numerical_gradient(x_batch, t_batch)
    return {key: float(np.average(np.abs(grad_backprop[key] - grad_numerical[key])))
            for key in grad_numerical}


def main():
    (x_train, t_train), _, source = get_mnist(train_limit=3, test_limit=1)
    x_batch = x_train[:3].astype(float)
    t_batch = t_train[:3]
    hidden = (20, 20) if is_quick() else (100, 100)
    print(f"数据：{source} 前 3 张；网络 784 → {list(hidden)} → 10，带 BN")
    for key, diff in check(x_batch, t_batch, hidden).items():
        print(f"{key}: 平均绝对误差 {diff:.1e}")


if __name__ == "__main__":
    main()
