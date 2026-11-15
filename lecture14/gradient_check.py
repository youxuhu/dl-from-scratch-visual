"""第14讲：SimpleConvNet 的梯度确认。

改编自《深度学习入门：基于Python的理论与实现》官方源码 ch07/gradient_check.py
（MIT 许可，Copyright (c) 2016 Koki Saitoh）。网络设置（输入 1×10×10、10 个 3×3 滤波器、隐藏层 10）、
随机输入 X、标签 T = 1，以及逐参数打印平均绝对误差的做法与原书一致；注释改写成中文，另固定了随机种子。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from common.simple_convnet import SimpleConvNet  # noqa: E402


def check(seed=0):
    network = SimpleConvNet(input_dim=(1, 10, 10),
                            conv_param={"filter_num": 10, "filter_size": 3, "pad": 0, "stride": 1},
                            hidden_size=10, output_size=10, weight_init_std=0.01, seed=seed)
    rng = np.random.default_rng(seed)
    X = rng.random(100).reshape((1, 1, 10, 10))
    T = np.array([1])
    grad_num = network.numerical_gradient(X, T)
    grad = network.gradient(X, T)
    return {key: float(np.abs(grad_num[key] - grad[key]).mean()) for key in grad_num}


def main():
    for key, diff in check().items():
        print(f"{key}: 平均绝对误差 {diff:.1e}")


if __name__ == "__main__":
    main()
