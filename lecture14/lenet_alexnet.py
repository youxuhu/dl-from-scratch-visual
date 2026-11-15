"""第14讲：LeNet 和 AlexNet 的结构链，用输出大小公式逐层算形状，并数一数卷积层参数。

只是结构示意（不训练）。LeNet 输入 1×32×32，AlexNet 输入 3×227×227（常见的修正版尺寸）。

本仓库补充，对应书中 7.7 节，书中没有对应源码文件。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.util import conv_output_size  # noqa: E402

LENET = [("conv", 6, 5, 1, 0), ("subsample", 2), ("conv", 16, 5, 1, 0), ("subsample", 2)]
ALEXNET = [("conv", 96, 11, 4, 0), ("pool", 3, 2), ("conv", 256, 5, 1, 2), ("pool", 3, 2),
           ("conv", 384, 3, 1, 1), ("conv", 384, 3, 1, 1), ("conv", 256, 3, 1, 1), ("pool", 3, 2)]


def walk(plan, channels, side):
    """返回每层之后的 (名字, 通道, 边长) 列表和卷积层参数总数。"""
    trace = [("input", channels, side)]
    n_params = 0
    for step in plan:
        if step[0] == "conv":
            _, fn, fs, stride, pad = step
            n_params += fn * channels * fs * fs + fn
            channels, side = fn, conv_output_size(side, fs, stride, pad)
        elif step[0] == "subsample":
            side //= step[1]
        else:
            _, size, stride = step
            side = conv_output_size(side, size, stride, 0)
        trace.append((step[0], channels, side))
    return trace, n_params


lenet_trace, lenet_params = walk(LENET, 1, 32)
alex_trace, alex_params = walk(ALEXNET, 3, 227)


def main():
    for name, trace, n in (("LeNet", lenet_trace, lenet_params), ("AlexNet", alex_trace, alex_params)):
        chain = " → ".join(f"{k}({c}×{s}×{s})" for k, c, s in trace)
        print(f"{name}：{chain}")
        print(f"  卷积层参数共 {n:,} 个")


if __name__ == "__main__":
    main()
