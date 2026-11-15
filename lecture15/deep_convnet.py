"""第15讲：DeepConvNet 的结构：逐层输出形状和参数个数。

对应《深度学习入门：基于Python的理论与实现》官方源码 ch08/deep_convnet.py（MIT 许可，Copyright (c) 2016 Koki Saitoh）。
类本身在 common/deep_convnet.py（构造参数 conv_param_1 到 conv_param_6 与原书一致），这里导入后打印结构。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.deep_convnet import CONV_PLAN, DeepConvNet  # noqa: E402

net = DeepConvNet(seed=0)
filter_nums = [c for c, _ in CONV_PLAN]
pads = [p for _, p in CONV_PLAN]
param_count = int(sum(v.size for v in net.params.values()))
layer_names = [type(layer).__name__ for layer in net.layers] + ["SoftmaxWithLoss"]


def main():
    print("卷积层通道数：", filter_nums, "，填充：", pads)
    print("层的顺序：", " → ".join(layer_names))
    print("形状变化：", " → ".join("×".join(map(str, s)) for s in net.shapes))
    print(f"参数总数 {param_count:,}")


if __name__ == "__main__":
    main()
