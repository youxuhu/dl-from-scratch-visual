"""第14讲：SimpleConvNet 的结构。

对应《深度学习入门：基于Python的理论与实现》官方源码 ch07/simple_convnet.py（MIT 许可，Copyright (c) 2016 Koki Saitoh）。
类本身在 common/simple_convnet.py（第15讲也要用，所以放在 common），层的命名 Conv1、Relu1、Pool1、Affine1、Relu2、Affine2
与原书一致；这里打印层的顺序、各层输出形状和参数形状。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.simple_convnet import SimpleConvNet  # noqa: E402,F401

if __name__ == "__main__":
    network = SimpleConvNet(input_dim=(1, 28, 28),
                            conv_param={"filter_num": 30, "filter_size": 5, "pad": 0, "stride": 1},
                            hidden_size=100, output_size=10, weight_init_std=0.01, seed=0)
    print("层的顺序：", " → ".join(network.layers.keys()), "→ SoftmaxWithLoss")
    print("各层输出形状：", " → ".join("×".join(map(str, s)) for s in network.shapes))
    for key, val in network.params.items():
        print(key, val.shape)
