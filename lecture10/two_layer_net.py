"""第10讲：反向传播版 TwoLayerNet 的结构。

对应《深度学习入门：基于Python的理论与实现》官方源码 ch05/two_layer_net.py（MIT 许可，Copyright (c) 2016 Koki Saitoh）。
类本身在 common/two_layer_net.py（第11讲以后也要导入，所以放在 common）；这里打印层的顺序和参数形状，
并说明它和第8讲的数值梯度版 NumericalTwoLayerNet 接口相同、只是求梯度的方法不同。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.two_layer_net import TwoLayerNet  # noqa: E402,F401


def load_lecture08_net():
    """导入第8讲的数值梯度版网络；第8讲还没提交时返回 None。"""
    try:
        from lecture08.numerical_two_layer_net import NumericalTwoLayerNet
    except ImportError:
        return None
    return NumericalTwoLayerNet


if __name__ == "__main__":
    net = TwoLayerNet(784, 50, 10, seed=0)
    print("层的顺序：", " → ".join(net.layers.keys()), "→ SoftmaxWithLoss")
    for k, v in net.params.items():
        print(k, v.shape)
    old = load_lecture08_net()
    if old is None:
        print("第8讲的 NumericalTwoLayerNet 还没找到，跳过对照。")
    else:
        print("第8讲的", old.__name__, "：同样的构造参数和 params 键，梯度用数值微分求。")
