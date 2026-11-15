"""第9讲：最简单的两种层，乘法层 MulLayer 和加法层 AddLayer。

对应《深度学习入门：基于Python的理论与实现》官方源码 ch05/layer_naive.py（MIT 许可，Copyright (c) 2016 Koki Saitoh）。
两个类的实现放在 common/layers.py（后面各讲的层都在那里，类名和方法与原书一致），这里导入后做个小演示。
每个层只管自己：forward 记下输入，backward 按局部导数把上游梯度分给两个输入。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.layers import AddLayer, MulLayer  # noqa: E402,F401

if __name__ == "__main__":
    m = MulLayer()
    print("MulLayer 正向 3 × 5 =", m.forward(3, 5), "，反向 dout=1 →", m.backward(1))
    a = AddLayer()
    print("AddLayer 正向 3 + 5 =", a.forward(3, 5), "，反向 dout=2 →", a.backward(2))
