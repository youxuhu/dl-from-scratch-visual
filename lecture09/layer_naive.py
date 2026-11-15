"""第9讲：最简单的两种层，乘法层 MulLayer 和加法层 AddLayer。

每个层只管自己：forward 记下输入，backward 按局部导数把上游梯度分给两个输入。
"""


class MulLayer:
    """z = x * y。反向时把两个输入「翻转」：dx = dout * y，dy = dout * x。"""

    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x, self.y = x, y
        return x * y

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy


class AddLayer:
    """z = x + y。两个局部导数都是 1，上游梯度原样分给两边。"""

    def forward(self, x, y):
        return x + y

    def backward(self, dout):
        return dout * 1, dout * 1


if __name__ == "__main__":
    m = MulLayer()
    print("MulLayer 正向 3 × 5 =", m.forward(3, 5), "，反向 dout=1 →", m.backward(1))
    a = AddLayer()
    print("AddLayer 正向 3 + 5 =", a.forward(3, 5), "，反向 dout=2 →", a.backward(2))
