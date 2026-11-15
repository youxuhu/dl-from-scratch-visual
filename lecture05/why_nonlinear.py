"""为什么激活函数必须是非线性的：线性叠三层还是一层。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from common.functions import sigmoid  # noqa: E402

c = 2.0


def h_linear(v):
    return c * v


# 一个数穿过三层 h(x) = 2x
x0 = 1.5
three_layers = h_linear(h_linear(h_linear(x0)))   # 2·2·2·1.5 = 12
one_layer = (c ** 3) * x0

# 矩阵版：三个线性层合起来就是一个矩阵
X = np.array([1.0, -1.0])
W1 = np.array([[1.0, 2.0], [0.0, 1.0]])
W2 = np.array([[0.5, 0.0], [1.0, -1.0]])
W3 = np.array([[2.0, 1.0], [-1.0, 0.0]])
stacked = np.dot(np.dot(np.dot(X, W1), W2), W3)
W_all = np.dot(np.dot(W1, W2), W3)
merged = np.dot(X, W_all)

# sigmoid 不满足「放大两倍输入，输出也放大两倍」
s_of_2x = sigmoid(2.0)
two_s_of_x = 2 * sigmoid(1.0)


def main():
    print("三层 h(x)=2x 作用在 1.5 上:", three_layers, " 一层 8x:", one_layer)
    print("逐层相乘:", stacked, " 合并成一个矩阵:", merged)
    print("W_all =")
    print(W_all)
    print("sigmoid(2) =", round(float(s_of_2x), 3), " 2*sigmoid(1) =", round(float(two_s_of_x), 3))


if __name__ == "__main__":
    main()
