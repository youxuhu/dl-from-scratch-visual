"""神经网络的梯度：一个只有 2×3 权重的小网络，用数值梯度求 dL/dW，再走一步。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from common.functions import cross_entropy_error, softmax  # noqa: E402
from common.gradient import numerical_gradient  # noqa: E402


class SimpleNet:
    def __init__(self):
        self.W = np.array([[0.3, -0.2, 0.6], [-0.5, 0.4, 0.1]])

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        return cross_entropy_error(softmax(self.predict(x)), t)


x = np.array([0.8, -0.5])
t = 1                         # 正确答案是第 1 类
LR = 0.5

net = SimpleNet()
scores = net.predict(x)
loss_before = net.loss(x, t)
dW = numerical_gradient(lambda _: net.loss(x, t), net.W)
net.W -= LR * dW
loss_after = net.loss(x, t)


def main():
    print("分数 x·W =", np.round(scores, 3))
    print("概率 =", np.round(softmax(scores), 4))
    print("损失（更新前）=", round(loss_before, 4))
    print("dW =")
    print(np.round(dW, 4))
    print(f"W -= {LR}·dW 之后损失 =", round(loss_after, 4))


if __name__ == "__main__":
    main()
