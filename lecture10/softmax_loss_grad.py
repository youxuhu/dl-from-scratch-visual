"""第10讲：Softmax-with-Loss 层的反向传播就是 (y - t) / 批大小。

两个 10 类样本，正确答案都是 3：一个预测得很差，一个已经很准。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from common.layers import SoftmaxWithLoss  # noqa: E402

label = 3
t = np.zeros(10)
t[label] = 1.0

bad_scores = np.array([1.0, 0.5, 2.0, 0.2, -0.3, 0.0, 1.5, -1.0, 0.3, 0.8])
good_scores = np.array([-1.0, -0.5, 0.0, 6.0, -2.0, 0.5, -1.5, 0.2, -0.8, 0.1])


def run(scores):
    layer = SoftmaxWithLoss()
    loss = layer.forward(scores, t)
    return layer.y, loss, layer.backward(1)


bad_y, bad_loss, bad_grad = run(bad_scores)
good_y, good_loss, good_grad = run(good_scores)


def main():
    np.set_printoptions(precision=3, suppress=True)
    print("预测很差的样本")
    print("  y     =", bad_y)
    print("  y - t =", bad_grad, f" 损失 {bad_loss:.4f}")
    print("预测很准的样本")
    print("  y     =", good_y)
    print("  y - t =", good_grad, f" 损失 {good_loss:.4f}")
    print(f"正确类别那格的梯度：差 {bad_grad[label]:.3f}，准 {good_grad[label]:.3f}")


if __name__ == "__main__":
    main()
