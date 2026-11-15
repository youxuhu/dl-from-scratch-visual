"""第10讲：Affine 层的形状和反向传播，单个样本版和批版本。

∂L/∂X = ∂L/∂Y · Wᵀ，∂L/∂W = Xᵀ · ∂L/∂Y，偏置的梯度沿批方向求和。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from common.layers import Affine  # noqa: E402

W = np.array([[0.2, -0.4, 0.1], [0.6, 0.2, -0.3]])   # (2, 3)
b = np.array([0.1, 0.1, 0.2])                         # (3,)

# 单个样本：X (2,) → Y (3,)
X = np.array([1.0, 0.5])
one = Affine(W, b)
Y = one.forward(X)
dY = np.array([0.3, -0.2, 0.1])
dX = one.backward(dY)          # 形状和 X 一样 (2,)
dW_one = one.dW                # 形状和 W 一样 (2, 3)

# 批版本：N = 2
X2 = np.array([[1.0, 0.5], [0.0, 2.0]])
batch = Affine(W, b)
Y2 = batch.forward(X2)
dY2 = np.array([[0.3, -0.2, 0.1], [-0.1, 0.4, 0.2]])
dX2 = batch.backward(dY2)
dW2 = batch.dW
db2 = batch.db                 # 两行梯度压成一行


def main():
    print("W.T 形状", W.T.shape, "：\n", W.T)
    print("Y = X·W + b =", np.round(Y, 4), "形状", Y.shape)
    print("dX = dY·Wᵀ =", np.round(dX, 4), "形状", dX.shape)
    print("dW = Xᵀ·dY =\n", np.round(dW_one, 4), "形状", dW_one.shape)
    print("\n批版本 Y2 =\n", np.round(Y2, 4))
    print("dX2 =\n", np.round(dX2, 4))
    print("dW2 =\n", np.round(dW2, 4))
    print("db2 = dY2 沿 axis=0 求和 =", np.round(db2, 4))


if __name__ == "__main__":
    main()
