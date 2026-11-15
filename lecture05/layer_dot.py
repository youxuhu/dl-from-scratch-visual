"""多维数组和神经网络的内积：2 个输入、3 个输出的一层，一行 np.dot 算完。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from common import matmul_steps  # noqa: E402

X = np.array([1, 2])
W = np.array([[2, -1, 3], [1, 4, -2]])
Y = np.dot(X, W)


def columns():
    """逐列计算：第 j 个输出 = X 和 W 第 j 列的内积。"""
    return [s for _, _, s in matmul_steps(X.reshape(1, -1), W)]


def main():
    print("np.ndim(X) =", np.ndim(X), " X.shape =", X.shape)
    print("np.ndim(W) =", np.ndim(W), " W.shape =", W.shape)
    for j, s in enumerate(columns()):
        print(f"Y[{j}]: {s}")
    print("Y = np.dot(X, W) =", Y, " Y.shape =", Y.shape)


if __name__ == "__main__":
    main()
