"""第1讲 第五部分：np.dot 矩阵乘法与行向量写法。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from common import matmul_steps

A = np.array([[1, 0, 2], [-1, 3, 1]])
B = np.array([[3, 1], [2, 1], [1, 0]])
C = np.dot(A, B)

B_bad = np.array([[1, 2], [3, 4]])

# 行向量写法：一个样本是一行
X = np.array([1.0, 0.5])
W = np.array([[0.2, -0.4, 0.1], [0.6, 0.2, -0.3]])
b = np.array([0.1, 0.1, 0.2])
Y = np.dot(X, W) + b

# 一次喂两个样本：每行一个
X2 = np.array([[1.0, 0.5], [0.0, 2.0]])
Y2 = np.dot(X2, W) + b


def shape_error_message():
    try:
        np.dot(A, B_bad)
    except ValueError as e:
        return str(e)
    return ""


def main():
    print("C = np.dot(A, B) =")
    print(C)
    print("C.shape =", C.shape)
    for i, j, s in matmul_steps(A, B):
        print(f"C[{i},{j}]: {s}")
    print("np.dot(A, B_bad) 报错：", shape_error_message())
    print("Y =", np.round(Y, 2))
    print("Y.shape =", Y.shape)
    print("Y2 =")
    print(np.round(Y2, 2))
    print("Y2.shape =", Y2.shape)


if __name__ == "__main__":
    main()
