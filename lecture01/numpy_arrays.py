"""第1讲 第二部分：NumPy 数组、shape、dtype、逐元素运算。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from common import describe

scores = [3, 1, 4]
x = np.array([3, 1, 4])
y = x * 2

a = np.array([1.5, -2.0, 4.0])
b = np.array([0.5, 0.5, 2.0])
add = a + b
mul = a * b
div = a / b

A = np.array([[2, 0, 5], [1, 3, -1]])


def main():
    print("scores * 2 =", scores * 2)
    print("x * 2 =", y)
    print("x:", describe(x))
    print("a:", describe(a))
    print("a + b =", add)
    print("a * b =", mul)
    print("a / b =", div)
    print("A.shape =", A.shape)
    print("A.ndim =", A.ndim)
    print("A:", describe(A))


if __name__ == "__main__":
    main()
