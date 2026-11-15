"""第1讲 第四部分：axis 到底沿哪个方向。"""
import numpy as np

A = np.array([[2, 0, 5], [1, 3, -1]])
col_sum = A.sum(axis=0)
row_sum = A.sum(axis=1)
row_max = A.max(axis=1)
row_argmax = A.argmax(axis=1)


def main():
    print("A.sum(axis=0) =", col_sum)
    print("A.sum(axis=1) =", row_sum)
    print("A.sum() =", A.sum())
    print("A.max(axis=1) =", row_max)
    print("A.argmax(axis=1) =", row_argmax)


if __name__ == "__main__":
    main()
