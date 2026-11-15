"""第1讲 第六部分：索引、切片、布尔掩码、astype(int)、flatten。"""
import numpy as np

temps = np.array([23, 8, 31, 17, 5, 26])
first = temps[0]
middle = temps[1:4]
mask = temps > 20
hot = temps[mask]
hot_flags = mask.astype(int)
hot_days = hot_flags.sum()

f = np.array([2.7, -1.6, 0.4])
f_int = f.astype(int)

A = np.array([[2, 0, 5], [1, 3, -1]])
row1 = A[1]
elem = A[1, 2]
flat = A.flatten()
positive = flat[flat > 0]


def main():
    print("temps[0] =", first)
    print("temps[1:4] =", middle)
    print("temps > 20 =", mask)
    print("temps[mask] =", hot)
    print("mask.astype(int) =", hot_flags)
    print("hot_days =", hot_days)
    print("f.astype(int) =", f_int)
    print("A[1] =", row1)
    print("A[1, 2] =", elem)
    print("A.flatten() =", flat)
    print("flat[flat > 0] =", positive)


if __name__ == "__main__":
    main()
