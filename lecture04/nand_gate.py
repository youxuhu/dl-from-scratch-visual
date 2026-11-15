"""与非门。改编自 oreilly-japan/deep-learning-from-scratch 的 ch02/nand_gate.py
（MIT, Copyright (c) 2016 Koki Saitoh）。权重和偏置与书中一致。"""
import numpy as np


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def weighted_sum(x1, x2):
    return float(np.sum(np.array([-0.5, -0.5]) * np.array([x1, x2])) + 0.7)


if __name__ == "__main__":
    print("NAND: w = [-0.5, -0.5], b = 0.7")
    for xs in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        s = weighted_sum(*xs)
        y = NAND(*xs)
        print(f"  {xs}  加权和 tmp = {s:+.2f}  输出 {y}")
