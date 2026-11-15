"""为什么一条直线分不开异或？把 w1、w2、b 在 -1 到 1 之间按 0.1 的步长全试一遍。"""
import itertools

import numpy as np

from gates import INPUTS

TARGETS = {
    "AND": [0, 0, 0, 1],
    "NAND": [1, 1, 1, 0],
    "OR": [0, 1, 1, 1],
    "XOR": [0, 1, 1, 0],
}
GRID = np.round(np.linspace(-1, 1, 21), 1)
X = np.array(INPUTS)


def count_solutions(target):
    target = np.array(target)
    hits = 0
    for w1, w2, b in itertools.product(GRID, GRID, GRID):
        out = (X @ np.array([w1, w2]) + b > 0).astype(int)
        hits += int(np.array_equal(out, target))
    return hits


def main():
    total = len(GRID) ** 3
    print(f"一共试了 {total} 组 (w1, w2, b)")
    for name, target in TARGETS.items():
        print(f"{name}: 能做对的组合有 {count_solutions(target)} 组")


if __name__ == "__main__":
    main()
