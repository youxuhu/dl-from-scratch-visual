"""第13讲：输出大小公式 OH = (H + 2P - FH) / S + 1，以及全连接层把图片拉直后丢掉的形状信息。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from common.util import conv_output_size  # noqa: E402

EXAMPLES = [  # (H, P, FH, S)
    (4, 0, 3, 1),
    (4, 1, 3, 1),
    (7, 0, 3, 2),
    (28, 0, 5, 1),
    (28, 2, 5, 1),
]
sizes = {ex: conv_output_size(ex[0], ex[2], ex[3], ex[1]) for ex in EXAMPLES}


def flat_index(row, col, width=28):
    """28×28 的图拉直成 784 后，第 row 行第 col 列在第几格。"""
    return row * width + col


# 上下相邻的两个像素，拉直后相隔 28 格
above, below = flat_index(10, 14), flat_index(11, 14)
img = np.arange(28 * 28).reshape(28, 28)
assert img.flatten()[below] - img.flatten()[above] == 28


def main():
    for (h, p, fh, s), o in sizes.items():
        print(f"H={h} P={p} FH={fh} S={s}：({h} + 2×{p} - {fh}) / {s} + 1 = {o}")
    try:
        conv_output_size(4, 3, 2, 1)
    except ValueError as e:
        print("除不尽的情况：", e)
    print(f"第10行第14列 → 第 {above} 格，第11行第14列 → 第 {below} 格，相隔 {below - above}")


if __name__ == "__main__":
    main()
