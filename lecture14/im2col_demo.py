"""第14讲：im2col 把卷积变成矩阵乘法。

沿用第13讲的 4×4 输入和 3×3 滤波器：4 个窗口位置展开成 4 行 9 列，滤波器拉直成 9 个数的一列，
相乘得到的 4 个数正好是第13讲逐格乘加的结果。再看 7×7 三通道输入的展开形状，以及 col2im 的重叠累加。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from common.util import col2im, im2col  # noqa: E402

image = np.array([[2, 1, 0, 3],
                  [1, 3, 2, 0],
                  [0, 2, 1, 1],
                  [3, 0, 1, 2]]).reshape(1, 1, 4, 4)
kernel = np.array([[1, 0, -1],
                   [2, 1, 0],
                   [0, 1, 1]])

rows = im2col(image, 3, 3)                 # (4, 9)
column = kernel.reshape(9, 1)              # 滤波器拉直成一列
result = np.dot(rows, column).ravel()      # [10, 8, 2, 11]
result_2x2 = result.reshape(2, 2)

# 同一个原始像素出现在几行里：把全 1 的行矩阵 col2im 回去就是出现次数
appear_count = col2im(np.ones_like(rows, dtype=float), image.shape, 3, 3)[0, 0].astype(int)
memory_ratio = rows.size / image.size      # 展开后元素个数是原来的几倍

# 7×7、3 通道、5×5 滤波器的展开形状
x1 = np.random.default_rng(0).random((1, 3, 7, 7))
col1 = im2col(x1, 5, 5, stride=1, pad=0)   # (9, 75)
x2 = np.random.default_rng(1).random((10, 3, 7, 7))
col2 = im2col(x2, 5, 5, stride=1, pad=0)   # (90, 75)


def main():
    print("im2col 展开 (4, 9)：\n", rows)
    print("乘拉直的滤波器：", result, "→ reshape 成 2×2：\n", result_2x2)
    print("每个像素在展开矩阵里出现的次数：\n", appear_count)
    print(f"展开后元素个数是原图的 {memory_ratio:.2f} 倍（用内存换速度）")
    print("(1, 3, 7, 7) 用 5×5 滤波器展开：", col1.shape)
    print("(10, 3, 7, 7) 用 5×5 滤波器展开：", col2.shape)


if __name__ == "__main__":
    main()
