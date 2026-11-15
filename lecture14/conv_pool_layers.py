"""第14讲：Convolution 层和 Pooling 层内部的形状变化（实现在 common/layers.py）。

卷积：im2col → 滤波器 reshape(FN, -1).T → 矩阵乘法加偏置 → reshape 成 (N, OH, OW, FN) → transpose(0, 3, 1, 2)。
池化：im2col 后每行一个窗口 → 每行取最大值 → reshape 回特征图。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from common.layers import Convolution, Pooling  # noqa: E402
from common.util import im2col  # noqa: E402

rng = np.random.default_rng(0)
x = rng.random((2, 3, 7, 7))
W = rng.standard_normal((4, 3, 5, 5))
b = np.zeros(4)

conv = Convolution(W, b)
y = conv.forward(x)
shapes = {
    "x": x.shape,
    "col": conv.col.shape,            # (N*OH*OW, C*FH*FW) = (18, 75)
    "col_W": conv.col_W.shape,        # (C*FH*FW, FN) = (75, 4)
    "out(N, OH, OW, FN)": (2, 3, 3, 4),
    "y(N, FN, OH, OW)": y.shape,
}

# transpose(0, 3, 1, 2) 把 (N, OH, OW, FN) 的第 3 轴搬到第 1 轴
demo = np.arange(2 * 3 * 3 * 4).reshape(2, 3, 3, 4)
moved = demo.transpose(0, 3, 1, 2)

# 池化：一个 2 通道 4×4 的特征图
fmap = np.array([[[7, 1, 6, 0], [2, 3, 1, 2], [1, 0, 2, 1], [8, 2, 3, 9]],
                 [[0, 5, 1, 1], [4, 2, 0, 3], [6, 1, 1, 0], [2, 2, 0, 4]]], dtype=float)[np.newaxis]
pool_rows = im2col(fmap, 2, 2, stride=2).reshape(-1, 4)   # 每行一个窗口
row_max = pool_rows.max(axis=1)
pool = Pooling(2, 2, stride=2)
pooled = pool.forward(fmap)


def main():
    for k, v in shapes.items():
        print(f"{k:>20}: {v}")
    print("transpose(0, 3, 1, 2)：", demo.shape, "→", moved.shape)
    print("池化的窗口行 (每行 4 格)：\n", pool_rows)
    print("每行最大值：", row_max)
    print("Pooling 层输出：\n", pooled[0])


if __name__ == "__main__":
    main()
