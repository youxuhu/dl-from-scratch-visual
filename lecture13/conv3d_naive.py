"""第13讲：3 维数据的卷积。3 个通道各自乘积累加，三张平面逐格相加；FN 个滤波器得到 FN 个输出通道。

形状约定 (N, C, H, W)。最后用 common 的 Convolution 层核对，并演示一批 10 张 28×28 图片的形状。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from common.layers import Convolution  # noqa: E402

x = np.array([
    [[2, 1, 0, 3], [1, 3, 2, 0], [0, 2, 1, 1], [3, 0, 1, 2]],   # 通道 0（第13讲的 4×4）
    [[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]],   # 通道 1（棋盘格）
    [[0, 1, 2, 1], [1, 0, 1, 2], [2, 1, 0, 1], [1, 2, 1, 0]],   # 通道 2
])                                                              # (C, H, W) = (3, 4, 4)

filters = np.array([
    [[[1, 0, -1], [2, 1, 0], [0, 1, 1]],      # 滤波器 0 的 3 个通道
     [[1, 1, 1], [0, 0, 0], [-1, -1, -1]],
     [[0, 1, 0], [1, -1, 1], [0, 1, 0]]],
    [[[0, 0, 0], [0, 1, 0], [0, 0, 0]],       # 滤波器 1：只看每个通道的中心
     [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
     [[0, 0, 0], [0, 1, 0], [0, 0, 0]]],
])                                             # (FN, C, FH, FW) = (2, 3, 3, 3)


def conv_one_channel(img, ker):
    oh, ow = img.shape[0] - ker.shape[0] + 1, img.shape[1] - ker.shape[1] + 1
    return np.array([[np.sum(img[i:i + 3, j:j + 3] * ker) for j in range(ow)] for i in range(oh)])


# 滤波器 0：每个通道一张平面，再逐格相加
planes = [conv_one_channel(x[c], filters[0, c]) for c in range(3)]
summed = sum(planes)
# 两个滤波器 → 两个输出通道
stacked = np.array([sum(conv_one_channel(x[c], f[c]) for c in range(3)) for f in filters])

layer = Convolution(filters.astype(float), np.zeros(2))
by_layer = layer.forward(x[np.newaxis].astype(float))[0]

# 批处理：10 张 1 通道 28×28 图片，30 个 5×5 滤波器
batch = np.random.default_rng(0).random((10, 1, 28, 28))
batch_layer = Convolution(np.zeros((30, 1, 5, 5)), np.zeros(30))
batch_out_shape = batch_layer.forward(batch).shape


def main():
    for c, p in enumerate(planes):
        print(f"滤波器0 × 通道{c}：\n", p)
    print("三张平面相加：\n", summed)
    print("两个滤波器叠成的输出 (FN, OH, OW) =", stacked.shape, "\n", stacked)
    print("Convolution 层结果一致：", bool(np.array_equal(by_layer, stacked)))
    print("输入形状 (C, H, W) =", x.shape, "，滤波器形状 (FN, C, FH, FW) =", filters.shape)
    print("批处理：输入", batch.shape, "→ 输出", batch_out_shape)


if __name__ == "__main__":
    main()
