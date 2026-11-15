"""第13讲：Max 池化（2×2 窗口、步幅 2），以及对微小平移的鲁棒性。

朴素循环是教学演示；最后用 common/layers.py 的 Pooling 层核对。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from common.layers import Pooling  # noqa: E402

image = np.array([[7, 1, 6, 0],
                  [2, 3, 1, 2],
                  [1, 0, 2, 1],
                  [8, 2, 3, 9]])


def max_pool_loops(img, size=2, stride=2):
    """返回 (输出, 每个窗口最大值所在的原图坐标)。"""
    oh = (img.shape[0] - size) // stride + 1
    ow = (img.shape[1] - size) // stride + 1
    out = np.zeros((oh, ow), dtype=img.dtype)
    where = {}
    for i in range(oh):
        for j in range(ow):
            win = img[i * stride:i * stride + size, j * stride:j * stride + size]
            r, c = np.unravel_index(np.argmax(win), win.shape)
            out[i, j] = win[r, c]
            where[(i, j)] = (int(i * stride + r), int(j * stride + c))
    return out, where


pooled, argmax_pos = max_pool_loops(image)

# 整体右移一格（左边补 0，最右一列移出画面）
shifted = np.zeros_like(image)
shifted[:, 1:] = image[:, :-1]
pooled_shifted, _ = max_pool_loops(shifted)
unchanged = int(np.sum(pooled == pooled_shifted))

by_layer = Pooling(2, 2, stride=2).forward(image.reshape(1, 1, 4, 4).astype(float))[0, 0]


def main():
    print("输入：\n", image)
    print("Max 池化输出：\n", pooled)
    print("最大值的位置：", argmax_pos)
    print("右移一格后的输入：\n", shifted)
    print("右移后的池化输出：\n", pooled_shifted)
    print(f"4 格里有 {unchanged} 格不变")
    print("Pooling 层结果一致：", bool(np.array_equal(by_layer, pooled)))


if __name__ == "__main__":
    main()
