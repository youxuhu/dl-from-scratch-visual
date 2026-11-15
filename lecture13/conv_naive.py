"""第13讲：卷积运算逐格乘加（朴素 for 循环版），再用 common 里的 im2col 和 Convolution 层核对。

输入 4×4、滤波器 3×3 是本课自己设计的数。朴素循环是教学演示，后面各讲用的是 common/layers.py 的 Convolution。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from common.layers import Convolution  # noqa: E402
from common.util import conv_output_size, im2col  # noqa: E402

image = np.array([[2, 1, 0, 3],
                  [1, 3, 2, 0],
                  [0, 2, 1, 1],
                  [3, 0, 1, 2]])
kernel = np.array([[1, 0, -1],
                   [2, 1, 0],
                   [0, 1, 1]])


def conv2d_loops(img, ker, bias=0, pad=0, stride=1):
    """朴素卷积：滤波器在输入上滑动，对应格相乘再求和，最后加偏置。

    返回 (输出, 每个窗口的 9 个乘积)。
    """
    if pad:
        img = np.pad(img, pad)
    kh, kw = ker.shape
    oh = (img.shape[0] - kh) // stride + 1
    ow = (img.shape[1] - kw) // stride + 1
    out = np.zeros((oh, ow), dtype=int)
    products = {}
    for i in range(oh):
        for j in range(ow):
            window = img[i * stride:i * stride + kh, j * stride:j * stride + kw]
            prod = window * ker
            products[(i, j)] = prod.ravel().tolist()
            out[i, j] = prod.sum() + bias
    return out, products


out, products = conv2d_loops(image, kernel)
first_products = products[(0, 0)]                 # 2、0、0、2、3、0、0、2、1
out_bias = conv2d_loops(image, kernel, bias=3)[0]
out_pad1 = conv2d_loops(image, kernel, pad=1)[0]

# 用 im2col 展开成 (4, 9)，再和拉直的滤波器相乘
cols = im2col(image.reshape(1, 1, 4, 4), 3, 3)
by_matmul = np.dot(cols, kernel.ravel())

# common 的 Convolution 层
layer = Convolution(kernel.reshape(1, 1, 3, 3).astype(float), np.array([3.0]))
by_layer = layer.forward(image.reshape(1, 1, 4, 4).astype(float))[0, 0]


def main():
    print("第一个窗口的 9 个乘积：", first_products, "求和 =", sum(first_products))
    print("输出（无填充、步幅 1）：\n", out)
    print("加偏置 3：\n", out_bias)
    print("填充 1 时输出 4×4：\n", out_pad1)
    print("输出边长公式：(4 + 2×0 - 3) / 1 + 1 =", conv_output_size(4, 3, 1, 0))
    print("im2col 展开形状", cols.shape, "，乘拉直的滤波器 =", by_matmul)
    print("Convolution 层（偏置 3）=\n", by_layer)


if __name__ == "__main__":
    main()
