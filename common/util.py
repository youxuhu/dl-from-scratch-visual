"""卷积相关的展开工具、数据打乱，以及训练脚本共用的取数据函数。

smooth_curve、shuffle_dataset、conv_output_size、im2col、col2im 的函数名和参数与
《深度学习入门：基于Python的理论与实现》官方源码 common/util.py 一致（MIT 许可，Copyright (c) 2016 Koki Saitoh），
展开的行列顺序也相同；im2col 这里用 sliding_window_view 实现。get_mnist、toy_digits、is_quick、get_pyplot 是本仓库加的。

im2col 把卷积要看的每一个窗口拉成一行，卷积就变成一次矩阵乘法；
col2im 反过来，把行矩阵里的数放回原图位置，重叠的位置累加。
"""
import os

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def get_pyplot(out_dir):
    """有 matplotlib 就返回 pyplot（Agg 后端，不弹窗口）并建好输出目录；没装就返回 None。

    各讲的画图都是可选的：没有 matplotlib 时只打印数值，不影响运行。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("提示：没装 matplotlib，跳过画图。")
        return None
    os.makedirs(out_dir, exist_ok=True)
    return plt


def is_quick():
    """环境变量 QUICK=1 时返回 True，训练脚本据此只跑很少的步数。"""
    return os.environ.get("QUICK", "") not in ("", "0")


def conv_output_size(input_size, filter_size, stride=1, pad=0, strict=True):
    """输出边长 = (输入边长 + 2*填充 - 滤波器边长) / 步幅 + 1。

    strict=True 时除不尽就报错（讲公式时用）；strict=False 时向下取整，
    最后放不下一个完整窗口的那几行、几列直接丢掉（池化遇到奇数边长时就是这样）。
    """
    span = input_size + 2 * pad - filter_size
    if span < 0 or (strict and span % stride != 0):
        raise ValueError(f"尺寸对不上：H={input_size} P={pad} FH={filter_size} S={stride}")
    return span // stride + 1


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """(N, C, H, W) 的输入展开成 (N*OH*OW, C*filter_h*filter_w) 的二维矩阵。

    每一行对应一个窗口位置，行的顺序是 样本、输出行、输出列；
    每一行里的列顺序是 通道、窗口内的行、窗口内的列（行优先）。
    """
    imgs = np.asarray(input_data)
    n, c, h, w = imgs.shape
    oh = conv_output_size(h, filter_h, stride, pad, strict=False)
    ow = conv_output_size(w, filter_w, stride, pad, strict=False)
    if pad:
        imgs = np.pad(imgs, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    # windows 的形状：(N, C, 所有行位置, 所有列位置, filter_h, filter_w)
    windows = sliding_window_view(imgs, (filter_h, filter_w), axis=(2, 3))
    windows = windows[:, :, ::stride, ::stride][:, :, :oh, :ow]
    # 调成 (N, OH, OW, C, filter_h, filter_w) 再压平
    rows = windows.transpose(0, 2, 3, 1, 4, 5)
    return rows.reshape(n * oh * ow, c * filter_h * filter_w).copy()


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """im2col 的反操作：把 (N*OH*OW, C*filter_h*filter_w) 的矩阵加回 (N, C, H, W)。

    同一个像素出现在几个窗口里，就会被加几次，反向传播时正好是梯度的累加。
    """
    n, c, h, w = input_shape
    oh = conv_output_size(h, filter_h, stride, pad, strict=False)
    ow = conv_output_size(w, filter_w, stride, pad, strict=False)
    blocks = np.asarray(col).reshape(n, oh, ow, c, filter_h, filter_w)
    canvas = np.zeros((n, c, h + 2 * pad, w + 2 * pad), dtype=blocks.dtype)
    for dy in range(filter_h):
        for dx in range(filter_w):
            # 窗口里第 (dy, dx) 格，在所有窗口位置上对应的原图像素
            piece = blocks[:, :, :, :, dy, dx].transpose(0, 3, 1, 2)
            canvas[:, :, dy:dy + stride * oh:stride, dx:dx + stride * ow:stride] += piece
    return canvas[:, :, pad:pad + h, pad:pad + w]


def shuffle_dataset(x, t, rng=None):
    """把样本和标签用同一个随机顺序打乱，返回新的 (x, t)。"""
    rng = np.random.default_rng() if rng is None else rng
    order = rng.permutation(len(x))
    return x[order], t[order]


def smooth_curve(values, window=11):
    """滑动平均，把抖动的损失曲线变平滑，长度不变。"""
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values.copy()
    kernel = np.ones(window) / window
    padded = np.pad(values, window // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def toy_digits(n_train=600, n_test=200, flatten=True, seed=0):
    """拿不到 MNIST 时的替身：10 类带噪声的 28×28 图案，形状和 MNIST 一样。

    每类有一个固定的「原型」笔画图案，样本是原型随机平移一点再加噪声，
    所以网络确实能学，只是比真 MNIST 容易。
    """
    rng = np.random.default_rng(seed)
    protos = np.zeros((10, 28, 28))
    for k in range(10):
        r0, c0 = 4 + (k % 5) * 3, 4 + (k // 5) * 8
        protos[k, r0:r0 + 12, c0:c0 + 3] = 1.0
        protos[k, 6 + k:9 + k, 6:22] = 1.0

    def make(count):
        labels = rng.integers(0, 10, size=count)
        imgs = np.empty((count, 28, 28))
        for i, lab in enumerate(labels):
            shift = rng.integers(-2, 3, size=2)
            imgs[i] = np.roll(protos[lab], tuple(shift), axis=(0, 1))
        imgs += rng.normal(0, 0.15, imgs.shape)
        imgs = np.clip(imgs, 0, 1).astype(np.float32)
        imgs = imgs.reshape(count, -1) if flatten else imgs.reshape(count, 1, 28, 28)
        return imgs, labels.astype(int)

    return make(n_train), make(n_test)


def get_mnist(flatten=True, train_limit=None, test_limit=None):
    """训练脚本统一用这个取数据：优先读 MNIST，下载失败就换成 toy_digits 并打印提示。

    train_limit / test_limit 用来只取前若干张，QUICK 模式下用得上。
    返回 ((x_train, t_train), (x_test, t_test), 数据来源说明)。
    """
    try:
        from common.mnist import load_mnist
        (xtr, ttr), (xte, tte) = load_mnist(normalize=True, flatten=flatten)
        source = "MNIST"
    except Exception as err:  # 断网、镜像挂了等情况
        print(f"提示：MNIST 读取失败（{err}），改用小规模合成数据，数值和视频不同。")
        n_tr = train_limit or 600
        n_te = test_limit or 200
        (xtr, ttr), (xte, tte) = toy_digits(n_tr, n_te, flatten=flatten)
        source = "合成数据"
    if train_limit:
        xtr, ttr = xtr[:train_limit], ttr[:train_limit]
    if test_limit:
        xte, tte = xte[:test_limit], tte[:test_limit]
    return (xtr, ttr), (xte, tte), source
