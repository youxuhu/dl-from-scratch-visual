"""第14讲：把第 1 层学到的滤波器作用在一张图上，看各自对什么有反应。

改编自《深度学习入门：基于Python的理论与实现》官方源码 ch07/apply_filter.py
（MIT 许可，Copyright (c) 2016 Koki Saitoh）。做法与原书一致：取 W1 的前 16 个滤波器，偏置设为 0，
各自建一个 Convolution 层作用在一张灰度图上，函数名 filter_show 也一样。
和原书的不同：原书用 lena_gray.png，本仓库不带这张图，改用 MNIST 测试集的第一张图；
参数优先读 train_convnet.py 存下的，没有就和 visualize_filter.py 一样当场训练一小会儿。
所以输出的响应强度取决于用的是哪份参数，README 只列形状。
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # 小矩阵用多线程反而慢，机器忙时差几十倍
import numpy as np  # noqa: E402

from common.layers import Convolution  # noqa: E402
from common.simple_convnet import SimpleConvNet  # noqa: E402
from common.util import get_mnist, get_pyplot, is_quick  # noqa: E402
from visualize_filter import get_trained  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"


def filter_show(filters, nx=4, show_num=16, name="apply_filter_W1.png"):
    """把前 show_num 个滤波器画成 nx 列的网格（需要 matplotlib，没有就跳过）。"""
    plt = get_pyplot(OUT)
    if plt is None:
        return None
    ny = int(np.ceil(show_num / nx))
    fig, axes = plt.subplots(ny, nx, figsize=(nx, ny))
    for i, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if i < show_num:
            ax.imshow(filters[i, 0], cmap="gray_r", interpolation="nearest")
    fig.savefig(OUT / name)
    plt.close(fig)
    return OUT / name


def apply_filters(network, img, show_num=16):
    """每个滤波器单独建 Convolution 层（偏置 0）作用在 img 上，返回 (show_num, H, W) 的输出。"""
    outs = []
    for i in range(show_num):
        w = network.params["W1"][i]
        w = w.reshape(1, *w.shape)
        conv_layer = Convolution(w, np.zeros(1))
        out = conv_layer.forward(img)
        outs.append(out.reshape(out.shape[2], out.shape[3]))
    return np.array(outs)


def main():
    network = SimpleConvNet(input_dim=(1, 28, 28),
                            conv_param={"filter_num": 30, "filter_size": 5, "pad": 0, "stride": 1},
                            hidden_size=100, output_size=10, weight_init_std=0.01, seed=0)
    how = get_trained(network, is_quick())
    _, (x_test, t_test), source = get_mnist(flatten=False, test_limit=1)
    img = x_test[:1].astype(float)          # (1, 1, 28, 28)
    outs = apply_filters(network, img)
    print(f"参数：{how}；图像：{source} 测试集第一张（标签 {int(t_test[0])}），形状 {img.shape}")
    print(f"16 个滤波器各自的输出形状 {outs.shape[1:]}，合起来 {outs.shape}")
    strength = np.abs(outs).mean(axis=(1, 2))
    print("各滤波器平均响应强度：", np.round(strength, 3))
    target = filter_show(network.params["W1"])
    plt = get_pyplot(OUT)
    if plt is not None:
        fig, axes = plt.subplots(4, 4, figsize=(4, 4))
        for ax, out in zip(axes.ravel(), outs):
            ax.imshow(out, cmap="gray_r", interpolation="nearest")
            ax.axis("off")
        fig.savefig(OUT / "apply_filter_out.png")
        plt.close(fig)
        print("图已保存：", target, OUT / "apply_filter_out.png")


if __name__ == "__main__":
    main()
