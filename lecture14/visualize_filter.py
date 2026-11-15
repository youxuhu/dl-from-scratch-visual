"""第14讲：第 1 层滤波器的样子，以及一个竖边滤波器作用在手写数字上的效果。

如果 lecture14/output/ 里已有 train_convnet.py 存下的参数就直接读，否则当场用 Adam 训练一小会儿。
比较训练前后 30 个 5×5 滤波器的「平滑程度」（相邻格的差别），训练后滤波器更像边缘、斑块而不是噪点。
装了 matplotlib 时把 6×5 网格图存到 lecture14/output/filters.png。
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # 小矩阵用多线程反而慢，机器忙时差几十倍
import numpy as np  # noqa: E402

from common.layers import Convolution  # noqa: E402
from common.simple_convnet import SimpleConvNet  # noqa: E402
from common.trainer import Trainer  # noqa: E402
from common.util import get_mnist, get_pyplot, is_quick  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"
PARAMS = OUT / "simple_convnet_params.pkl"


def roughness(filters):
    """每个滤波器先缩放到 [-1, 1]，再算相邻格差值绝对值的平均。噪点大，边缘、斑块小。"""
    f = filters.reshape(filters.shape[0], 5, 5)
    f = f / (np.abs(f).max(axis=(1, 2), keepdims=True) + 1e-12)
    dx = np.abs(np.diff(f, axis=2)).mean(axis=(1, 2))
    dy = np.abs(np.diff(f, axis=1)).mean(axis=(1, 2))
    return (dx + dy) / 2


def edge_response(digit):
    """竖边滤波器和横边滤波器分别作用在一张图上，返回两张输出特征图。"""
    vertical = np.tile([1.0, 1.0, 0.0, -1.0, -1.0], (5, 1))
    horizontal = vertical.T.copy()
    W = np.stack([vertical, horizontal])[:, np.newaxis]   # (2, 1, 5, 5)
    out = Convolution(W, np.zeros(2)).forward(digit.reshape(1, 1, 28, 28).astype(float))[0]
    return np.abs(out[0]), np.abs(out[1])


def get_trained(net, quick):
    if PARAMS.exists():
        net.load_params(PARAMS)
        return "读取已保存的参数"
    (x_tr, t_tr), (x_te, t_te), _ = get_mnist(flatten=False, train_limit=1000 if quick else 10000,
                                              test_limit=200)
    Trainer(net, x_tr, t_tr, x_te, t_te, epochs=2 if quick else 5, mini_batch_size=100,
            optimizer="Adam", optimizer_param={"lr": 0.001}, verbose=False, seed=0).train()
    return "当场训练"


def main():
    net = SimpleConvNet(seed=0)
    before = net.params["W1"].copy()
    how = get_trained(net, is_quick())
    after = net.params["W1"].copy()
    print(f"滤波器形状 {after.shape}；训练后参数来源：{how}")
    print(f"相邻格差别（越小越平滑）：训练前平均 {roughness(before).mean():.3f}，训练后平均 {roughness(after).mean():.3f}")
    # 找一张数字 1，竖笔画多
    (_, _), (x_te, t_te), source = get_mnist(flatten=False, test_limit=1000)
    one = x_te[np.where(t_te == 1)[0][0], 0] if np.any(t_te == 1) else x_te[0, 0]
    v, h = edge_response(one)
    print(f"数据：{source}；数字 1 上竖边滤波器响应总和 {v.sum():.1f}，横边滤波器 {h.sum():.1f}")
    plt = get_pyplot(OUT)
    if plt is not None:
        fig, axes = plt.subplots(2, 1, figsize=(6, 9))
        for ax, filters, title in zip(axes, (before, after), ("before", "after")):
            grid = np.ones((6 * 6, 5 * 6)) * np.nan
            for k in range(30):
                r, c = divmod(k, 5)
                grid[r * 6:r * 6 + 5, c * 6:c * 6 + 5] = filters[k, 0]
            ax.imshow(grid, cmap="bwr")
            ax.set_title(title)
            ax.axis("off")
        fig.savefig(OUT / "filters.png")
        plt.close(fig)


if __name__ == "__main__":
    main()
