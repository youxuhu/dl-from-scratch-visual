"""第15讲：数据扩充。对一张手写数字做旋转、平移、裁剪后放大，生成一圈「新样本」。

全部用 NumPy 写：旋转用最近邻插值，裁剪后放大也用最近邻。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from common.util import get_mnist, get_pyplot  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"


def rotate(img, degrees):
    """绕图片中心旋转（最近邻插值），转出去的地方补 0。"""
    h, w = img.shape
    theta = np.deg2rad(degrees)
    cy, cx = (h - 1) / 2, (w - 1) / 2
    ys, xs = np.mgrid[0:h, 0:w]
    # 输出像素 (y, x) 从输入的哪个位置取：反向旋转
    src_x = np.cos(theta) * (xs - cx) + np.sin(theta) * (ys - cy) + cx
    src_y = -np.sin(theta) * (xs - cx) + np.cos(theta) * (ys - cy) + cy
    sx, sy = np.rint(src_x).astype(int), np.rint(src_y).astype(int)
    ok = (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
    out = np.zeros_like(img)
    out[ok] = img[sy[ok], sx[ok]]
    return out


def shift(img, dy, dx):
    out = np.zeros_like(img)
    h, w = img.shape
    out[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = \
        img[max(-dy, 0):h - max(dy, 0), max(-dx, 0):w - max(dx, 0)]
    return out


def crop_resize(img, top, left, size):
    """裁出 size×size 的一块，再用最近邻放大回原尺寸。"""
    piece = img[top:top + size, left:left + size]
    idx = (np.arange(img.shape[0]) * size / img.shape[0]).astype(int)
    return piece[np.ix_(idx, idx)]


def augment(img):
    return {
        "旋转+15°": rotate(img, 15), "旋转-15°": rotate(img, -15),
        "右移3": shift(img, 0, 3), "左移3": shift(img, 0, -3),
        "上移3": shift(img, -3, 0), "下移3": shift(img, 3, 0),
        "裁剪放大": crop_resize(img, 3, 3, 22),
    }


def main():
    _, (x_test, t_test), source = get_mnist(flatten=False, test_limit=10)
    img = x_test[0, 0]
    variants = augment(img)
    print(f"数据：{source}；原图标签 {t_test[0]}，亮度总和 {img.sum():.1f}")
    for name, v in variants.items():
        same = np.mean((v > 0.5) == (img > 0.5))
        print(f"  {name}：形状 {v.shape}，亮度总和 {v.sum():.1f}，和原图逐像素相同的比例 {same:.1%}")
    plt = get_pyplot(OUT)
    if plt is not None:
        fig, axes = plt.subplots(1, len(variants) + 1, figsize=(12, 2))
        for ax, (name, v) in zip(axes, [("原图", img)] + list(variants.items())):
            ax.imshow(v, cmap="gray_r")
            ax.axis("off")
        fig.savefig(OUT / "augmentation.png")
        plt.close(fig)


if __name__ == "__main__":
    main()
