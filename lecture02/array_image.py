"""数组就是图片：imshow 把数值变成灰度方块。"""
import os

import numpy as np

from _plotting import OUT_DIR, plt, save

# 自己设计的 5×5 数组，取值 0 到 1，中间一个亮十字
grid = np.array([
    [0.0, 0.1, 0.8, 0.1, 0.0],
    [0.1, 0.3, 0.9, 0.3, 0.1],
    [0.8, 0.9, 1.0, 0.9, 0.8],
    [0.1, 0.3, 0.9, 0.3, 0.1],
    [0.0, 0.1, 0.8, 0.1, 0.0],
])
brightest = tuple(int(i) for i in np.unravel_index(grid.argmax(), grid.shape))
n_bright = int((grid >= 0.8).sum())


def draw_grid():
    fig = plt.figure(figsize=(3, 3))
    plt.imshow(grid, cmap="gray", vmin=0, vmax=1)
    plt.title("5x5 grid")
    return save(fig, "grid.png")


def roundtrip_png():
    """先把数组存成 png，再用 imread 读回来，看看读到的是什么。"""
    path = OUT_DIR / "grid_raw.png"
    OUT_DIR.mkdir(exist_ok=True)
    plt.imsave(path, grid, cmap="gray", vmin=0, vmax=1)
    img = plt.imread(path)
    return img


def first_digit():
    """有 MNIST 缓存时，顺便显示测试集第一张图（28×28）。没有就跳过，不联网。"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from common import mnist
    if not mnist.CACHE.exists() and os.environ.get("QUICK") == "1":
        return None
    (_, _), (x_test, t_test) = mnist.load_mnist(flatten=False)
    fig = plt.figure(figsize=(3, 3))
    plt.imshow(x_test[0, 0], cmap="gray")
    plt.title(f"label = {t_test[0]}")
    save(fig, "digit.png")
    return int(t_test[0])


def main():
    print("最亮的格子:", brightest, "值 =", grid[brightest])
    print("值 >= 0.8 的格子个数:", n_bright)
    img = roundtrip_png()
    print("imread 读回来的形状:", img.shape, "（RGBA 四个通道）")
    print("图已保存:", draw_grid())
    label = first_digit()
    if label is not None:
        print("MNIST 测试集第一张图的标签:", label)


if __name__ == "__main__":
    main()
