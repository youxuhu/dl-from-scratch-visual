"""一张画布分成 1×3：曲线、灰度图、损失曲线。subplot 的编号从 1 开始，按行数。"""
import array_image
import loss_curve
import sin_cos
from _plotting import plt, save

PANELS = ["sin", "grid", "loss"]


def draw():
    fig = plt.figure(figsize=(10, 3))
    plt.subplot(1, 3, 1)
    plt.plot(sin_cos.x, sin_cos.y_sin)
    plt.title("1: sin")
    plt.subplot(1, 3, 2)
    plt.imshow(array_image.grid, cmap="gray", vmin=0, vmax=1)
    plt.title("2: imshow")
    plt.subplot(1, 3, 3)
    plt.plot(loss_curve.smooth)
    plt.ylim(0, 2.6)
    plt.title("3: loss")
    n_axes = len(fig.axes)
    return save(fig, "subplots.png"), n_axes


def main():
    path, n_axes = draw()
    print("子图个数:", n_axes, "顺序:", PANELS)
    print("图已保存:", path)


if __name__ == "__main__":
    main()
