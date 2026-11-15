"""损失曲线预告：一条会抖的下降曲线 + 滑动平均。

这里的数据是用固定随机种子造出来的「模拟训练记录」，不是真的训练；
真正的训练曲线从第8讲开始由训练脚本生成。
"""
import numpy as np

from _plotting import plt, save

STEPS = 10000
WINDOW = 200

rng = np.random.default_rng(2026)
it = np.arange(STEPS)
loss = 0.15 + 2.15 * np.exp(-it / 1200) + rng.normal(0, 0.08, STEPS) * np.exp(-it / 4000)
loss = np.clip(loss, 0.01, None)


def moving_average(v, k):
    kernel = np.ones(k) / k
    return np.convolve(v, kernel, mode="valid")


smooth = moving_average(loss, WINDOW)


def draw():
    fig = plt.figure(figsize=(6, 3.5))
    plt.plot(it, loss, linewidth=0.5, alpha=0.5, label="loss")
    plt.plot(it[WINDOW - 1:], smooth, linewidth=2, label=f"moving avg ({WINDOW})")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.ylim(0, 2.6)
    plt.legend()
    return save(fig, "loss_curve.png")


def main():
    print("前 3 个 loss:", np.round(loss[:3], 3))
    print("滑动平均 起点 / 终点:", round(float(smooth[0]), 3), round(float(smooth[-1]), 3))
    print("图已保存:", draw())


if __name__ == "__main__":
    main()
