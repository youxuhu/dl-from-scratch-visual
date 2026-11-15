"""从一排点到一条曲线：sin 和 cos。"""
import numpy as np

from _plotting import plt, save

x = np.arange(0, 7, 0.2)        # 采样点：0.0, 0.2, 0.4, ..., 6.8
y_sin = np.sin(x)
y_cos = np.cos(x)

n_points = len(x)
n_pos = int((y_sin > 0).sum())   # 正半周的点，画成正值色
n_neg = int((y_sin < 0).sum())   # 负半周的点，画成负值色


def draw():
    fig = plt.figure(figsize=(6, 3.5))
    plt.plot(x, y_sin, label="sin")
    plt.plot(x, y_cos, linestyle="--", label="cos")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("sin & cos")
    plt.legend()
    return save(fig, "sin_cos.png")


def main():
    print("x 的前 4 个值:", x[:4])
    print("采样点个数:", n_points)
    print("sin > 0 的点:", n_pos, " sin < 0 的点:", n_neg)
    print("x = 1.6 时 sin =", round(float(y_sin[8]), 4))
    print("图已保存:", draw())


if __name__ == "__main__":
    main()
