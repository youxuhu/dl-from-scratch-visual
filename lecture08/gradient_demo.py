"""偏导数、梯度、梯度下降。函数 f(x0, x1) = x0² + 2·x1²（一个椭圆形的碗）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common.gradient import gradient_descent, numerical_gradient  # noqa: E402


def bowl(x):
    return x[0] ** 2 + 2 * x[1] ** 2


# 偏导数：在 (2.0, -1.0) 处，∂f/∂x0 = 2·x0 = 4，∂f/∂x1 = 4·x1 = -4
point = np.array([2.0, -1.0])
grad_at_point = numerical_gradient(bowl, point.copy())

START = np.array([-2.0, 3.0])
STEPS = 50
runs = {lr: gradient_descent(bowl, START, lr=lr, step_num=STEPS) for lr in (0.1, 0.6, 1e-4)}


def gradient_field(n=5):
    """5×5 网格上每个点的负梯度（箭头方向）。"""
    xs = np.linspace(-2.0, 2.0, n)
    pts, arrows = [], []
    for a in xs:
        for b in xs:
            p = np.array([a, b])
            pts.append(p)
            arrows.append(-numerical_gradient(bowl, p.copy()))
    return np.array(pts), np.array(arrows)


def draw():
    out = Path(__file__).resolve().parent / "output"
    out.mkdir(exist_ok=True)
    pts, arrows = gradient_field(9)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].quiver(pts[:, 0], pts[:, 1], arrows[:, 0], arrows[:, 1])
    axes[0].set_title("-grad f")
    _, path = runs[0.1]
    axes[1].plot(path[:, 0], path[:, 1], "o-", markersize=3)
    axes[1].set_xlim(-3, 3)
    axes[1].set_ylim(-3.5, 3.5)
    axes[1].set_title("gradient descent, lr = 0.1")
    path_png = out / "gradient.png"
    fig.savefig(path_png, dpi=100)
    plt.close(fig)
    return path_png


def main():
    print("在 (2.0, -1.0) 处的梯度:", np.round(grad_at_point, 6))
    pts, arrows = gradient_field()
    print("网格点 (2, 2) 处的负梯度:", np.round(arrows[-1], 4))
    for lr, (end, path) in runs.items():
        print(f"lr = {lr}: 第 1 步走到 {np.round(path[1], 4)}，{STEPS} 步后在 {end}")
    print("图已保存:", draw())


if __name__ == "__main__":
    main()
