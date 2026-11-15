"""几何视角：画出每个门的分界线 w1·x1 + w2·x2 + b = 0，存到 output/。"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from gates import INPUTS, PARAMS, perceptron  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"


def line_points(gate):
    """分界线和 x1 = 0、x1 = 1.5 的交点（w2 不为 0）。"""
    w, b = PARAMS[gate]
    xs = np.array([0.0, 1.5])
    ys = np.round(-(w[0] * xs + b) / w[1], 2) + 0.0
    return xs, ys


def draw():
    OUT.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    for ax, gate in zip(axes, PARAMS):
        for x1, x2 in INPUTS:
            y = perceptron(gate, x1, x2)
            ax.scatter(x1, x2, s=80, marker="o" if y else "x")
        xs, ys = line_points(gate)
        ax.plot(xs, ys)
        ax.set_xlim(-0.3, 1.3)
        ax.set_ylim(-0.3, 1.3)
        ax.set_title(gate)
    path = OUT / "decision_lines.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def main():
    for gate in PARAMS:
        xs, ys = line_points(gate)
        print(f"{gate}: 分界线过 ({xs[0]}, {ys[0]:.2f}) 和 ({xs[1]}, {ys[1]:.2f})")
    print("图已保存:", draw())


if __name__ == "__main__":
    main()
