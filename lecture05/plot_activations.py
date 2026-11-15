"""三条曲线画在同一张图上，存到 output/。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common.functions import relu, sigmoid, step_function  # noqa: E402

xs = np.linspace(-6.0, 6.0, 121)   # -6.0, -5.9, ..., 6.0
curves = {"step": step_function(xs), "sigmoid": sigmoid(xs), "relu": relu(xs)}


def draw():
    out = Path(__file__).resolve().parent / "output"
    out.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(6, 4))
    for name, ys in curves.items():
        plt.plot(xs, ys, label=name)
    plt.ylim(-0.2, 2.0)
    plt.legend()
    plt.title("step / sigmoid / ReLU")
    path = out / "activations.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def main():
    print("采样点个数:", len(xs))
    print("sigmoid 在 x = -6 和 x = 6 处:", round(float(curves["sigmoid"][0]), 4),
          round(float(curves["sigmoid"][-1]), 4))
    print("图已保存:", draw())


if __name__ == "__main__":
    main()
