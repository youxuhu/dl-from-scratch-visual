"""交叉熵只看正确类别的概率 p：损失是 -log(p)。p 越小，惩罚越大。"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

probs = np.array([0.9, 0.6, 0.3, 0.1])
penalty = -np.log(probs)


def draw():
    out = Path(__file__).resolve().parent / "output"
    out.mkdir(exist_ok=True)
    p = np.linspace(0.01, 1.0, 100)
    fig = plt.figure(figsize=(5, 3.5))
    plt.plot(p, -np.log(p))
    plt.scatter(probs, penalty)
    plt.xlabel("p (correct class)")
    plt.ylabel("-log p")
    path = out / "cross_entropy.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def main():
    for p, e in zip(probs, penalty):
        print(f"p = {p}:  -log p = {e:.3f}")
    print("图已保存:", draw())


if __name__ == "__main__":
    main()
