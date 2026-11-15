"""第11讲：权重初始值不同，隐藏层激活值的分布就不同。

1000 个随机输入，每个 100 维，穿过 5 个隐藏层（每层 100 个神经元，不带偏置）。
sigmoid 试 标准差 1、0.01、Xavier；ReLU 试 0.01、Xavier、He。打印每层的统计量和直方图计数。

改编自《深度学习入门：基于Python的理论与实现》官方源码 ch06/weight_init_activation_histogram.py
（MIT 许可，Copyright (c) 2016 Koki Saitoh）。实验设置（1000 个 100 维输入、5 层、每层 100 个神经元）与原书一致；
注释改写成中文，另把几种初始值放在一起比较并打印统计量。
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # 小矩阵用多线程反而慢，机器忙时差几十倍
import numpy as np  # noqa: E402

from common.functions import relu, sigmoid  # noqa: E402
from common.util import get_pyplot  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"

NODES = 100
DEPTH = 5
SAMPLES = 1000

CASES = {
    ("sigmoid", "std=1"): (sigmoid, 1.0),
    ("sigmoid", "std=0.01"): (sigmoid, 0.01),
    ("sigmoid", "Xavier"): (sigmoid, np.sqrt(1.0 / NODES)),
    ("relu", "std=0.01"): (relu, 0.01),
    ("relu", "Xavier"): (relu, np.sqrt(1.0 / NODES)),
    ("relu", "He"): (relu, np.sqrt(2.0 / NODES)),
}


def activations(act, scale, seed=0):
    """返回 5 层的激活值列表，每个形状 (1000, 100)。同一个 seed 下各情况的随机数相同。"""
    rng = np.random.default_rng(seed)
    h = rng.standard_normal((SAMPLES, NODES))
    layers = []
    for _ in range(DEPTH):
        W = rng.standard_normal((NODES, NODES)) * scale
        h = act(np.dot(h, W))
        layers.append(h)
    return layers


def layer_stats(acts):
    """每层的 平均值、标准差、贴近 0 的比例(<0.01)、贴近 1 的比例(>0.99)。"""
    rows = []
    for a in acts:
        rows.append({"mean": float(a.mean()), "std": float(a.std()),
                     "near0": float(np.mean(a < 0.01)), "near1": float(np.mean(a > 0.99))})
    return rows


results = {key: layer_stats(activations(act, scale)) for key, (act, scale) in CASES.items()}


def histogram_counts(acts, bins=10):
    return [np.histogram(a, bins=bins, range=(0, 1))[0].tolist() for a in acts]


def main():
    for (act, label), rows in results.items():
        print(f"\n{act} + {label}")
        for i, r in enumerate(rows, start=1):
            print(f"  第{i}层：均值 {r['mean']:.3f}  标准差 {r['std']:.3f}  "
                  f"<0.01 占 {r['near0']:.1%}  >0.99 占 {r['near1']:.1%}")
    counts = histogram_counts(activations(sigmoid, 1.0))
    print("\nsigmoid + std=1 第1层直方图（[0,1] 分 10 格）：", counts[0])
    plt = get_pyplot(OUT)
    if plt is not None:
        fig, axes = plt.subplots(len(CASES), DEPTH, figsize=(12, 12))
        for row, (key, (act, scale)) in zip(axes, CASES.items()):
            for i, (ax, a) in enumerate(zip(row, activations(act, scale))):
                ax.hist(a.ravel(), bins=30, range=(0, 1))
                ax.set_title(f"{key[0]} {key[1]} L{i + 1}", fontsize=7)
                ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(OUT / "activation_histogram.png")
        plt.close(fig)


if __name__ == "__main__":
    main()
