"""第11讲：四种优化器在 f(x, y) = x²/20 + y² 上赛跑。

起点 (-7.0, 2.0)，学习率 SGD 0.95、Momentum 0.1、AdaGrad 1.5、Adam 0.3，各走 30 步。
同时打印前 3 步 Momentum 的速度 v、AdaGrad 的水位 h、Adam 的 m 和 v。

改编自《深度学习入门：基于Python的理论与实现》官方源码 ch06/optimizer_compare_naive.py
（MIT 许可，Copyright (c) 2016 Koki Saitoh）。函数名 f、df，起点 init_pos、学习率和步数与原书一致；另记录前 3 步的内部状态。注释改写成中文，保留中间结果打印和 QUICK 模式。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from common.optimizer import SGD, AdaGrad, Adam, Momentum  # noqa: E402
from common.util import get_pyplot  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"

init_pos = (-7.0, 2.0)
STEPS = 30  # 原书同样走 30 步
SETTINGS = {
    "SGD": (SGD, 0.95),
    "Momentum": (Momentum, 0.1),
    "AdaGrad": (AdaGrad, 1.5),
    "Adam": (Adam, 0.3),
}


def f(x, y):
    return x ** 2 / 20.0 + y ** 2


def df(x, y):
    return x / 10.0, 2.0 * y


def race(name, steps=STEPS):
    """返回轨迹 (steps+1, 2) 和前 3 步的内部状态快照。"""
    cls, lr = SETTINGS[name]
    opt = cls(lr=lr)
    params = {"x": np.array(init_pos[0]), "y": np.array(init_pos[1])}
    path = [init_pos]
    states = []
    for i in range(steps):
        gx, gy = df(float(params["x"]), float(params["y"]))
        opt.update(params, {"x": np.array(gx), "y": np.array(gy)})
        path.append((float(params["x"]), float(params["y"])))
        if i < 3:
            snap = {}
            for attr in ("v", "h", "m"):
                store = getattr(opt, attr, None)
                if store:
                    snap[attr] = (float(store["x"]), float(store["y"]))
            states.append(snap)
    return np.array(path), states


paths = {}
first_states = {}
for _name in SETTINGS:
    paths[_name], first_states[_name] = race(_name)


def draw():
    plt = get_pyplot(OUT)
    if plt is None:
        return None
    xs, ys = np.meshgrid(np.linspace(-10, 10, 200), np.linspace(-5, 5, 100))
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    for ax, (name, path) in zip(axes.ravel(), paths.items()):
        ax.contour(xs, ys, f(xs, ys), levels=12, linewidths=0.6)
        ax.plot(path[:, 0], path[:, 1], "o-", ms=3)
        ax.plot(0, 0, "+", ms=10)
        ax.set_title(name)
    fig.tight_layout()
    target = OUT / "optimizer_race.png"
    fig.savefig(target)
    plt.close(fig)
    return target


def main():
    for name, path in paths.items():
        x, y = path[-1]
        print(f"{name:>8}: 30 步后到 ({x:.4f}, {y:.4f})，f = {f(x, y):.6f}")
    for name, states in first_states.items():
        for i, s in enumerate(states, start=1):
            text = "，".join(f"{k} = ({a:.4f}, {b:.4f})" for k, (a, b) in s.items())
            if text:
                print(f"{name} 第 {i} 步：{text}")
    target = draw()
    if target:
        print("图已保存：", target)


if __name__ == "__main__":
    main()
