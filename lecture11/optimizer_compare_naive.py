"""第11讲：四种优化器在 f(x, y) = x²/20 + y² 上赛跑。

起点 (-7.0, 2.0)，学习率 SGD 0.95、Momentum 0.1、AdaGrad 1.5、Adam 0.3，各走 30 步。
同时打印前 3 步 Momentum 的速度 v、AdaGrad 的水位 h、Adam 的 m 和 v。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from common.optimizer import SGD, AdaGrad, Adam, Momentum  # noqa: E402

START = (-7.0, 2.0)
STEPS = 30
SETTINGS = {
    "SGD": (SGD, 0.95),
    "Momentum": (Momentum, 0.1),
    "AdaGrad": (AdaGrad, 1.5),
    "Adam": (Adam, 0.3),
}


def f(x, y):
    return x ** 2 / 20.0 + y ** 2


def grad_f(x, y):
    return x / 10.0, 2.0 * y


def race(name, steps=STEPS):
    """返回轨迹 (steps+1, 2) 和前 3 步的内部状态快照。"""
    cls, lr = SETTINGS[name]
    opt = cls(lr=lr)
    params = {"x": np.array(START[0]), "y": np.array(START[1])}
    path = [START]
    states = []
    for i in range(steps):
        gx, gy = grad_f(float(params["x"]), float(params["y"]))
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
    plt = __import__("_plot").get_pyplot()
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
    target = __import__("_plot").OUT / "optimizer_race.png"
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
