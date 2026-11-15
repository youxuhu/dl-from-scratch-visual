"""为什么不拿精度当学习的指标：慢慢拧一个权重，精度是一级一级的台阶，损失却是平滑的。

用第6讲的网络（common/simple_net.py）和前 200 张测试图，只改输出层的 W3[0, 7] 这一个数。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from common.functions import cross_entropy_error  # noqa: E402
from common.mnist import load_mnist  # noqa: E402
from common.simple_net import init_network, predict  # noqa: E402

N = 200
_, (x_test, t_test) = load_mnist(normalize=True)
x, t = x_test[:N], t_test[:N]
net = init_network()
w0 = float(net["W3"][0, 7])
offsets = np.linspace(-6.0, 6.0, 121)


def scan():
    accs, losses = [], []
    for d in offsets:
        net["W3"][0, 7] = w0 + d
        y = predict(net, x)
        accs.append(float(np.mean(np.argmax(y, axis=1) == t)))
        losses.append(cross_entropy_error(y, t))
    net["W3"][0, 7] = w0
    return np.array(accs), np.array(losses)


accs, losses = scan()


def slopes_at_zero(h=1e-3):
    """在原来的权重处，精度和损失对这个权重的数值导数。"""
    vals = {}
    for sign in (+1, -1):
        net["W3"][0, 7] = w0 + sign * h
        y = predict(net, x)
        vals[sign] = (float(np.mean(np.argmax(y, axis=1) == t)), cross_entropy_error(y, t))
    net["W3"][0, 7] = w0
    d_acc = (vals[1][0] - vals[-1][0]) / (2 * h)
    d_loss = (vals[1][1] - vals[-1][1]) / (2 * h)
    return d_acc, d_loss


def main():
    print("原来的 W3[0, 7] =", round(w0, 4))
    print(f"扫了 {len(offsets)} 个值，精度只出现了 {len(np.unique(accs))} 种：", np.unique(accs))
    print(f"损失出现了 {len(np.unique(np.round(losses, 6)))} 种，从 {losses.min():.4f} 到 {losses.max():.4f}")
    d_acc, d_loss = slopes_at_zero()
    print("在原值处的斜率：精度", d_acc, " 损失", round(d_loss, 5))


if __name__ == "__main__":
    main()
