"""第10讲：ReLU 层和 Sigmoid 层的正向、反向。

ReLU 用一张 mask 记住哪些位置被挡住；Sigmoid 拆成 ×(-1)、exp、+1、倒数 四个节点，
逐节点回传后合并成 dout · y(1 - y)。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from common.layers import Relu, Sigmoid  # noqa: E402

# ReLU：2×2 的输入，以及从上游传回来的梯度
relu_in = np.array([[1.0, -0.5], [-2.0, 3.0]])
relu_dout = np.array([[0.4, 0.8], [-1.2, 0.6]])
relu = Relu()
relu_out = relu.forward(relu_in)
relu_mask = relu.mask.copy()
relu_dx = relu.backward(relu_dout)


def sigmoid_by_nodes(x, dout=1.0):
    """按计算图的四个节点算 sigmoid，返回正向各节点值和反向各节点的梯度。"""
    a = -x               # ×(-1) 节点
    e = np.exp(a)        # exp 节点
    c = 1.0 + e          # +1 节点
    y = 1.0 / c          # 倒数节点
    grad_c = dout * (-1.0 / c ** 2)   # 倒数的局部导数 -1/c²
    grad_e = grad_c * 1.0             # +1 原样传
    grad_a = grad_e * e               # exp 的导数还是 exp(a)
    grad_x = grad_a * -1.0            # ×(-1) 节点乘 -1
    forward_vals = {"-x": a, "exp(-x)": e, "1+exp(-x)": c, "y": y}
    backward_vals = {"d(1+exp)": grad_c, "d(exp)": grad_e, "d(-x)": grad_a, "dx": grad_x}
    return forward_vals, backward_vals


sig_x = 1.0
sig_forward, sig_backward = sigmoid_by_nodes(sig_x)
sig_y = sig_forward["y"]
sig_simple = 1.0 * sig_y * (1.0 - sig_y)   # 化简后的 dout · y(1 - y)

sig_layer = Sigmoid()
sig_layer.forward(np.array([sig_x]))
sig_layer_dx = float(sig_layer.backward(np.array([1.0]))[0])


def main():
    print("ReLU 输入：\n", relu_in)
    print("mask = (x <= 0)：\n", relu_mask)
    print("正向输出：\n", relu_out)
    print("上游梯度：\n", relu_dout)
    print("反向输出（mask 处置 0）：\n", relu_dx)
    print(f"\nSigmoid，x = {sig_x}")
    for k, v in sig_forward.items():
        print(f"  正向 {k:>10} = {v:.6f}")
    for k, v in sig_backward.items():
        print(f"  反向 {k:>10} = {v:.6f}")
    print(f"  化简公式 y(1-y) = {sig_simple:.6f}，Sigmoid 层 backward = {sig_layer_dx:.6f}")


if __name__ == "__main__":
    main()
