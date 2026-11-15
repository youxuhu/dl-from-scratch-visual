"""第12讲：Batch Normalization 一步一步算。

一个 mini-batch 有 4 个样本，某个神经元的输入是 [3.0, -1.0, 5.0, 1.0]：
求均值 → 求方差 → 开根号 → 减均值再除以标准差 → 乘 gamma 加 beta。
最后用 common/layers.py 的 BatchNormalization 层再算一遍对照。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from common.layers import BatchNormalization  # noqa: E402

x = np.array([3.0, -1.0, 5.0, 1.0])
mu = x.mean()                      # 2.0
xc = x - mu                        # 到均值的距离
var = np.mean(xc ** 2)             # 5.0
std = np.sqrt(var + 10e-7)         # 约 2.236
xn = xc / std                      # 标准化后
gamma, beta = 2.0, 0.5
out = gamma * xn + beta            # 再拉伸、平移

layer = BatchNormalization(np.array([gamma]), np.array([beta]))
layer_out = layer.forward(x.reshape(4, 1), train_flg=True).ravel()
running_mean_after_one = float(layer.running_mean[0])   # 0.9*0 + 0.1*2.0
running_var_after_one = float(layer.running_var[0])     # 0.9*0 + 0.1*5.0


def main():
    print("输入：", x)
    print("第一步 均值 mu =", mu)
    print("第二步 xc = x - mu =", xc, "，方差 var =", var)
    print(f"第三步 std = sqrt(var + 10e-7) = {std:.4f}")
    print("第四步 xn = xc / std =", np.round(xn, 3))
    print(f"第五步 gamma = {gamma}，beta = {beta}，输出 =", np.round(out, 3))
    print("BatchNormalization 层的输出 =", np.round(layer_out, 3))
    print(f"一批之后 running_mean = {running_mean_after_one:.3f}，running_var = {running_var_after_one:.3f}")


if __name__ == "__main__":
    main()
