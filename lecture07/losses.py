"""均方误差和交叉熵误差：同一个标签，一个猜对的输出和一个猜错的输出。

损失函数本体在 common/functions.py。鱼书函数名是 sum_squared_error（单个样本时和 mean_squared_error 数值相同）和 cross_entropy_error。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from common.functions import (  # noqa: E402
    cross_entropy_error, mean_squared_error, sum_squared_error)

# 正确答案是 3，写成 one hot
t = np.zeros(10)
t[3] = 1.0

# 猜对的输出：第 3 格概率最大
y_good = np.array([0.02, 0.05, 0.03, 0.70, 0.04, 0.06, 0.02, 0.03, 0.03, 0.02])
# 猜错的输出：第 8 格概率最大
y_bad = np.array([0.02, 0.05, 0.03, 0.10, 0.04, 0.06, 0.02, 0.03, 0.63, 0.02])

diff_good = y_good - t
sq_good = diff_good ** 2
mse_good = mean_squared_error(y_good, t)
mse_bad = mean_squared_error(y_bad, t)
ce_good = cross_entropy_error(y_good, t)
ce_bad = cross_entropy_error(y_bad, t)


def main():
    print("y_good 的和 =", round(float(y_good.sum()), 6), " y_bad 的和 =", round(float(y_bad.sum()), 6))
    print("y_good - t =", np.round(diff_good, 2))
    print("平方 =", np.round(sq_good, 4))
    print("平方和 =", round(float(sq_good.sum()), 4), " 乘 0.5 =", round(mse_good, 4))
    print("均方误差：猜对", round(mse_good, 4), " 猜错", round(mse_bad, 4))
    print("sum_squared_error（鱼书函数名，单样本）猜对", round(float(sum_squared_error(y_good, t)), 4),
          " 猜错", round(float(sum_squared_error(y_bad, t)), 4))
    print("交叉熵：猜对", round(ce_good, 4), " 猜错", round(ce_bad, 4))
    print("用标签下标 3 代替 one hot，交叉熵一样:", round(cross_entropy_error(y_good, 3), 4))


if __name__ == "__main__":
    main()
