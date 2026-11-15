"""数值微分：前向差分 vs 中心差分，h 取多大合适。函数是自己选的 f(x) = 0.05x² - 0.3x。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from common.gradient import numerical_diff  # noqa: E402


def f(x):
    return 0.05 * x ** 2 - 0.3 * x


X0 = 4.0
TRUE_SLOPE = 0.1 * X0 - 0.3      # 解析导数 f'(x) = 0.1x - 0.3，在 x = 4 处是 0.1
HS = [1.0, 0.1, 0.01, 1e-4]


def forward_diff(fn, x, h):
    return (fn(x + h) - fn(x)) / h


table = [(h, forward_diff(f, X0, h), numerical_diff(f, X0, h)) for h in HS]

# h 太小：x + 1e-50 在浮点数里就等于 x，分子变成 0
tiny = 1e-50
tiny_result = forward_diff(f, X0, tiny)
tiny_float32 = float(np.float32(tiny))


def tangent_line(x):
    """x = 4 处的切线 y = f(4) + 0.1·(x - 4)。"""
    return f(X0) + TRUE_SLOPE * (x - X0)


def main():
    print("f(4) =", round(f(X0), 4), " 真实导数 f'(4) =", round(TRUE_SLOPE, 10))
    for h, fwd, ctr in table:
        print(f"h = {h:<6}  前向差分 {fwd:.6f}   中心差分 {ctr:.6f}")
    print("h = 1e-50：前向差分 =", tiny_result, "  np.float32(1e-50) =", tiny_float32)
    print("切线在 x = 0 处的值:", round(tangent_line(0.0), 4))


if __name__ == "__main__":
    main()
