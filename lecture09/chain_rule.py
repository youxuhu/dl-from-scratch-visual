"""第9讲：链式法则。z = t²，t = x + y，求 ∂z/∂x。

代数上 ∂z/∂x = ∂z/∂t · ∂t/∂x = 2t · 1；在计算图上就是把两个局部导数相乘。
这里取 x = 3，y = 1，用解析式和第8讲的数值微分 numerical_diff 各算一遍。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.gradient import numerical_diff  # noqa: E402

x, y = 3.0, 1.0
t = x + y            # 4
z = t ** 2           # 16

dz_dt = 2 * t        # 8，平方节点的局部导数
dt_dx = 1.0          # 加法节点的局部导数
dz_dx = dz_dt * dt_dx


def numeric_dz_dx():
    return numerical_diff(lambda v: (v + y) ** 2, x)


def main():
    print(f"正向：t = x + y = {t}，z = t² = {z}")
    print(f"局部导数：∂z/∂t = 2t = {dz_dt}，∂t/∂x = {dt_dx}")
    print(f"链式法则：∂z/∂x = {dz_dt} × {dt_dx} = {dz_dx}")
    print(f"数值微分核对：{numeric_dz_dx():.6f}")


if __name__ == "__main__":
    main()
