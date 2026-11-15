"""三种激活函数：阶跃、sigmoid、ReLU。函数本体在 common/functions.py。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from common.functions import relu, sigmoid, step_function  # noqa: E402


def step_scalar(x):
    """只认一个实数的阶跃函数：传数组进来会报错。"""
    if x > 0:
        return 1
    return 0


# 支持数组的阶跃函数，拆成两步看
x = np.array([-1.5, 0.0, 2.5])
mask = x > 0              # [False False  True]
y_step = mask.astype(int)  # [0 0 1]

# 同一组 x 下三个函数的输出
probe = np.array([-2.0, -0.5, 0.0, 1.0, 3.0])
table = {
    "step": step_function(probe),
    "sigmoid": np.round(sigmoid(probe), 3),
    "relu": relu(probe),
}


def step_scalar_on_array():
    try:
        step_scalar(x)
    except ValueError as e:
        return type(e).__name__
    return "no error"


def main():
    print("step_scalar(2.5) =", step_scalar(2.5))
    print("step_scalar(数组) ->", step_scalar_on_array())
    print("x > 0 =", mask)
    print("(x > 0).astype(int) =", y_step)
    print("probe =", probe)
    for name, v in table.items():
        print(f"{name:8s}", v)


if __name__ == "__main__":
    main()
