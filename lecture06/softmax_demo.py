"""softmax：先 exp，再除以总和；输入太大会溢出，减去最大值就好了。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from common.functions import softmax  # noqa: E402

a = np.array([1.2, 3.1, 0.4])
exp_a = np.exp(a)
total = exp_a.sum()
p = exp_a / total


def naive_softmax(v):
    e = np.exp(v)
    return e / e.sum()


big = np.array([1500.0, 1498.0, 1495.0])


def overflow_demo():
    with np.errstate(over="ignore", invalid="ignore"):
        bad = naive_softmax(big)
    shifted = big - big.max()      # [0, -2, -5]
    good = naive_softmax(shifted)
    return bad, shifted, good


def main():
    print("a =", a)
    print("exp(a) =", np.round(exp_a, 3), " 总和 =", round(float(total), 3))
    print("softmax(a) =", np.round(p, 3), " 和 =", round(float(p.sum()), 6))
    print("common 里的 softmax 结果一样:", np.allclose(softmax(a), p))
    print("argmax 不变:", int(np.argmax(a)), int(np.argmax(p)))
    bad, shifted, good = overflow_demo()
    print("直接算 [1500, 1498, 1495]:", bad)
    print("减去最大值后:", shifted, " softmax =", np.round(good, 3))


if __name__ == "__main__":
    main()
