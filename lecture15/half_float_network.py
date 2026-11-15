"""第15讲：运算精度的位数缩减。把 DeepConvNet 的参数和输入换成 16 位浮点数，看精度变不变。

做法：参数和测试图都舍入成 float16，每一层的输出也立刻舍入成 float16（层内的乘加用 float32 算，
因为 NumPy 的 float16 矩阵乘法没有加速，非常慢）。这模拟的是「数据用半精度存储和传递」。
QUICK=1 只用前 300 张测试图。

改编自《深度学习入门：基于Python的理论与实现》官方源码 ch08/half_float_network.py
（MIT 许可，Copyright (c) 2016 Koki Saitoh）。思路与原书一致（把参数和输入换成 float16 后比较精度）；原书只把参数转成 float16 再用 NumPy 算，这里额外把每层输出也舍入成 float16，注释改写成中文。
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # 小矩阵用多线程反而慢，机器忙时差几十倍
import numpy as np  # noqa: E402

from _trained import trained_deepnet  # noqa: E402
from common.layers import Dropout  # noqa: E402
from common.util import get_mnist, is_quick  # noqa: E402

BITS = {np.float64: 64, np.float32: 32, np.float16: 16}


def accuracy_with_dtype(net, x, t, dtype, batch_size=100):
    """参数、输入、每层输出都舍入到 dtype 后的精度。算完把参数恢复原样。"""
    saved = {k: v.copy() for k, v in net.params.items()}
    for v in net.params.values():
        v[...] = v.astype(dtype)          # 数组本身仍是 float64，但数值已经是 dtype 能表示的值
    hits = 0
    for i in range(0, len(x), batch_size):
        h = x[i:i + batch_size].astype(dtype)
        for layer in net.layers:
            inp = h.astype(np.float32) if dtype != np.float64 else h
            h = layer.forward(inp, False) if isinstance(layer, Dropout) else layer.forward(inp)
            h = h.astype(dtype)
        hits += int(np.sum(h.argmax(axis=1) == t[i:i + batch_size]))
    for k, v in saved.items():
        net.params[k][...] = v
    return hits / len(x)


def main():
    quick = is_quick()
    net, how = trained_deepnet(quick)
    _, (x_test, t_test), source = get_mnist(flatten=False, test_limit=300 if quick else None)
    print(f"参数：{how}；数据：{source} {len(x_test)} 张")
    for dtype in (np.float64, np.float16):
        acc = accuracy_with_dtype(net, x_test, t_test, dtype)
        print(f"{BITS[dtype]} 位浮点数：精度 {acc:.4f}")
    print("float16 能表示的最小正规数", float(np.finfo(np.float16).tiny), "，有效数字约 3 位")


if __name__ == "__main__":
    main()
