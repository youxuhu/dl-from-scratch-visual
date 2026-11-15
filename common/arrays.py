"""数组相关的共用函数（独立实现）。"""
import numpy as np


def describe(arr):
    """返回形如 'shape=(2, 3) dtype=int64' 的一行说明，代码面板行尾就显示它。"""
    arr = np.asarray(arr)
    return f"shape={arr.shape} dtype={arr.dtype}"


def _fmt(v):
    v = float(v)
    return str(int(v)) if v.is_integer() else f"{v:g}"


def _term(v):
    s = _fmt(v)
    return f"({s})" if s.startswith("-") else s


def matmul_steps(A, B):
    """把 np.dot(A, B) 的每一格展开成算式字符串。

    返回列表，元素是 (i, j, 算式)，例如 (0, 0, '1×3 + 0×2 + 2×1 = 5')。
    视频里逐格展开的动画就照这个顺序播放。
    """
    A = np.asarray(A)
    B = np.asarray(B)
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[0]:
        raise ValueError(f"形状对不上：{A.shape} 和 {B.shape}")
    C = np.dot(A, B)
    steps = []
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            terms = " + ".join(f"{_term(a)}×{_term(b)}" for a, b in zip(A[i], B[:, j]))
            steps.append((i, j, f"{terms} = {_fmt(C[i, j])}"))
    return steps
