"""第12讲：Dropout 每次正向随机熄灭一部分神经元，像在训练很多个不同的子网络。

一个隐藏层 10 个神经元，dropout_ratio = 0.5，连续正向 5 次看 mask；测试时全部点亮，输出乘 0.5。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from common.layers import Dropout  # noqa: E402

hidden = np.array([0.8, 0.1, 0.5, 0.9, 0.3, 0.7, 0.2, 0.6, 0.4, 1.0])
drop = Dropout(dropout_ratio=0.5, rng=np.random.default_rng(7))

masks = []
for _ in range(5):
    drop.forward(hidden, train_flg=True)
    masks.append(drop.mask.copy())

test_out = drop.forward(hidden, train_flg=False)


def main():
    for i, m in enumerate(masks, start=1):
        lit = "".join("●" if on else "○" for on in m)
        print(f"第 {i} 次：{lit}  亮着 {int(m.sum())} 个")
    print("mask 示例：", masks[0])
    print("测试时：全部点亮，输出 × (1 - 0.5) =", test_out)


if __name__ == "__main__":
    main()
