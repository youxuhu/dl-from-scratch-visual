"""第15讲：加深层可以减少参数。一次 5×5 卷积和两次 3×3 卷积「看到」同样大的 5×5 区域。

参数（单通道、不算偏置）：5×5 是 25 个，两层 3×3 是 9 + 9 = 18 个。
用 common 的 Convolution 层反向传播，算出第二层中心那一个输出受哪些输入像素影响。

本仓库补充，对应书中 8.1.2 节，书中没有对应源码文件。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from common.layers import Convolution  # noqa: E402

params_5x5 = 5 * 5
params_two_3x3 = 3 * 3 + 3 * 3

x = np.zeros((1, 1, 5, 5))
first = Convolution(np.ones((1, 1, 3, 3)), np.zeros(1))
second = Convolution(np.ones((1, 1, 3, 3)), np.zeros(1))
middle = first.forward(x)             # 5×5 → 3×3
top = second.forward(middle)          # 3×3 → 1×1
grad_mid = second.backward(np.ones_like(top))
grad_in = first.backward(grad_mid)    # 每个输入像素对最终输出的影响次数
covered = int(np.count_nonzero(grad_in))


def main():
    print(f"5×5 卷积参数 {params_5x5} 个，两层 3×3 卷积参数 {params_two_3x3} 个")
    print("形状：输入", x.shape[2:], "→ 第一层", middle.shape[2:], "→ 第二层", top.shape[2:])
    print("最终输出对每个输入像素的依赖次数：\n", grad_in[0, 0].astype(int))
    print(f"覆盖了 {covered} 个像素，也就是整个 5×5 区域")


if __name__ == "__main__":
    main()
