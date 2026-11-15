"""第9讲：买 2 个苹果（单价 100），再乘消费税 1.1。先正向算总价，再反向求每个输入的梯度。

改编自《深度学习入门：基于Python的理论与实现》官方源码 ch05/buy_apple.py
（MIT 许可，Copyright (c) 2016 Koki Saitoh）。变量名和计算顺序与原书一致，注释改写成中文，
另加了「单价加 0.01」的对照小实验。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.layers import MulLayer  # noqa: E402

apple = 100
apple_num = 2
tax = 1.1

mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()


# 正向
apple_price = mul_apple_layer.forward(apple, apple_num)  # 200
price = mul_tax_layer.forward(apple_price, tax)          # 220

# 反向
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)


def nudge_check(delta=0.01):
    """把苹果单价加 delta，看总价变了多少，和梯度 2.2 对照（第8讲数值微分的思路）。"""
    total = MulLayer().forward(MulLayer().forward(apple + delta, apple_num), tax)
    return total, (total - price) / delta


def main():
    print(f"正向：100 × 2 = {apple_price}，再 × 1.1 = {round(price, 6)}")
    print(f"反向：dTax = {round(dtax, 6)}")
    print(f"      dApple_num = {round(dapple_num, 6)}")
    print(f"      dApple = {round(dapple, 6)}")
    total, slope = nudge_check()
    print(f"单价加 0.01：总价 {round(total, 6)}，每元变化 {round(slope, 4)}（梯度是 {round(dapple, 6)}）")


if __name__ == "__main__":
    main()
