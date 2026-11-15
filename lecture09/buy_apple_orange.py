"""第9讲：苹果 2 个（单价 100）加橘子 3 个（单价 150），合计再乘消费税 1.1。

改编自《深度学习入门：基于Python的理论与实现》官方源码 ch05/buy_apple_orange.py
（MIT 许可，Copyright (c) 2016 Koki Saitoh）。变量名和计算顺序与原书一致，注释改写成中文，
另把正向、反向的中间值收进两个字典，方便打印和测试。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.layers import AddLayer, MulLayer  # noqa: E402

apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

# 层
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

# 正向：从左往右
apple_price = mul_apple_layer.forward(apple, apple_num)                # (1) 200
orange_price = mul_orange_layer.forward(orange, orange_num)            # (2) 450
all_price = add_apple_orange_layer.forward(apple_price, orange_price)  # (3) 650
price = mul_tax_layer.forward(all_price, tax)                          # (4) 715

# 反向：从右往左，起点 dprice = 1
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)                          # (4)
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)  # (3)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)            # (2)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)                # (1)

forward_values = {"apple_price": apple_price, "orange_price": orange_price,
                  "all_price": all_price, "price": price}
backward_values = {"dApple": dapple, "dApple_num": dapple_num, "dOrange": dorange,
                   "dOrange_num": dorange_num, "dTax": dtax}


def main():
    for k, v in forward_values.items():
        print(f"正向 {k} = {round(v, 6)}")
    for k, v in backward_values.items():
        print(f"反向 {k} = {round(v, 6)}")


if __name__ == "__main__":
    main()
