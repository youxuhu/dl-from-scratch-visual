"""第9讲：苹果 2 个（单价 100）加橘子 3 个（单价 150），合计再乘消费税 1.1。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.layers import AddLayer, MulLayer  # noqa: E402

apple, apple_num = 100, 2
orange, orange_num = 150, 3
tax = 1.1

apple_node = MulLayer()
orange_node = MulLayer()
sum_node = AddLayer()
tax_node = MulLayer()

# 正向：从左往右
apple_total = apple_node.forward(apple, apple_num)        # 200
orange_total = orange_node.forward(orange, orange_num)    # 450
fruit_total = sum_node.forward(apple_total, orange_total)  # 650
price = tax_node.forward(fruit_total, tax)                # 715

# 反向：从右往左，起点是 1
dfruit_total, dtax = tax_node.backward(1)
dapple_total, dorange_total = sum_node.backward(dfruit_total)
dorange, dorange_num = orange_node.backward(dorange_total)
dapple, dapple_num = apple_node.backward(dapple_total)

forward_values = {"apple_total": apple_total, "orange_total": orange_total,
                  "fruit_total": fruit_total, "price": price}
backward_values = {"dApple": dapple, "dApple_num": dapple_num, "dOrange": dorange,
                   "dOrange_num": dorange_num, "dTax": dtax}


def main():
    for k, v in forward_values.items():
        print(f"正向 {k} = {round(v, 6)}")
    for k, v in backward_values.items():
        print(f"反向 {k} = {round(v, 6)}")


if __name__ == "__main__":
    main()
