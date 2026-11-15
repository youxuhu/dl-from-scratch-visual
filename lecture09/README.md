# 第9讲 误差反向传播法（上）：计算图与链式法则

## 这讲讲什么
用计算图表示「买苹果、买橘子、算消费税」这样的计算，正向从左往右算出总价，反向从右往左把梯度一站一站传回去。讲清局部计算、链式法则，以及两条最基本的规则：加法节点把梯度原样分给两边，乘法节点把梯度乘上「另一个输入」（翻转）。最后把这两条规则写成 MulLayer、AddLayer 两个类，接口统一是 forward 和 backward。

## 文件
| 文件 | 内容 |
|---|---|
| layer_naive.py | MulLayer（乘法层）、AddLayer（加法层），各有 forward、backward |
| buy_apple.py | 苹果 100 元 × 2 个 × 消费税 1.1：正向、反向，以及单价加 0.01 的对照小实验 |
| buy_apple_orange.py | 再加橘子 150 元 × 3 个，两条分支在加法节点汇合 |
| chain_rule.py | z = t²，t = x + y 的链式法则，解析结果和数值微分对照 |

## 怎么运行
```bash
cd lecture09
python buy_apple.py
python buy_apple_orange.py
python chain_rule.py
cd ..
python -m pytest -q tests/test_lecture09.py
```

## 视频里出现的数值（都由脚本算出）
- 苹果：100 × 2 = 200，200 × 1.1 = 220。反向：dTax = 200，dApple_num = 110，dApple = 2.2。
- 对照小实验：单价改成 100.01，总价 220 → 220.022，每元变化 2.2，和梯度一致。
- 苹果加橘子：苹果 200，橘子 450，合计 650，含税 715。反向：dApple = 2.2，dApple_num = 110，dOrange = 3.3，dOrange_num = 165，dTax = 650。
- 链式法则：x = 3，y = 1 时 t = 4，z = 16；∂z/∂t = 8，∂t/∂x = 1，∂z/∂x = 8，数值微分同样是 8.000000。

小提示：1.1 不能被二进制浮点数精确表示，直接打印会看到 220.00000000000003 这样的尾巴，脚本里统一四舍五入到 6 位小数再显示。

## 测试
tests/test_lecture09.py 检查上面全部数值，并做梯度确认：把整张购物计算图写成一个函数，用 common/gradient.py 的数值梯度对每个输入求偏导，和 MulLayer、AddLayer 反向传播得到的结果逐个比较。
