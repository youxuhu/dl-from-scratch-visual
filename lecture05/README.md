# 第5讲 神经网络（上）：激活函数与多维数组运算

## 这讲讲什么
从感知机到神经网络：把感知机的输出拆成「加权和 a」和「激活函数 h」两步；阶跃函数、sigmoid、ReLU 的形状、实现和区别；为什么激活函数必须是非线性的（线性叠多少层都等于一层）；多维数组的 ndim 和 shape；用一行 `Y = np.dot(X, W)` 算完一整层神经元。

3 层网络的完整前向传播、softmax 和 MNIST 推理按大纲放在第6讲。

## 文件
| 文件 | 内容 |
|---|---|
| activations.py | 只认实数的 step_scalar；数组版阶跃函数拆成 `x > 0` 和 `astype(int)` 两步；同一组 x 下三个函数的输出表 |
| plot_activations.py | 三条曲线画在一张图上，存到 output/ |
| why_nonlinear.py | h(x) = 2x 叠三层等于 8x；三个线性矩阵层合并成一个矩阵；sigmoid(2) 不等于 2·sigmoid(1) |
| layer_dot.py | 2 输入 3 输出的一层：ndim、shape、逐列计算、np.dot |

阶跃、sigmoid、ReLU 的函数本体在 common/functions.py（step_function、sigmoid、relu）。

## 怎么运行
```bash
cd lecture05
python activations.py
python plot_activations.py
python why_nonlinear.py
python layer_dot.py
```

## 视频里出现的数值
- x = [-1.5, 0.0, 2.5]：`x > 0` 得到 [False False True]，`astype(int)` 后 [0 0 1]。step_scalar 传数组会报 ValueError。
- probe = [-2.0, -0.5, 0.0, 1.0, 3.0]：step [0 0 0 1 1]；sigmoid [0.119 0.378 0.5 0.731 0.953]；relu [0 0 0 1 3]。x = 1.0 时 step 为 1、sigmoid 为 0.731、relu 为 1.0。
- 曲线采样 np.linspace(-6, 6, 121)；sigmoid 在两端是 0.0025 和 0.9975。
- 线性叠加：h(x) = 2x 三层作用在 1.5 上得 12.0，等于一层 8x；X = [1, -1] 逐层乘 W1、W2、W3 得 [4.0, 1.5]，和乘合并矩阵 W_all = [[7, 2.5], [3, 1]] 一样。sigmoid(2) = 0.881，2·sigmoid(1) = 1.462，不相等。
- 一层：X = [1, 2]，W = [[2, -1, 3], [1, 4, -2]]（负权重 -1、-2 用负值色）：Y[0] = 1×2 + 2×1 = 4，Y[1] = 1×(-1) + 2×4 = 7，Y[2] = 1×3 + 2×(-2) = -1，Y = [4 7 -1]，shape (3,)；X.ndim = 1，W.shape = (2, 3)。

## 依赖
- 第4讲：感知机（知识前置，本讲把它拆成加权和和激活函数两步）。
- common/functions.py：`step_function`、`sigmoid`、`relu`（本讲讲解的激活函数，代码只放在 common 里；activations.py 里的 `step_scalar` 是只认实数的教学版，后面不再使用）。
- 第1讲 / common/arrays.py：`matmul_steps`（逐列计算的算式）。

## 和大纲的不同
- 大纲的阶跃示例数组 [-1.0, 1.0, 2.0] 换成 [-1.5, 0.0, 2.5]，多了一个正好等于 0 的格子，顺便讲清「0 输出 0」。
- 大纲的一层示例 X = [1, 2] 乘出 `[5 11 17]` 和书中一样，这里换成自己的 W，结果 `[4 7 -1]`，带负权重和负输出，能用上负值色。
- 曲线采样点从 np.arange(-5, 5, 0.1) 换成 np.linspace(-6, 6, 121)。
- 「为什么要非线性」新增了矩阵版：三个线性层可以先乘成一个 W_all。
