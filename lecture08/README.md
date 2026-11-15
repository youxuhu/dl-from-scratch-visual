# 第8讲 神经网络的学习（下）：数值微分、梯度与学习算法

## 这讲讲什么
导数和数值微分（前向差分的误差、中心差分、h 太小会被舍入成 0）；偏导数和梯度；梯度指向函数值增加最快的方向，沿负梯度走就是梯度下降；学习率太大发散、太小不动；神经网络的梯度（损失对权重矩阵每一格的偏导）；2 层神经网络 NumericalTwoLayerNet 的类，mini-batch 训练，每个 epoch 用训练集和测试集评价。最后实测数值梯度有多慢，引出第9、10讲的反向传播。

## 文件
| 文件 | 内容 |
|---|---|
| __init__.py | 让 lecture08 能作为包导入（第10讲用它做对比） |
| numerical_two_layer_net.py | `NumericalTwoLayerNet`：params、predict、loss、accuracy、numerical_gradient |
| derivative.py | f(x) = 0.05x² - 0.3x 在 x = 4 处：前向差分 vs 中心差分，h = 1e-50 的舍入问题 |
| gradient_demo.py | f(x0, x1) = x0² + 2·x1² 的偏导、负梯度箭头、三种学习率的梯度下降，存图到 output/ |
| weight_gradient.py | 2×3 权重的小网络 SimpleNet：数值梯度 dW、走一步后损失下降 |
| train_numerical.py | 用数值梯度真的训练 NumericalTwoLayerNet（MNIST 缩成 7×7），并估算原尺寸网络算一次数值梯度的耗时 |
| results/train_log.json | 完整训练一次的真实记录（每步 loss、每个 epoch 的训练/测试精度），视频里的曲线用它驱动 |

### NumericalTwoLayerNet 接口（第10讲依赖，不要随便改）
```python
from lecture08.numerical_two_layer_net import NumericalTwoLayerNet
net = NumericalTwoLayerNet(input_size, hidden_size, output_size, weight_init_std=0.01, seed=None)
net.params                    # {"W1", "b1", "W2", "b2"}；W 用 weight_init_std × 标准正态，b 是 0
net.predict(x)                # softmax 之前的分数 (N, output_size)，隐藏层 sigmoid
net.loss(x, t)                # 平均交叉熵；t 是标签下标或 one hot 都行
net.accuracy(x, t)
net.numerical_gradient(x, t)  # {"W1", "b1", "W2", "b2"}，中心差分 h = 1e-4
```
从仓库根目录 `from lecture08.numerical_two_layer_net import NumericalTwoLayerNet` 没有副作用（不读数据、不打印）。

## 怎么运行
```bash
cd lecture08
python derivative.py
python gradient_demo.py
python weight_gradient.py
QUICK=1 python train_numerical.py   # 只训练 20 步，约 5 秒
python train_numerical.py           # 完整训练 1500 步，约 5 分钟，另外估算原尺寸网络的梯度耗时
```

## 视频里出现的数值
- 数值微分：f(x) = 0.05x² - 0.3x，f(4) = -0.4，真实导数 f'(4) = 0.1。
  - 前向差分：h = 1 → 0.15，h = 0.1 → 0.105，h = 0.01 → 0.1005，h = 1e-4 → 0.100005。
  - 中心差分：四个 h 都是 0.100000（二次函数的中心差分没有截断误差）。
  - h = 1e-50：4 + 1e-50 在浮点数里就是 4，前向差分算出 0.0；np.float32(1e-50) 也是 0.0。
  - 切线 y = -0.4 + 0.1·(x - 4)，在 x = 0 处是 -0.8。
- 梯度：f(x0, x1) = x0² + 2·x1² 在 (2.0, -1.0) 处梯度 (4, -4)；网格点 (2, 2) 处的负梯度 (-4, -8)，箭头在 x1 方向更长（碗在 x1 方向更陡）。
- 梯度下降，起点 (-2.0, 3.0)，50 步：
  - lr = 0.1：第 1 步到 (-1.6, 1.8)，50 步后约 (-2.9e-5, 2.4e-11)，到达谷底。
  - lr = 0.6：第 1 步到 (0.4, -4.2)，x0 方向收敛，x1 方向每步乘 -1.4，50 步后 x1 约 6.1e7，发散。
  - lr = 1e-4：第 1 步到 (-1.9996, 2.9988)，50 步后约 (-1.9801, 2.9406)，几乎没动。
- SimpleNet：W = [[0.3, -0.2, 0.6], [-0.5, 0.4, 0.1]]，x = [0.8, -0.5]，正确类别 1。分数 [0.49, -0.36, 0.43]，概率 [0.4221, 0.1804, 0.3975]，损失 1.7125；dW = [[0.3377, -0.6557, 0.318], [-0.211, 0.4098, -0.1988]]（正确类别那一列是负的）；W -= 0.5·dW 后损失降到 1.2888。
- 数值梯度训练（实测，results/train_log.json）：MNIST 缩成 7×7，49 → 20 → 10 共 1210 个参数，训练集前 10000 张，batch 100，lr 2.0，weight_init_std 0.1，种子 8，1500 步 = 15 个 epoch，用时 289 秒（每步约 0.19 秒）。
  - loss 从 2.26 左右下降，全程最低 0.1399，后半段大致在 0.14 到 0.52 之间波动。
  - 训练 / 测试精度：epoch 1 是 0.6673 / 0.6516，epoch 5 是 0.8691 / 0.8617，epoch 10 是 0.8857 / 0.8815，epoch 15 是 0.8994 / 0.8983。两条线几乎重合，没有过拟合。
- 计算量的伏笔：原尺寸 784 → 50 → 10 有 39760 个参数，实测前 2000 个参数的耗时再按比例估算，算一次数值梯度在我们的机器上大约需要一个小时，跟硬件有关。第10讲的反向传播一次只要毫秒级。

## 依赖
- 第5讲 / common/functions.py：`sigmoid`；第6讲 / common/functions.py：`softmax`；第7讲 / common/functions.py：`cross_entropy_error`。
- 本讲 / common/gradient.py：`numerical_diff`、`numerical_gradient`、`gradient_descent`（本讲讲解的函数，代码只放在 common 里；derivative.py 里的 `forward_diff` 是演示前向差分误差的教学版）。
- common/mnist.py：`load_mnist`。
- 被依赖：第10讲 import `lecture08.numerical_two_layer_net.NumericalTwoLayerNet`，和反向传播版做梯度确认和速度对比。

## 和大纲的不同
- 数值微分的函数从书中的 0.01x² + 0.1x（x = 5，导数 0.2）换成自己的 0.05x² - 0.3x（x = 4，导数 0.1）。
- 梯度的函数从 x0² + x1² 换成 x0² + 2·x1²（椭圆碗，梯度方向和指向原点的方向不再重合，更能看出「梯度不一定直指最低点」）；起点从 (-3.0, 4.0) 换成 (-2.0, 3.0)；学习率三档从 0.1 / 10 / 1e-10 换成 0.1 / 0.6 / 1e-4，50 步。大纲里那组实测数值作废，以上面为准。
- 大纲说训练 TwoLayerNet（784 → 50 → 10）并画出精度曲线。用数值梯度训练原尺寸网络一步就要约一小时，所以这里改成 7×7 的小图、49 → 20 → 10 的网络，真的用数值梯度训练，精度曲线是真实的（最终测试精度 0.8983）。原尺寸网络的完整训练放到第10讲用反向传播做。
- 类名用 NumericalTwoLayerNet（和第10讲约定），书中是 TwoLayerNet。
