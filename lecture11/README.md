# 第11讲 与学习相关的技巧（上）：参数更新与权重初始值

## 这讲讲什么
SGD 在 f(x, y) = x²/20 + y² 这种「一边陡一边缓」的函数上会走之字形。本讲依次介绍 Momentum（带速度的小球）、AdaGrad（按累积梯度平方给每个参数单独降学习率）、Adam（两者结合并做偏置修正），在等高线上让四个优化器赛跑，再到 MNIST 上比一比。后半讲讲权重初始值：为什么不能全一样，标准差 1 和 0.01 时激活值分布出了什么问题，Xavier 初始值适合 sigmoid，He 初始值适合 ReLU。

优化器在 common/optimizer.py（SGD、Momentum、Nesterov、AdaGrad、RMSprop、Adam，接口都是 update(params, grads)，Adam 和 Nesterov 的更新式按原书写法：Adam 把偏置修正合并进学习率 lr_t）；多层网络在 common/multi_layer_net.py（MultiLayerNet，weight_init_std 可取 'relu'/'he'、'sigmoid'/'xavier' 或一个数字）。

对应鱼书：第6章 ch06/optimizer_compare_naive.py、ch06/optimizer_compare_mnist.py、ch06/weight_init_activation_histogram.py、ch06/weight_init_compare.py（第6章 6.1、6.2节；优化器对应 common/optimizer.py，网络对应 common/multi_layer_net.py）

## 文件
| 文件 | 内容 |
|---|---|
| optimizer_compare_naive.py | 改编自书中同名文件。等高线赛跑：起点 (-7.0, 2.0)，30 步；打印终点和前 3 步的 v、h、m |
| optimizer_compare_mnist.py | 改编自书中同名文件。MNIST 上比较四种优化器（5 层网络，批 128，2000 次；支持 QUICK=1） |
| weight_init_activation_histogram.py | 改编自书中同名文件。5 个隐藏层的激活值分布：sigmoid 和 ReLU 各三种初始值 |
| weight_init_compare.py | 改编自书中同名文件。MNIST 上比较 std=0.01、Xavier、He 三种初始值（支持 QUICK=1） |
| zero_init.py | 本仓库补充（书中 6.2.1 节的讲解，没有对应源码文件）。权重全相同时，两个隐藏神经元训练后仍完全相同 |

画图是可选的：装了 matplotlib 时，图存到 lecture11/output/（common/util.py 的 get_pyplot），没装就只打印数值。

### 依赖
- 第5讲：common/functions.py 的 sigmoid、relu（激活值分布实验）。
- 第8讲：common/gradient.py 的 numerical_gradient（MultiLayerNet.numerical_gradient 内部调用，测试里做梯度确认）；common/mnist.py 的 load_mnist（经 get_mnist 调用）。
- 第10讲：common/layers.py 的 Affine、Relu、Sigmoid、SoftmaxWithLoss（MultiLayerNet 由它们组装）。
- 本讲新增到 common：common/optimizer.py 的 SGD、Momentum、Nesterov、AdaGrad、RMSprop、Adam；common/multi_layer_net.py 的 MultiLayerNet 和 init_scale（Xavier、He 初始值）。

## 怎么运行
```bash
cd lecture11
python optimizer_compare_naive.py
python weight_init_activation_histogram.py
python zero_init.py
QUICK=1 python optimizer_compare_mnist.py   # 2000 张图、120 次迭代，几秒钟
python optimizer_compare_mnist.py           # 完整：全部 60000 张，2000 次迭代
QUICK=1 python weight_init_compare.py
```

## 视频里出现的数值（都由脚本算出）
- 赛跑设置：起点 (-7.0, 2.0)，学习率 SGD 0.95、Momentum 0.1、AdaGrad 1.5、Adam 0.3，各 30 步。
- 30 步后的位置：SGD (-0.3504, 0.0848)，Momentum (0.8964, 0.0881)，AdaGrad (-0.1202, 0.0000)，Adam (0.1360, 0.3182)。
- Momentum 前 3 步的速度 v：(0.0700, -0.4000)、(0.1323, -0.6800)、(0.1870, -0.7960)。第一步就是 -lr·梯度 = -0.1 × (-0.7, 4.0)。
- AdaGrad 前 3 步的水位 h：(0.49, 16.0)、(0.7925, 17.0)、(1.0016, 17.0742)。y 方向第一步梯度 4，平方 16，水位一下子就很高，所以 y 方向的步子马上变小。
- Adam 前 3 步：m = (-0.07, 0.4)、(-0.13, 0.7)、(-0.181, 0.9104)；v = (0.0005, 0.016)、(0.0009, 0.0275)、(0.0013, 0.0354)。
- 激活值分布（1000×100 输入，5 层，每层 100 个神经元，固定随机种子）：
  - sigmoid + std=1：各层约一半的值贴在两端（<0.01 和 >0.99 各占约 20% 到 32%），第 1 层直方图两端的柱子各约 41000，梯度消失。
  - sigmoid + std=0.01：全挤在 0.5 附近，标准差只有约 0.01 到 0.025，表现力受限。
  - sigmoid + Xavier：标准差约 0.11 到 0.21，分布较宽。
  - ReLU + std=0.01：第 3 层起 100% 的值小于 0.01；ReLU + Xavier：平均值逐层从 0.397 降到 0.116；ReLU + He：各层平均值都在 0.56 到 0.66，分布一致。
- 权重全设 0.5，训练 20 步后 W1 的两列仍完全相同（0.451714、0.528061、0.522863）。
- MNIST 上的优化器比较，QUICK=1（2000 张训练图、120 次迭代、测试集前 1000 张）：平滑后最终损失 SGD 1.2590、Momentum 0.2382、AdaGrad 0.0475、Adam 0.1223，测试精度 SGD 0.6420、Momentum 0.8460、AdaGrad 0.8850、Adam 0.8670，SGD 明显落后。完整设置（60000 张、2000 次迭代）：未实测。
- 初始值比较，QUICK=1（150 次迭代）：std=0.01 损失 2.3025 → 2.3017 几乎不降（测试精度 0.0990），Xavier 降到 2.1824（0.3260），He 降到 0.9796（0.6930）。完整设置：未实测。

## 测试
tests/test_lecture11.py：检查优化器单步更新的公式、赛跑终点和前 3 步状态、激活值分布的特征、双胞胎神经元，以及 MultiLayerNet（ReLU+He、sigmoid+Xavier）的数值梯度 vs 反向传播梯度确认。
