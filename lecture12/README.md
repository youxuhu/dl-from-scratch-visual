# 第12讲 与学习相关的技巧（下）：Batch Normalization、正则化与超参数验证

## 这讲讲什么
Batch Normalization 以 mini-batch 为单位把每个神经元的输入标准化（均值 0、方差 1），再用 gamma、beta 缩放平移，让学习更快、更不挑初始值。然后讲过拟合（参数多、数据少），以及两种正则化：权值衰减（损失加 0.5·λ·ΣW²，梯度加 λW）和 Dropout（训练时随机关掉神经元，测试时全开并乘保留比例）。最后讲超参数验证：从训练数据里切出验证数据，在对数尺度上随机搜索，再逐步缩小范围；测试数据始终不碰。

BatchNormalization、Dropout 在 common/layers.py；带 BN/Dropout 开关的网络是 common/multi_layer_net_extend.py 的 MultiLayerNetExtend（参数 use_batchnorm、use_dropout、dropout_ration，最后这个拼写沿用原书接口，Dropout 类自己的参数叫 dropout_ratio）；训练循环是 common/trainer.py 的 Trainer；打乱数据是 common/util.py 的 shuffle_dataset。

### 依赖
- 第8讲：common/gradient.py 的 numerical_gradient（测试里对 BN 层、带 BN 的网络、带权值衰减的网络做梯度确认）；common/mnist.py 的 load_mnist（经 get_mnist 调用）。
- 第10讲：common/layers.py 的 Affine、Relu、SoftmaxWithLoss。
- 第11讲：common/optimizer.py 的 SGD（Trainer 按名字创建优化器）；common/multi_layer_net.py 的 MultiLayerNet（weight_decay_lambda 参数）和 init_scale。
- 本讲新增到 common：common/layers.py 的 BatchNormalization、Dropout；common/multi_layer_net_extend.py 的 MultiLayerNetExtend；common/trainer.py 的 Trainer；common/util.py 的 shuffle_dataset。
- 说明：batchnorm_steps.py 先用 NumPy 一行一行「从零算一遍」，再调用 common 的 BatchNormalization 层对照，后面各讲用的是 common 版本。

对应鱼书：第6章 ch06/batch_norm_test.py、ch06/batch_norm_gradient_check.py、ch06/overfit_weight_decay.py、ch06/overfit_dropout.py、ch06/hyperparameter_optimization.py（第6章 6.3到6.5节；网络对应 common/multi_layer_net_extend.py，训练循环对应 common/trainer.py）

## 文件
| 文件 | 内容 |
|---|---|
| batchnorm_steps.py | 本仓库补充（书中 6.3.1 节的公式）。[3.0, -1.0, 5.0, 1.0] 的 BN 五步计算，和 BatchNormalization 层对照；滑动平均的第一步 |
| dropout_demo.py | 本仓库补充（书中 6.4.3 节的讲解）。10 个神经元连续 5 次 Dropout 的 mask，测试时乘 0.5 |
| batch_norm_gradient_check.py | 改编自书中同名文件。带 BN 的 MultiLayerNetExtend 做梯度确认；原书取 1 张图，批大小 1 时 BN 输出全是 0、比较不出东西，这里取前 3 张（支持 QUICK=1） |
| batch_norm_test.py | 改编自书中同名文件。16 种权重初始值标准差下有无 BN 的对比（支持 QUICK=1） |
| overfit_weight_decay.py | 改编自书中同名文件。300 个训练样本制造过拟合，λ = 0 和 0.1 对比（支持 QUICK=1） |
| overfit_dropout.py | 改编自书中同名文件。同样设置下有无 Dropout(0.2) 对比，用 Trainer 训练（支持 QUICK=1） |
| hyperparameter_optimization.py | 改编自书中同名文件。验证数据 + 学习率、权值衰减的对数随机搜索，两轮逐步缩小（支持 QUICK=1） |

## 怎么运行
```bash
cd lecture12
python batchnorm_steps.py
python dropout_demo.py
QUICK=1 python batch_norm_gradient_check.py  # 隐藏层 [20, 20]，几秒
python batch_norm_gradient_check.py          # 和原书相同的 [100, 100]，约 20 秒
QUICK=1 python batch_norm_test.py            # 4 个标准差、300 张、4 个 epoch
python batch_norm_test.py                    # 完整：16 个标准差、1000 张、20 个 epoch
QUICK=1 python overfit_weight_decay.py       # 60 个 epoch
QUICK=1 python overfit_dropout.py            # 150 个 epoch
QUICK=1 python hyperparameter_optimization.py  # 每轮 8 组、每组 30 个 epoch
```
去掉 QUICK=1 就是完整设置：overfit_weight_decay.py 201 个 epoch，overfit_dropout.py 301 个 epoch（和原书相同），hyperparameter_optimization.py 每轮 100 组、每组 50 个 epoch。

## 视频里出现的数值（都由脚本算出）
- BN 逐步计算：输入 [3.0, -1.0, 5.0, 1.0]，均值 2.0，xc = [1, -3, 3, -1]，方差 5.0，σ = √(5.0 + 10e-7) ≈ 2.2361，标准化后 [0.447, -1.342, 1.342, -0.447]。
- 再乘 gamma = 2.0、加 beta = 0.5：[1.394, -2.183, 3.183, -0.394]，和 BatchNormalization 层的输出一致。
- 滑动平均（momentum 0.9，从 0 开始）：一批之后 running_mean = 0.2，running_var = 0.5。
- Dropout（ratio 0.5，固定随机种子 7）：5 次正向分别亮着 6、6、5、2、8 个神经元；测试时 [0.8, 0.1, ...] 全部乘 0.5 变成 [0.4, 0.05, ...]。
- 实验设置：batch_norm_test 用 1000 个训练样本、20 个 epoch、np.logspace(0, -4, num=16) 共 16 个标准差；两个过拟合实验都只用 300 个训练样本；超参数搜索用 500 个样本（其中 20% 当验证数据），学习率 10^-6 到 10^-2、权值衰减 10^-8 到 10^-4。
- QUICK=1 下的运行结果（固定随机种子，重复运行结果相同；只用来确认流程）。完整设置下的结果：未实测。
  - 有无 BN：标准差 2.15e-3 时 BN 训练精度 0.780，无 BN 0.130（学不起来）。
  - 权值衰减（60 个 epoch）：λ = 0 时训练 0.927、测试 0.630，差距 0.297；λ = 0.1 时差距缩小到 0.175，ΣW² 从 1236.9 降到 863.6。
  - Dropout（150 个 epoch）：不用时训练精度 1.000、测试 0.698；用 Dropout(0.2) 后差距从 0.302 缩小到 0.103（学得也更慢）。

- BN 梯度确认（MNIST 前 3 张，seed 0）：[100, 100] 时各参数平均绝对误差 W1 2.1e-07、W2 1.7e-07、W3 4.0e-07、b3 7.6e-07、gamma1 4.1e-07、beta1 4.1e-07、gamma2 2.9e-07、beta2 2.9e-07；QUICK=1（[20, 20]）时都在 1e-7 量级或更小。b1、b2 的误差是 1e-13 以下，因为 BN 会减去均值，接在 BN 前面的偏置真实梯度就是 0。

## 测试
tests/test_lecture12.py：BN 五步的数值、batch_norm_gradient_check.py 在合成数据上的误差（< 1e-5）、BN 层和 MultiLayerNetExtend（use_batchnorm）的数值梯度 vs 反向传播梯度确认、推理模式用滑动平均、Dropout 的 mask 和测试缩放、权值衰减的梯度（带 λ 的 MultiLayerNet 梯度确认）、Trainer 和超参数搜索在合成数据上能跑通。
