# 第15讲 深度学习：加深网络、历史、高速化、应用与未来

## 这讲讲什么
为什么要加深网络：两层 3×3 卷积和一层 5×5 卷积看到同样大的区域，参数却更少，还多了一次非线性。然后看一个加深的网络 DeepConvNet（6 个 3×3 卷积层、He 初始值、Adam、Dropout），它在 MNIST 上认错的样本连人都容易看错；数据扩充（旋转、平移、裁剪）让训练数据「变多」。之后是深度学习小历史（ImageNet、VGG、GoogLeNet、ResNet）、高速化（GPU、分布式、16 位半精度），以及物体检测、图像分割、看图说话、风格变换、图像生成、自动驾驶、强化学习等应用，最后回顾整套课。本讲以概念为主，代码只覆盖能用 NumPy 演示的部分。

DeepConvNet 在 common/deep_convnet.py。

### 依赖
- 第8讲：common/gradient.py 的 numerical_gradient（DeepConvNet.numerical_gradient，测试里做梯度确认）；common/mnist.py 的 load_mnist（经 get_mnist 调用）。
- 第10讲：common/layers.py 的 Affine、Relu、SoftmaxWithLoss。
- 第11讲：common/optimizer.py 的 Adam；He 初始值 sqrt(2 / 前一层节点数)。
- 第12讲：common/layers.py 的 Dropout；common/trainer.py 的 Trainer。
- 第13讲、第14讲：common/layers.py 的 Convolution、Pooling；common/util.py 的 im2col、conv_output_size。
- 本讲新增到 common：common/deep_convnet.py 的 DeepConvNet。
- lecture15/_trained.py 是本讲脚本共用的小模块（读取或当场训练 DeepConvNet），本身也只调用 common。

## 文件
| 文件 | 内容 |
|---|---|
| receptive_field.py | 5×5 卷积 25 个参数 vs 两层 3×3 卷积 18 个参数；用 Convolution 层反向传播求两层 3×3 的感受野 |
| deep_convnet_structure.py | DeepConvNet 的层顺序、逐层形状、参数总数 |
| train_deepnet.py | 用 Trainer（Adam，学习率 0.001）训练 DeepConvNet（支持 QUICK=1），参数存到 lecture15/output/ |
| misclassified_mnist.py | 列出认错的测试样本（预测、真实标签），有 matplotlib 时存图 |
| half_float_network.py | 参数、输入、每层输出舍入成 float16 后的精度，和 float64 对比 |
| data_augmentation.py | 对一张手写数字做旋转 ±15°、上下左右平移 3 格、裁剪放大 |
| _trained.py | 共用：有保存的参数就读取，没有就当场简单训练 |

## 怎么运行
```bash
cd lecture15
python receptive_field.py
python deep_convnet_structure.py
python data_augmentation.py
QUICK=1 python train_deepnet.py      # 1200 张、2 个 epoch，二十秒左右
python train_deepnet.py              # 完整：60000 张、20 个 epoch（纯 NumPy，要几个小时）
QUICK=1 python misclassified_mnist.py
QUICK=1 python half_float_network.py
```
misclassified_mnist.py 和 half_float_network.py 会优先读取 train_deepnet.py 存下的参数，所以先跑 train_deepnet.py；没有参数时它们会当场训练一小会儿，精度会很低。

## 视频里出现的数值
都由脚本算出：
- 参数个数：5×5 卷积 25 个，两层 3×3 卷积 18 个。
- 两层 3×3 卷积的最终输出对 5×5 输入每个像素的依赖次数为 [[1,2,3,2,1],[2,4,6,4,2],[3,6,9,6,3],[2,4,6,4,2],[1,2,3,2,1]]，覆盖全部 25 个像素。
- DeepConvNet：卷积通道数 16、16、32、32、64、64，填充 1、1、1、2、1、1；形状 1×28×28 → 16×28×28 → 16×28×28 → 16×14×14 → 32×14×14 → 32×16×16 → 32×8×8 → 64×8×8 → 64×8×8 → 64×4×4 → 50 → 10；参数总数 123,552。
- 数据扩充（MNIST 测试集第 0 张，标签 7）：7 种变换，形状都是 28×28；平移后与原图逐像素相同的比例约 86% 到 87%，旋转 ±15° 约 90%。

QUICK=1 下一次运行的结果（只确认流程，不代表模型实力）：
- train_deepnet.py：测试精度 0.5150（200 张测试图）。
- half_float_network.py（读取上面 QUICK 训练的参数，300 张测试图）：64 位 0.4967，16 位 0.4967，没有变化。

未实测：
- 大纲里的 DeepConvNet 精度 0.9797 和 float16 精度 0.9798 是用原书仓库自带参数测得的，本仓库不使用那份参数，也还没有跑过 train_deepnet.py 的完整训练，所以完整训练后的精度、错例和 float16 对比都还没有实测值。视频制作时需要先完整训练一次，再用实际输出替换。

## 测试
tests/test_lecture15.py：参数个数和感受野、DeepConvNet 的结构和参数总数、小号 DeepConvNet 的数值梯度 vs 反向传播梯度确认（最大绝对误差 / 最大梯度绝对值 < 1e-4）、半精度函数不改动原参数、数据扩充的平移和旋转、错例查找。
