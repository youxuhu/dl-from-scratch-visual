# 第14讲 卷积神经网络（下）：im2col、CNN 的实现与可视化

## 这讲讲什么
im2col 把卷积要看的每个窗口拉成一行，卷积就变成一次矩阵乘法（用内存换速度）；col2im 反过来，把梯度放回原图位置，重叠处累加。用它实现 Convolution 层和 Pooling 层，再组装成 SimpleConvNet（Conv → ReLU → Pool → Affine → ReLU → Affine → Softmax）训练 MNIST，最后看第 1 层滤波器学到了什么，并了解 LeNet 和 AlexNet。

im2col、col2im 在 common/util.py；Convolution、Pooling 在 common/layers.py；SimpleConvNet 在 common/simple_convnet.py。

### 依赖
- 第1讲：reshape、transpose。
- 第8讲：common/gradient.py 的 numerical_gradient（测试里对 SimpleConvNet 做梯度确认）；common/mnist.py 的 load_mnist（flatten=False，经 get_mnist 调用）。
- 第10讲：common/layers.py 的 Affine、Relu、SoftmaxWithLoss（SimpleConvNet 复用）。
- 第11讲：common/optimizer.py 的 Adam（Trainer 按名字创建）。
- 第12讲：common/trainer.py 的 Trainer。
- 第13讲：卷积、池化的原理和 4×4 示例数据；common/util.py 的 conv_output_size。
- 本讲新增到 common：common/util.py 的 im2col、col2im；common/layers.py 的 Convolution、Pooling；common/simple_convnet.py 的 SimpleConvNet。

对应鱼书：第7章 ch07/simple_convnet.py、ch07/gradient_check.py、ch07/train_convnet.py、ch07/visualize_filter.py、ch07/apply_filter.py（第7章 7.4到7.7节；im2col、col2im 对应 common/util.py，Convolution、Pooling 对应 common/layers.py）

## 文件
| 文件 | 内容 |
|---|---|
| im2col_demo.py | 本仓库补充。4×4 示例展开成 (4, 9) 再乘拉直的滤波器；像素重复次数；(9, 75) 和 (90, 75) 两个形状 |
| conv_pool_layers.py | 本仓库补充。Convolution 层内部 col、col_W 的形状，transpose(0, 3, 1, 2)；Pooling 的窗口行和每行最大值 |
| simple_convnet.py | 对应书中同名文件，类本身放在 common/simple_convnet.py。打印层的顺序（Conv1 → Relu1 → Pool1 → Affine1 → Relu2 → Affine2）、各层输出形状和参数形状 |
| gradient_check.py | 改编自书中同名文件。输入 1×10×10、10 个 3×3 滤波器、隐藏层 10 的小网络，数值梯度 vs 反向传播梯度 |
| train_convnet.py | 改编自书中同名文件。用 Trainer 训练 SimpleConvNet（支持 QUICK=1），参数存到 lecture14/output/ |
| visualize_filter.py | 改编自书中同名文件。训练前后第 1 层 30 个滤波器的平滑程度；竖边、横边滤波器作用在数字 1 上 |
| apply_filter.py | 改编自书中同名文件。把学到的前 16 个滤波器（偏置 0）分别作用在一张图上；原书用 lena_gray.png，这里换成 MNIST 测试集第一张 |
| lenet_alexnet.py | 本仓库补充。 LeNet 和 AlexNet 的逐层形状和卷积层参数个数 |

## 怎么运行
```bash
cd lecture14
python im2col_demo.py
python conv_pool_layers.py
python simple_convnet.py
python gradient_check.py
QUICK=1 python train_convnet.py   # 2000 张、3 个 epoch，十秒左右
python train_convnet.py           # 完整：60000 张、20 个 epoch（纯 NumPy，较慢）
python visualize_filter.py        # 优先读 train_convnet.py 存下的参数
python apply_filter.py            # 同上
python lenet_alexnet.py
```

## 视频里出现的数值（都由脚本算出）
- im2col 展开 4×4 示例得到 4 行 9 列，乘拉直的滤波器得 [10, 8, 2, 11]，reshape 成 [[10, 8], [2, 11]]，和第13讲逐格乘加一致。
- 每个像素在展开矩阵里出现的次数：[[1,2,2,1],[2,4,4,2],[2,4,4,2],[1,2,2,1]]，展开后元素个数是原图的 2.25 倍。
- 展开形状：(1, 3, 7, 7) 配 5×5 滤波器得 (9, 75)；(10, 3, 7, 7) 得 (90, 75)。
- Convolution 层（输入 (2, 3, 7, 7)，4 个 5×5 滤波器）：col (18, 75)，col_W (75, 4)，矩阵乘法后 reshape 成 (2, 3, 3, 4)，transpose(0, 3, 1, 2) 后为 (2, 4, 3, 3)。
- Pooling：第一个通道 [[7,1,6,0],[2,3,1,2],[1,0,2,1],[8,2,3,9]] 池化成 [[7, 6], [8, 9]]。
- SimpleConvNet 各层输出形状：1×28×28 → 30×24×24 → 30×12×12 → 100 → 10（filter_num 30、filter_size 5、pad 0、stride 1、hidden_size 100）。
- 梯度确认（gradient_check.py，seed 0）：各参数平均绝对误差 W1 5.2e-08、b1 5.2e-10、W2 5.9e-11、b2 3.3e-09、W3 1.3e-10、b3 1.8e-07。
- 测试精度：大纲里的 0.96 是用原书仓库自带参数测得的，本仓库不使用那份参数，而是用 train_convnet.py 自己训练。QUICK=1（2000 张、3 个 epoch，测试集前 300 张）为 0.8100；完整训练（20 个 epoch）的精度：未实测。
- 滤波器：训练前相邻格差别平均 0.493（随机噪点），读取 QUICK=1 训练的参数后为 0.249，变得更平滑、更像斑块和边缘。完整训练后的数值：未实测。
- 数字 1 上：竖边滤波器响应总和 688.0，横边滤波器 240.5，竖笔画处更亮。
- apply_filter.py：16 个 5×5 滤波器分别作用在 28×28 的图上，每个输出 24×24，合起来 (16, 24, 24)。响应强度取决于读到的是 QUICK 还是完整训练的参数，不列具体数值。
- LeNet：1×32×32 → 6×28×28 → 6×14×14 → 16×10×10 → 16×5×5，卷积层参数 2,572 个；AlexNet 最后到 256×6×6，卷积层参数 3,747,200 个。

## 测试
tests/test_lecture14.py：im2col 和 col2im 的数值与形状、col2im 是 im2col 的伴随（内积相等）、SimpleConvNet 的数值梯度 vs 反向传播梯度确认（含 gradient_check.py 的原书设置，平均绝对误差 < 1e-6）、保存与读取参数、在合成数据上训练几步损失下降、LeNet 参数个数。
