# 第6讲 神经网络（下）：3层网络、输出层与手写数字识别

对应鱼书：第3章 ch03/neuralnet_mnist.py、neuralnet_mnist_batch.py；softmax 与 3 层前向传播见第3章正文，权重加载对齐 ch03 的 init_network / predict

## 这讲讲什么
用 NumPy 写出一个 3 层神经网络的前向传播，跟踪每一层的形状；输出层用恒等函数（回归）还是 softmax（分类）；softmax 的溢出问题和「减去最大值」的修正；用训练好的权重识别 MNIST 手写数字，再把一张一张的推理改成 100 张一批。

## 文件
| 文件 | 内容 |
|---|---|
| three_layer.py | init_small_network、forward：2 → 3 → 2 → 2 的小网络，权重自己定 |
| softmax_demo.py | softmax 的三步（exp、求和、相除）；直接算会溢出、减去最大值后正常 |
| mnist_inference.py | get_data、逐张和按批算精度；init_network（读 weights/ 里的权重）和 predict 在 common/simple_net.py，因为第7讲也要用 |
| neuralnet_mnist.py | 对齐鱼书 ch03/neuralnet_mnist.py：一张一张算精度，打印 Accuracy: |
| neuralnet_mnist_batch.py | 对齐鱼书 ch03/neuralnet_mnist_batch.py：100 张一批算精度 |
| train_weights.py | 生成 weights/ 里那份权重的脚本（手写的反向传播，原理在第9、10讲）。默认不覆盖已提交的权重，可用 --out、--log 指定路径 |
| weights/mnist_784_50_100_10.npz | 训练好的权重，784 → 50 → 100 → 10，float32，约 167 KB |
| results/train_log.json | 用同一脚本完整复现训练的日志（每 100 步的 loss、最后 100 步平均 loss、用时、测试精度、种子） |

sigmoid、softmax、identity、load_mnist 都从 common/ 导入，本讲不重新定义。softmax_demo.py 里的 naive_softmax 只是为了演示溢出而写的「教学版」。

## 怎么运行
```bash
cd lecture06
python three_layer.py
python softmax_demo.py
python mnist_inference.py        # 第一次运行会下载 MNIST 到仓库根目录的 dataset/
python neuralnet_mnist.py        # 对齐鱼书的逐张推理
python neuralnet_mnist_batch.py  # 对齐鱼书的批量推理
python train_weights.py          # 完整训练，权重默认写到 output/，不会覆盖已提交的权重
QUICK=1 python train_weights.py  # 只训练 50 步，结果存到 output/
```

## 权重是怎么来的
- 网络结构 784 → 50 → 100 → 10，隐藏层 sigmoid，输出 softmax。
- 初始化：标准正态 / sqrt(输入个数)，随机种子 6；mini-batch 100，学习率 0.5，12000 步（20 个 epoch），手写反向传播。
- 仓库里已提交的权重（weights/mnist_784_50_100_10.npz）是视频和测试使用的那一份，测试集精度 0.9743。
- results/train_log.json 是后来用同一脚本、同一组超参数做的一次复现：种子 6，12000 步，用时 69.955 秒，最后 100 步平均训练 loss 0.042576（四位小数 0.0426），测试集精度同样是 0.9743。
- 没有使用鱼书仓库的 sample_weight.pkl。

## 视频里出现的数值
- 3 层网络，X = [1.0, -0.5]：
  - A1 = X·W1 + b1 = [0.1, -0.15, 0.5]，Z1 = sigmoid(A1) = [0.525, 0.463, 0.622]
  - A2 = [0.376, 0.032]，Z2 = [0.593, 0.508]
  - Y = A3 = [0.262, -0.042]（恒等函数）
  - 形状链 (2,) → (2, 3) → (3,) → (3, 2) → (2,) → (2, 2) → (2,)
- softmax，a = [1.2, 3.1, 0.4]：exp(a) = [3.320, 22.198, 1.492]，总和 27.01，概率 [0.123, 0.822, 0.055]，和为 1，最大的位置不变（下标 1）。
- 溢出：直接算 [1500, 1498, 1495] 得到 [nan nan nan]；减去最大值变成 [0, -2, -5]，softmax = [0.876, 0.118, 0.006]。
- MNIST：测试集第 0 张标签 7，预测 7，概率 0.999；第一批 100 张输入 (100, 784)、输出 (100, 10)，这一批 100 张全对；**测试集 10000 张精度 0.9743（实测）**，逐张算和按批算结果相同（QUICK=1 时逐张只算前 1000 张，精度 0.983）。

## 依赖
- 第5讲 / common/functions.py：`sigmoid`；本讲 / common/functions.py：`softmax`、`identity`（本讲讲解，代码放在 common；softmax_demo.py 里的 `naive_softmax` 是演示溢出用的教学版）。
- 第7讲要用的推理网络 `init_network`、`predict` 放在 common/simple_net.py。
- train_weights.py 用 common/functions.py 的 `cross_entropy_error`（第7讲讲解）生成权重。
- common/mnist.py：`load_mnist`。

## 和大纲的不同
- 大纲的 3 层示例 X = [1.0, 0.5]、A1 = [0.3 0.7 1.1]、Z1 = [0.574 0.668 0.750] 和书中一样，这里换成自己的权重和输入，数值见上。
- softmax 示例从 [0.3, 2.9, 4.0] 换成 [1.2, 3.1, 0.4]；溢出示例从 [1010, 1000, 990] 换成 [1500, 1498, 1495]。
- 大纲的 0.9352 来自鱼书仓库的 sample_weight.pkl。我们不发布那份文件，改用自己训练的权重，精度 0.9743。台词里的精度请用 0.9743。
