# 第10讲 误差反向传播法（下）：激活函数层、Affine/Softmax 层与完整实现

## 这讲讲什么
把第9讲的「层」思路推广到神经网络：ReLU 层用 mask 挡梯度，Sigmoid 层的反向化简成 dout · y(1 - y)，Affine 层的反向靠转置对齐形状，Softmax-with-Loss 层的反向化简成干净的 (y - t) / 批大小。然后把这些层按顺序放进 OrderedDict，组装成用反向传播求梯度的 TwoLayerNet，先用数值梯度做梯度确认，再拿它训练 MNIST。

各层的实现在 common/layers.py（Relu、Sigmoid、Affine、SoftmaxWithLoss），TwoLayerNet 在 common/two_layer_net.py，第11讲以后直接导入。

## 文件
| 文件 | 内容 |
|---|---|
| activation_layers.py | ReLU 的 mask 与正反向；Sigmoid 拆成四个节点逐步回传，并和化简公式对照 |
| affine_layer.py | Affine 层的形状、Wᵀ、单样本与批版本的反向，偏置梯度沿批方向求和 |
| softmax_loss_grad.py | 预测很差和预测很准的两个样本，各自的 y - t |
| two_layer_net_bp.py | 打印 common/two_layer_net.py 里 TwoLayerNet 的层顺序和参数形状（OrderedDict 保存层，predict、loss、accuracy、numerical_gradient、gradient），并找到第8讲的数值梯度版做对照 |
| gradient_check.py | 梯度确认：第8讲 NumericalTwoLayerNet 的数值梯度 vs 本讲 TwoLayerNet 的反向传播梯度（同一 seed、拷贝参数），MNIST 前 3 张，平均绝对误差和耗时对比 |
| train_neuralnet.py | 用反向传播训练 TwoLayerNet（支持 QUICK=1） |

### 依赖
- 第5讲、第7讲：common/functions.py 的 sigmoid、softmax、cross_entropy_error（Sigmoid、SoftmaxWithLoss 层内部调用）。
- 第8讲：common/gradient.py 的 numerical_gradient（梯度确认）；lecture08/numerical_two_layer_net.py 的 NumericalTwoLayerNet（gradient_check.py 和测试里拿来和反向传播版对照）；common/mnist.py 的 load_mnist（经 common/util.py 的 get_mnist 调用）。
- 第9讲：计算图与层的 forward/backward 接口（common/layers.py）。
- 本讲新增到 common：common/layers.py 的 Relu、Sigmoid、Affine、SoftmaxWithLoss，common/two_layer_net.py 的 TwoLayerNet。
- 说明：activation_layers.py 里的 sigmoid_by_nodes 是把 sigmoid 拆成四个节点「从零写一遍」的教学演示，真正给后面用的是 common 里的 Sigmoid 层。

## 怎么运行
```bash
cd lecture10
python activation_layers.py
python affine_layer.py
python softmax_loss_grad.py
python gradient_check.py
QUICK=1 python train_neuralnet.py   # 只跑 300 步，几秒钟
python train_neuralnet.py           # 完整训练：10000 步，批大小 100，学习率 0.1
```
第一次读 MNIST 会自动下载到仓库根目录的 dataset/（不进 git）。下载失败时脚本会提示并改用小规模合成数据，数值会和视频不同。

## 视频里出现的数值（都由脚本算出）
- ReLU：输入 [[1.0, -0.5], [-2.0, 3.0]]，mask = [[False, True], [True, False]]，输出 [[1, 0], [0, 3]]；上游梯度 [[0.4, 0.8], [-1.2, 0.6]] 反向后变成 [[0.4, 0], [0, 0.6]]。
- Sigmoid（x = 1.0，dout = 1）：正向 -x = -1，exp(-x) = 0.367879，1 + exp(-x) = 1.367879，y = 0.731059；反向依次 -0.534447、-0.534447、-0.196612、0.196612；化简公式 y(1 - y) = 0.196612，与 Sigmoid 层一致。
- Affine：X = [1.0, 0.5]，W = [[0.2, -0.4, 0.1], [0.6, 0.2, -0.3]]，b = [0.1, 0.1, 0.2]，Y = [0.6, -0.2, 0.15]；dY = [0.3, -0.2, 0.1] 时 dX = [0.15, 0.11]，dW = [[0.3, -0.2, 0.1], [0.15, -0.1, 0.05]]。
- 批版本：X2 = [[1.0, 0.5], [0.0, 2.0]]，dY2 = [[0.3, -0.2, 0.1], [-0.1, 0.4, 0.2]]，db = [0.2, 0.2, 0.3]，dX2 = [[0.15, 0.11], [-0.16, -0.04]]。
- Softmax-with-Loss（正确类别 3）：预测很差的样本损失 2.9417，正确类那格梯度 -0.947，其余格是正的；预测很准的样本损失 0.0166，正确类那格梯度 -0.016，其余几乎为 0。
- 梯度确认（MNIST 前 3 张，seed 0，和第8讲的网络对照，隐藏层用 sigmoid 与第8讲一致）：W1 2.2e-10，b1 1.0e-9，W2 7.3e-8，b2 1.5e-7，量级在 1e-10 到 1e-7。两次运行中数值梯度用了约 1.8 秒和 4 秒，反向传播都不到 1 毫秒，快几千到一万倍（耗时随机器负载变化，误差数值每次都一样，因为 seed 固定）。
- QUICK=1 训练（2000 张训练图、300 步，测试集取前 1000 张）：损失从 2.3001 降到 0.3459，最终测试精度 0.8290。
- 完整训练（10000 步）的精度：未实测。

## 测试
tests/test_lecture10.py：检查上面的固定数值；对 Relu、Sigmoid、Affine、SoftmaxWithLoss 每一层单独做梯度确认；再对一个小号 TwoLayerNet 做数值梯度 vs 反向传播的整体梯度确认（用随机数据，不需要网络）；第8讲的 NumericalTwoLayerNet 存在时，用同一 seed、拷贝参数后对照两者的正向结果和梯度。
