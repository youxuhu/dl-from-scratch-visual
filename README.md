# 从零开始的深度学习（可视化版）

配套「白猫」讲解的中文深度学习教程视频。不借助任何深度学习框架，只用 Python 和 NumPy，一步一步把神经网络里的每个计算写出来、画出来。视频里右侧代码面板出现的每一行代码，都能在这个仓库里找到并直接运行。

仓库地址：https://github.com/youxuhu/dl-from-scratch-visual

## 快速开始

```bash
pip install -r requirements.txt
cd lecture01
python matmul.py        # 每个脚本都能单独运行，结果打印在终端
cd ..
python -m pytest -q     # 检查视频里显示的数值和代码算出来的一致
QUICK=1 python lecture08/train_numerical.py   # 训练类脚本加 QUICK=1 只跑很少几步
```

脚本不会弹出窗口；需要画图的脚本把图片存到被 .gitignore 忽略的 `output/` 目录。

## 每讲目录

| 讲次 | 标题 | 目录 | 状态 |
|---|---|---|---|
| 第0讲 | 引子：深度学习能做什么 | [lecture00/](lecture00/) | 代码已完成 |
| 第1讲 | Python 与 NumPy 基础 | [lecture01/](lecture01/) | 代码已完成 |
| 第2讲 | Matplotlib 画图 | [lecture02/](lecture02/) | 代码已完成 |
| 第3讲 | Pandas 够用就好 | [lecture03/](lecture03/) | 代码已完成 |
| 第4讲 | 感知机 | [lecture04/](lecture04/) | 代码已完成 |
| 第5讲 | 神经网络（上）：激活函数与多维数组运算 | [lecture05/](lecture05/) | 代码已完成 |
| 第6讲 | 神经网络（下）：3层网络、输出层与手写数字识别 | [lecture06/](lecture06/) | 代码已完成 |
| 第7讲 | 神经网络的学习（上）：从数据中学习与损失函数 | [lecture07/](lecture07/) | 代码已完成 |
| 第8讲 | 神经网络的学习（下）：数值微分、梯度与学习算法 | [lecture08/](lecture08/) | 代码已完成 |
| 第9讲 | 误差反向传播法（上）：计算图与链式法则 | [lecture09/](lecture09/) | 代码已完成 |
| 第10讲 | 误差反向传播法（下）：激活函数层、Affine/Softmax 层与完整实现 | [lecture10/](lecture10/) | 代码已完成 |
| 第11讲 | 与学习相关的技巧（上）：参数更新与权重初始值 | [lecture11/](lecture11/) | 代码已完成 |
| 第12讲 | 与学习相关的技巧（下）：Batch Normalization、正则化与超参数验证 | [lecture12/](lecture12/) | 代码已完成 |
| 第13讲 | 卷积神经网络（上）：卷积层与池化层 | [lecture13/](lecture13/) | 代码已完成 |
| 第14讲 | 卷积神经网络（下）：im2col、CNN 的实现与可视化 | [lecture14/](lecture14/) | 代码已完成 |
| 第15讲 | 深度学习：加深网络、历史、高速化、应用与未来 | [lecture15/](lecture15/) | 代码已完成 |

## 目录结构

```
common/      各讲共用的函数和类：functions（激活、softmax、损失）、gradient（数值微分与梯度）、
             mnist（下载与读取）、arrays（第1讲的小工具）、simple_net（第6讲的推理网络），
             以及第9讲起的各种层、优化器、im2col 等
dataset/     MNIST 缓存（第一次运行时自动下载，不进仓库）
lecture00/ … lecture15/   每讲一个目录，各有 README（讲什么、文件、怎么运行、关键数值、依赖）
tests/       pytest 测试，核对视频中显示的数值
```

约定：前面讲里写出来、后面还要用的函数和类都放在 common/，后面各讲直接 import，不重复实现。向量一律写成行向量，`Y = np.dot(X, W) + b`；类型转换用 `astype(int)`。

## 致谢

本仓库的代码改编自斋藤康毅（Koki Saitoh）所著《深度学习入门：基于Python的理论与实现》的官方源代码（https://github.com/oreilly-japan/deep-learning-from-scratch）。该源码以 MIT 许可发布，我们在 MIT 许可下改编使用，版权声明见 [LICENSE](LICENSE)。书中的正文和插图受版权保护，本教程没有使用原书正文或插图，台词和画面均为自行编写、绘制。

## 许可证

MIT，见 [LICENSE](LICENSE)。
