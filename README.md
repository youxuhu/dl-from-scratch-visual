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
```

脚本不会弹出窗口；需要画图的脚本把图片存到被 .gitignore 忽略的 `output/` 目录。

## 每讲目录

| 讲次 | 标题 | 目录 | 状态 |
|---|---|---|---|
| 第0讲 | 引子：深度学习能做什么 | [lecture00/](lecture00/) | 已完成 |
| 第1讲 | Python 与 NumPy 基础 | [lecture01/](lecture01/) | 已完成 |
| 第2讲 | Matplotlib 画图 | lecture02/ | 制作中 |
| 第3讲 | Pandas 够用就好 | lecture03/ | 制作中 |
| 第4讲 | 感知机 | lecture04/ | 制作中 |
| 第5讲 | 神经网络（上）：激活函数与多维数组运算 | lecture05/ | 制作中 |
| 第6讲 | 神经网络（下）：3层网络、输出层与手写数字识别 | lecture06/ | 制作中 |
| 第7讲 | 神经网络的学习（上）：从数据中学习与损失函数 | lecture07/ | 制作中 |
| 第8讲 | 神经网络的学习（下）：数值微分、梯度与学习算法 | lecture08/ | 制作中 |
| 第9讲 | 误差反向传播法（上）：计算图与链式法则 | lecture09/ | 制作中 |
| 第10讲 | 误差反向传播法（下）：各种层与完整实现 | lecture10/ | 制作中 |
| 第11讲 | 与学习相关的技巧（上）：参数更新与权重初始值 | lecture11/ | 制作中 |
| 第12讲 | 与学习相关的技巧（下）：Batch Normalization、正则化与超参数 | lecture12/ | 制作中 |
| 第13讲 | 卷积神经网络（上）：卷积层与池化层 | lecture13/ | 制作中 |
| 第14讲 | 卷积神经网络（下）：im2col 与 CNN 的实现 | lecture14/ | 制作中 |
| 第15讲 | 深度学习：加深网络与前沿 | lecture15/ | 制作中 |

## 目录结构

```
common/      各讲共用的小工具（目前有 describe、matmul_steps）
lecture00/   第0讲的代码
lecture01/   第1讲的代码
tests/       pytest 测试，核对视频中显示的数值
```

约定：向量一律写成行向量，`Y = np.dot(X, W) + b`；类型转换用 `astype(int)`。

## 致谢

知识路线参考《深度学习入门：基于Python的理论与实现》，代码为独立实现

## 许可证

MIT，见 [LICENSE](LICENSE)。
