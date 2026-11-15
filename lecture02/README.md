# 第2讲 Matplotlib 画图

## 这讲讲什么
plot 就是把一堆点连起来：采样点、sin 和 cos、xlabel/ylabel/title/legend；imshow 把二维数组显示成灰度图（为后面看 MNIST、卷积核、权重做准备）；imread 读图；损失曲线和滑动平均（以后每一讲都会再见）；subplot 并排多张图、ylim。

所有脚本都用 Agg 后端，不弹窗口，图片存进本目录的 `output/`（已被 .gitignore 忽略）。

## 文件
| 文件 | 内容 |
|---|---|
| _plotting.py | 设置 Agg 后端和输出目录 |
| sin_cos.py | 采样点 → sin/cos 曲线（本讲动画用的稀疏采样 np.arange(0, 7, 0.2)） |
| sin_cos_graph.py | 对齐鱼书 ch01/sin_cos_graph.py：np.arange(0, 6, 0.1)，共 60 个点 |
| array_image.py | 5×5 数组 imshow、imsave/imread 往返；有 MNIST 缓存时显示测试集第一张图 |
| loss_curve.py | 模拟的 1 万步损失记录 + 200 步滑动平均 |
| subplots.py | 1×3 子图 |

## 怎么运行
```bash
cd lecture02
python sin_cos.py
python sin_cos_graph.py
python array_image.py   # QUICK=1 且没有 MNIST 缓存时跳过数字图，不联网
python loss_curve.py
python subplots.py
```

## 视频里出现的数值
- `x = np.arange(0, 7, 0.2)`：前几个值 `[0. 0.2 0.4 0.6]`，共 35 个点；sin 为正 18 个（正值色）、为负 16 个（负值色），x = 0 处为 0；x = 1.6 时 sin = 0.9996。
- 5×5 数组：最亮格 (2, 2) 值 1.0，值 ≥ 0.8 的格子 9 个（亮十字）；imread 读回形状 (5, 5, 4)。
- MNIST 测试集第一张图标签为 7。
- 损失曲线（固定种子 2026 造的模拟数据，不是真实训练）：前 3 个 loss 为 2.237、2.317、2.145；滑动平均从 2.137 降到 0.151。

## 依赖
- 第1讲：NumPy 数组、np.arange（只是知识上的前置，没有 import 第1讲的代码）。
- common/mnist.py：`load_mnist`（只在 array_image.py 里显示测试集第一张图时用）。

## 和大纲的不同
- 大纲写的采样点是 `np.arange(0, 6, 0.1)`（60 个点），这里改成自己的 `np.arange(0, 7, 0.2)`（35 个点），点少一些更容易逐点动画。
- 大纲说损失曲线用「预先生成的真实下降数据」；这一讲还没讲训练，先用固定种子生成的模拟曲线，并在台词里说明。真实曲线在第8讲由训练脚本生成。
- 不用 lena 图，imread 演示读的是自己存出来的 5×5 灰度图。
