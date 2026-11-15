# 第7讲 神经网络的学习（上）：从数据中学习与损失函数

## 这讲讲什么
「从数据中学习」是什么意思；训练数据和测试数据、泛化和过拟合；均方误差和交叉熵误差；one hot 标签和标签下标两种写法；mini-batch：随机抽 100 张算平均损失；为什么学习时看损失而不看精度（精度对参数的微小变化不敏感，导数几乎处处是 0）。

## 文件
| 文件 | 内容 |
|---|---|
| losses.py | 标签 3 的 one hot；猜对和猜错两组输出的均方误差、交叉熵，逐项计算 |
| cross_entropy_curve.py | -log(p) 曲线和 4 个点，存图到 output/ |
| minibatch.py | 随机抽 100 张训练图，用第6讲的网络算这一批的平均交叉熵 |
| accuracy_vs_loss.py | 只拧一个权重 W3[0, 7]，精度是台阶、损失是平滑曲线 |
| memorizer.py | 「背答案」模型：训练集全对，测试集和瞎猜差不多（过拟合的极端例子） |

## 怎么运行
```bash
cd lecture07
python losses.py
python cross_entropy_curve.py
python minibatch.py          # 需要 MNIST（第一次运行会自动下载）
python accuracy_vs_loss.py
python memorizer.py
```

## 视频里出现的数值
- 正确答案 3。猜对的 y_good = [0.02, 0.05, 0.03, 0.70, 0.04, 0.06, 0.02, 0.03, 0.03, 0.02]；猜错的 y_bad 把第 3 格换成 0.10、第 8 格换成 0.63。两组和都是 1。
- 均方误差逐项：y_good - t 第 3 格是 -0.3，其余是小正数；平方和 0.1016，乘 0.5 得 **0.0508**；猜错时 **0.6088**。
- 交叉熵：猜对 **0.3567**（= -log 0.7），猜错 **2.3026**（= -log 0.1）；用标签下标 3 代替 one hot 结果一样。
- -log(p)：p = 0.9 → 0.105，0.6 → 0.511，0.3 → 1.204，0.1 → 2.303。
- mini-batch：训练集 (60000, 784)，one hot 标签 (60000, 10)；`rng = np.random.default_rng(7)` 抽到的前 5 个下标 [40984, 707, 58170, 5489, 30465]；前 3 行正确类别概率 0.9541、0.9999、0.9957；这一批平均交叉熵 0.0553。
- 精度 vs 损失（前 200 张测试图，W3[0, 7] 原值 0.6326，在 ±6 范围扫 121 个值）：精度只有 5 种值 0.98 / 0.985 / 0.99 / 0.995 / 1.0，损失有 120 种值，从 0.0136 到 0.0479；在原值处精度的斜率是 0.0，损失的斜率是 -0.00115。
- 背答案模型背下前 2000 张训练图：训练精度 1.0，测试精度 0.089。

## 依赖
- 本讲 / common/functions.py：`cross_entropy_error`、`mean_squared_error`（本讲讲解的就是这两个函数，代码只放在 common 里，不在本目录重复定义）。
- 第6讲 / common/simple_net.py：`init_network`、`predict`（第6讲训练好的 784 → 50 → 100 → 10 网络和权重 lecture06/weights/；`predict` 内部会调用 softmax，本讲脚本不单独 import softmax）。
- common/mnist.py：`load_mnist`。

## 和大纲的不同
- 大纲的均方误差例子（标签 2，结果 0.0975 和 0.5975）和书中一样，这里换成自己的：标签 3，结果 0.0508 和 0.6088。
- 交叉熵曲线上的点从「0.6 滑到 0.1」扩展成 0.9、0.6、0.3、0.1 四个点。
- 抽 mini-batch 用 `np.random.default_rng(7).choice(..., replace=False)`，结果可复现；大纲写的是 `np.random.choice(train_size, batch_size)`。
- 新增 accuracy_vs_loss.py 和 memorizer.py，把「台阶上的小球不知道往哪走」和「过拟合」用真实数字演示出来。
