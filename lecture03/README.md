# 第3讲 Pandas 够用就好

## 这讲讲什么
鱼书正文没有 Pandas，这是本课补充的预备讲，讲到够用为止：Series 和 DataFrame、read_csv、head/shape/columns/dtypes/describe、按列取值和条件筛选、loc、缺失值（isnull、dropna、fillna）、groupby 和 value_counts、和 NumPy 互转、把类别列变成 one hot。

主例是一张训练日志表（epoch、loss、acc），读进来直接画学习曲线，正好接上后面各讲；辅助例子是 Iris。

## 文件
| 文件 | 内容 |
|---|---|
| data/training_log.csv | 自己编的 8 个 epoch 的训练日志 |
| data/iris_sample.csv | 鸢尾花数据集里挑的 12 行（每种 4 行），随仓库提供，不用联网。已和 UCI 原始文件 iris.data（https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data）逐行核对，完全一致，对应其中第 0 到 3、50 到 53、100 到 103 行（从 0 数）；物种名去掉了 `Iris-` 前缀 |
| training_log.py | 读表、统计、筛选、缺失值、转数组、画学习曲线（存到 output/） |
| iris_demo.py | value_counts、groupby 求均值、筛选、to_numpy、get_dummies 做 one hot |

## 怎么运行
```bash
cd lecture03
python training_log.py
python iris_demo.py
```

## 视频里出现的数值
- 训练日志 shape (8, 3)，列 epoch、loss、acc；head(3) 的 loss 是 2.0734、1.1835、0.6912。
- describe：loss 均值 0.7427，acc 均值 0.8079，acc 最小 0.4120、最大 0.9157。
- `df[df['acc'] > 0.9]` 选出 epoch 6、7、8 三行；acc 最高是 epoch 8。
- `df[['loss', 'acc']].to_numpy()` 形状 (8, 2)。
- 把 epoch 4 的 acc 去掉：缺失 1 个，dropna 后 7 行，用其余 7 个的平均 fillna 填上 0.7989。
- Iris 样本 shape (12, 5)，三种各 4 行；平均花瓣长度 setosa 1.400、versicolor 4.525、virginica 5.650；花瓣长度 > 5.0 的 4 行全是 virginica；one hot 矩阵形状 (12, 3)。

## 依赖
- 第1讲：布尔索引的思想（知识前置）；第2讲：画学习曲线（知识前置）。
- 不 import common/ 的代码；数据随本目录提供。

## 和大纲的不同
- 大纲画面里写的是「6 行 4 列的学生表（身高、体重、成绩、班级）」，主控已定改用训练日志表，所以这里是 8 行 3 列；条件筛选用 `acc > 0.9` 代替 `成绩 > 80`，to_numpy 演示用 `df[['loss', 'acc']]` 代替 `df[['身高','体重']]`。
- groupby 和 one hot 放到 Iris 上演示（训练日志没有类别列）。
- 缺失值用 fillna(平均值) 演示，而不是插值。
