"""辅助例子：鸢尾花（Iris）数据集里挑的 12 行，每种 4 行。

数据随仓库提供（data/iris_sample.csv），不需要联网。这里演示 groupby、
value_counts，以及把类别列变成 one hot。
"""
from pathlib import Path

import pandas as pd

IRIS_CSV = Path(__file__).resolve().parent / "data" / "iris_sample.csv"

iris = pd.read_csv(IRIS_CSV)
counts = iris["species"].value_counts().sort_index()
group_mean = iris.groupby("species")["petal_length"].mean()
long_petal = iris[iris["petal_length"] > 5.0]
X = iris[["petal_length", "petal_width"]].to_numpy()
one_hot = pd.get_dummies(iris["species"]).astype(int)
T = one_hot.to_numpy()


def main():
    print("shape =", iris.shape)
    print(counts)
    print("每种花的平均花瓣长度：")
    print(group_mean.round(3))
    print("花瓣长度 > 5.0 的有", len(long_petal), "行：", list(long_petal["species"]))
    print("X.shape =", X.shape)
    print("one hot 的前 2 行和后 1 行：")
    print(one_hot.iloc[[0, 1, -1]])
    print("T.shape =", T.shape)


if __name__ == "__main__":
    main()
