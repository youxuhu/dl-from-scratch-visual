"""主例：一张训练日志表（epoch、loss、acc）。读进来、看一看、筛一筛、转成数组、画学习曲线。"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
LOG_CSV = HERE / "data" / "training_log.csv"

df = pd.read_csv(LOG_CSV)
stats = df.describe()
good = df[df["acc"] > 0.9]
arr = df[["loss", "acc"]].to_numpy()
best_row = df.loc[df["acc"].idxmax()]


def with_gap():
    """造一个缺失值：第 4 个 epoch 的 acc 没记上。"""
    gap = df.copy()
    gap.loc[3, "acc"] = float("nan")
    return gap


def learning_curve():
    out = HERE / "output"
    out.mkdir(exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(6, 3.5))
    ax1.plot(df["epoch"], df["loss"], marker="o", label="loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax2 = ax1.twinx()
    ax2.plot(df["epoch"], df["acc"], marker="s", linestyle="--", color="tab:green", label="acc")
    ax2.set_ylabel("acc")
    path = out / "learning_curve.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def main():
    print(df.head(3))
    print("shape =", df.shape)
    print("columns =", list(df.columns))
    print(df.dtypes)
    print(stats.loc[["mean", "min", "max"]].round(4))
    print("acc > 0.9 的行：")
    print(good)
    print("arr.shape =", arr.shape)
    print("acc 最高的一行：epoch", int(best_row["epoch"]), "acc", best_row["acc"])
    gap = with_gap()
    print("缺失值个数 =", int(gap["acc"].isnull().sum()))
    print("dropna 后行数 =", len(gap.dropna()))
    filled = gap["acc"].fillna(gap["acc"].mean())
    print("fillna(其余 7 个的平均) 填上的值 =", round(float(filled[3]), 4))
    back = pd.DataFrame(arr, columns=["loss", "acc"])
    print("数组变回表格 shape =", back.shape)
    print("图已保存:", learning_curve())


if __name__ == "__main__":
    main()
