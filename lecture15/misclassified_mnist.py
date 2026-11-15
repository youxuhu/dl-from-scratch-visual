"""第15讲：把 DeepConvNet 认错的样本挑出来，预测和真实标签并排显示。

QUICK=1 只看前 300 张测试图。装了 matplotlib 时把前 20 张错例存成 lecture15/output/misclassified.png。

改编自《深度学习入门：基于Python的理论与实现》官方源码 ch08/misclassified_mnist.py
（MIT 许可，Copyright (c) 2016 Koki Saitoh）。思路与原书一致（逐批推理、挑出预测和标签不同的样本并排显示）；参数改为读取本仓库自己训练的结果，注释改写成中文。
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # 小矩阵用多线程反而慢，机器忙时差几十倍
import numpy as np  # noqa: E402

from _trained import trained_deepnet  # noqa: E402
from common.util import get_mnist, get_pyplot, is_quick  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"


def find_mistakes(net, x, t, batch_size=100):
    preds = np.concatenate([net.predict(x[i:i + batch_size], train_flg=False).argmax(axis=1)
                            for i in range(0, len(x), batch_size)])
    wrong = np.where(preds != t)[0]
    return preds, wrong


def main():
    quick = is_quick()
    net, how = trained_deepnet(quick)
    _, (x_test, t_test), source = get_mnist(flatten=False, test_limit=300 if quick else None)
    preds, wrong = find_mistakes(net, x_test, t_test)
    print(f"参数：{how}；数据：{source} {len(x_test)} 张")
    print(f"精度 {1 - len(wrong) / len(x_test):.4f}，认错 {len(wrong)} 张")
    for i in wrong[:10]:
        print(f"  第 {i} 张：预测 {preds[i]}，真实 {t_test[i]}")
    plt = get_pyplot(OUT)
    if plt is not None and len(wrong):
        fig, axes = plt.subplots(4, 5, figsize=(7, 6))
        for ax in axes.ravel():
            ax.axis("off")
        for ax, i in zip(axes.ravel(), wrong[:20]):
            ax.imshow(x_test[i, 0], cmap="gray_r")
            ax.set_title(f"{preds[i]} / {t_test[i]}", fontsize=9)
        fig.savefig(OUT / "misclassified.png")
        plt.close(fig)


if __name__ == "__main__":
    main()
