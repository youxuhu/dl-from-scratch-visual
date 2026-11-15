"""第12讲：用 Dropout 缓解过拟合。

300 个训练样本，7 层网络 784 → 100×6 → 10，dropout_ratio = 0.2，SGD 学习率 0.01，301 个 epoch，用 Trainer 训练。
和不用 Dropout 的同一网络对比训练、测试精度的差距。QUICK=1：150 个 epoch，测试集 1000 张。

改编自《深度学习入门：基于Python的理论与实现》官方源码 ch06/overfit_dropout.py
（MIT 许可，Copyright (c) 2016 Koki Saitoh）。实验设置（300 个样本、7 层网络、dropout_ratio = 0.2、用 Trainer、301 个 epoch）与原书一致；注释改写成中文，保留中间结果打印和 QUICK 模式。
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # 小矩阵用多线程反而慢，机器忙时差几十倍
from common.multi_layer_net_extend import MultiLayerNetExtend  # noqa: E402
from common.trainer import Trainer  # noqa: E402
from common.util import get_mnist, is_quick  # noqa: E402


def run(x_train, t_train, x_test, t_test, use_dropout, epochs, seed=0):
    net = MultiLayerNetExtend(784, [100] * 6, 10, use_dropout=use_dropout, dropout_ration=0.2, seed=seed)
    trainer = Trainer(net, x_train, t_train, x_test, t_test, epochs=epochs, mini_batch_size=100,
                      optimizer="SGD", optimizer_param={"lr": 0.01}, verbose=False, seed=seed)
    trainer.train()
    return trainer.train_acc_list, trainer.test_acc_list


def main():
    quick = is_quick()
    (x_train, t_train), (x_test, t_test), source = get_mnist(
        train_limit=300, test_limit=1000 if quick else None)
    epochs = 150 if quick else 301
    print(f"数据：{source}，训练 300 张，{epochs} 个 epoch")
    for flag in (False, True):
        tr, te = run(x_train, t_train, x_test, t_test, flag, epochs)
        name = "用 Dropout(0.2)" if flag else "不用 Dropout"
        print(f"{name}：训练精度 {tr[-1]:.3f}，测试精度 {te[-1]:.3f}，差距 {tr[-1] - te[-1]:.3f}")


if __name__ == "__main__":
    main()
