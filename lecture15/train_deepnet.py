"""第15讲：训练 DeepConvNet（6 个 3×3 卷积层、He 初始值、Adam、Dropout）。

完整训练：60000 张，20 个 epoch，批大小 100，Adam 学习率 0.001，用 Trainer。纯 NumPy 很慢，要跑几个小时。
QUICK=1：1200 张训练图、200 张测试图、2 个 epoch，只确认流程。参数存到 lecture15/output/deep_convnet_params.pkl。
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # 小矩阵用多线程反而慢，机器忙时差几十倍
from common.deep_convnet import DeepConvNet  # noqa: E402
from common.trainer import Trainer  # noqa: E402
from common.util import get_mnist, is_quick  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"
PARAMS = OUT / "deep_convnet_params.pkl"


def main():
    quick = is_quick()
    (x_train, t_train), (x_test, t_test), source = get_mnist(
        flatten=False, train_limit=1200 if quick else None, test_limit=200 if quick else None)
    net = DeepConvNet(seed=0)
    trainer = Trainer(net, x_train, t_train, x_test, t_test,
                      epochs=2 if quick else 20, mini_batch_size=100,
                      optimizer="Adam", optimizer_param={"lr": 0.001},
                      evaluate_sample_num_per_epoch=200 if quick else 1000, seed=0)
    acc = trainer.train()
    OUT.mkdir(exist_ok=True)
    net.save_params(PARAMS)
    print(f"数据：{source}；测试精度 {acc:.4f}；参数已保存到 {PARAMS}")


if __name__ == "__main__":
    main()
