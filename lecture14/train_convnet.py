"""第14讲：训练 SimpleConvNet（Conv → ReLU → Pool → Affine → ReLU → Affine → Softmax）。

完整训练：60000 张训练图，20 个 epoch，批大小 100，Adam 学习率 0.001，用 common/trainer.py 的 Trainer。
QUICK=1：2000 张训练图、300 张测试图、3 个 epoch。训练完把参数存到 lecture14/output/simple_convnet_params.pkl。
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # 小矩阵用多线程反而慢，机器忙时差几十倍
from common.simple_convnet import SimpleConvNet  # noqa: E402
from common.trainer import Trainer  # noqa: E402
from common.util import get_mnist, is_quick  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"
PARAMS = OUT / "simple_convnet_params.pkl"


def main():
    quick = is_quick()
    (x_train, t_train), (x_test, t_test), source = get_mnist(
        flatten=False, train_limit=2000 if quick else None, test_limit=300 if quick else None)
    net = SimpleConvNet(input_dim=(1, 28, 28),
                        conv_param={"filter_num": 30, "filter_size": 5, "pad": 0, "stride": 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01, seed=0)
    print("各层输出形状：", " → ".join("×".join(map(str, s)) for s in net.shapes))
    trainer = Trainer(net, x_train, t_train, x_test, t_test,
                      epochs=3 if quick else 20, mini_batch_size=100,
                      optimizer="Adam", optimizer_param={"lr": 0.001},
                      evaluate_sample_num_per_epoch=1000, seed=0)
    acc = trainer.train()
    OUT.mkdir(exist_ok=True)
    net.save_params(PARAMS)
    print(f"数据：{source}；测试精度 {acc:.4f}；参数已保存到 {PARAMS}")


if __name__ == "__main__":
    main()
