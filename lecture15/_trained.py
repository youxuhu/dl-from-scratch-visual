"""第15讲几个脚本共用：拿到一个训练过的 DeepConvNet（复用 common 里的网络和 Trainer）。

lecture15/output/ 里有 train_deepnet.py 存下的参数就读取；没有就当场用 Adam 训练一小会儿。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.deep_convnet import DeepConvNet  # noqa: E402
from common.trainer import Trainer  # noqa: E402
from common.util import get_mnist  # noqa: E402

PARAMS = Path(__file__).resolve().parent / "output" / "deep_convnet_params.pkl"


def trained_deepnet(quick):
    net = DeepConvNet(seed=0)
    if PARAMS.exists():
        net.load_params(PARAMS)
        return net, "读取 train_deepnet.py 保存的参数"
    (x_tr, t_tr), (x_te, t_te), _ = get_mnist(flatten=False, train_limit=600 if quick else 5000,
                                              test_limit=100)
    Trainer(net, x_tr, t_tr, x_te, t_te, epochs=1 if quick else 3, mini_batch_size=100,
            optimizer="Adam", optimizer_param={"lr": 0.001}, verbose=False, seed=0).train()
    return net, "当场简单训练（没有找到保存的参数）"


if __name__ == "__main__":
    print("这个文件是第15讲其他脚本共用的辅助模块，单独运行不做事。参数文件：", PARAMS)
