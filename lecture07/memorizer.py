"""训练数据和测试数据：一个「背答案」的模型在训练集上全对，在测试集上和瞎猜差不多。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from common.mnist import load_mnist  # noqa: E402

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=False)
N = 2000


class Memorizer:
    """把见过的每张图原封不动记下来；没见过的图就随便猜一个数字。"""

    def __init__(self, seed=0):
        self.book = {}
        self.rng = np.random.default_rng(seed)

    def fit(self, images, labels):
        for img, lab in zip(images, labels):
            self.book[img.tobytes()] = int(lab)

    def predict(self, images):
        return np.array([self.book.get(img.tobytes(), int(self.rng.integers(10))) for img in images])


model = Memorizer()
model.fit(x_train[:N], t_train[:N])
train_acc = float(np.mean(model.predict(x_train[:N]) == t_train[:N]))
test_acc = float(np.mean(model.predict(x_test[:N]) == t_test[:N]))


def main():
    print(f"训练集 {len(x_train)} 张，测试集 {len(x_test)} 张")
    print(f"背下前 {N} 张训练图：训练精度 {train_acc}，测试精度 {test_acc}")


if __name__ == "__main__":
    main()
