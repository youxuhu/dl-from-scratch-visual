"""给第6讲准备一组训练好的权重：784 → 50 → 100 → 10，sigmoid 隐藏层，softmax 输出。

这一讲只讲推理，训练的原理在第7、8讲（梯度）和第9、10讲（反向传播）。
这里先用一段手写的反向传播把权重训练出来，存成 weights/mnist_784_50_100_10.npz。

QUICK=1 时只训练 50 步，结果存到 output/，不覆盖仓库里的权重。
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from common.functions import cross_entropy_error, sigmoid, softmax  # noqa: E402
from common.mnist import load_mnist  # noqa: E402

HERE = Path(__file__).resolve().parent
SIZES = [784, 50, 100, 10]
WEIGHTS = HERE / "weights" / "mnist_784_50_100_10.npz"


def init_params(seed=6):
    rng = np.random.default_rng(seed)
    params = {}
    for i, (n_in, n_out) in enumerate(zip(SIZES[:-1], SIZES[1:]), start=1):
        params[f"W{i}"] = rng.standard_normal((n_in, n_out)) / np.sqrt(n_in)
        params[f"b{i}"] = np.zeros(n_out)
    return params


def loss_and_grads(params, x, t):
    """前向 + 手写反向传播（第10讲会把它拆成一层一层的类）。"""
    a1 = x @ params["W1"] + params["b1"]
    z1 = sigmoid(a1)
    a2 = z1 @ params["W2"] + params["b2"]
    z2 = sigmoid(a2)
    y = softmax(z2 @ params["W3"] + params["b3"])
    n = x.shape[0]
    delta3 = y.copy()
    delta3[np.arange(n), t] -= 1.0
    delta3 /= n
    g = {"W3": z2.T @ delta3, "b3": delta3.sum(axis=0)}
    delta2 = (delta3 @ params["W3"].T) * z2 * (1 - z2)
    g["W2"], g["b2"] = z1.T @ delta2, delta2.sum(axis=0)
    delta1 = (delta2 @ params["W2"].T) * z1 * (1 - z1)
    g["W1"], g["b1"] = x.T @ delta1, delta1.sum(axis=0)
    return cross_entropy_error(y, t), g


def train(steps, lr=0.5, batch=100, seed=6):
    (x_train, t_train), _ = load_mnist(normalize=True, flatten=True)
    params = init_params(seed)
    rng = np.random.default_rng(seed + 1)
    losses = []
    for _ in range(steps):
        pick = rng.choice(len(x_train), batch, replace=False)
        loss, g = loss_and_grads(params, x_train[pick], t_train[pick])
        for k in params:
            params[k] -= lr * g[k]
        losses.append(loss)
    return params, losses


def main():
    quick = os.environ.get("QUICK") == "1"
    steps = 50 if quick else 12000
    start = time.time()
    params, losses = train(steps)
    target = HERE / "output" / "quick_weights.npz" if quick else WEIGHTS
    target.parent.mkdir(exist_ok=True)
    np.savez_compressed(target, **{k: v.astype(np.float32) for k, v in params.items()})
    print(f"训练 {steps} 步，用时 {time.time() - start:.1f} 秒")
    print("前 / 后 100 步平均 loss:", round(float(np.mean(losses[:100])), 4),
          round(float(np.mean(losses[-100:])), 4))
    print("权重已保存:", target, f"({target.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
