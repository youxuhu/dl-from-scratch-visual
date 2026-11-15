"""给第6讲准备一组训练好的权重：784 → 50 → 100 → 10，sigmoid 隐藏层，softmax 输出。

这一讲只讲推理，训练的原理在第7、8讲（梯度）和第9、10讲（反向传播）。
这里先用一段手写的反向传播把权重训练出来。仓库里的
weights/mnist_784_50_100_10.npz 是已经提交的一份（测试精度 0.9743），
默认不会被覆盖；完整训练把权重写到 output/，并用 --out / --log 指定别的路径。

QUICK=1 时只训练 50 步，结果默认存到 output/，不覆盖仓库里的权重。
"""
import argparse
import json
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
SHIPPED_WEIGHTS = HERE / "weights" / "mnist_784_50_100_10.npz"
SEED = 6
LR = 0.5
BATCH = 100


def init_params(seed=SEED):
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


def evaluate_test_accuracy(params):
    _, (x_test, t_test) = load_mnist(normalize=True, flatten=True)
    z1 = sigmoid(x_test @ params["W1"] + params["b1"])
    z2 = sigmoid(z1 @ params["W2"] + params["b2"])
    y = softmax(z2 @ params["W3"] + params["b3"])
    return float(np.mean(np.argmax(y, axis=1) == t_test))


def train(steps, lr=LR, batch=BATCH, seed=SEED):
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


def write_log(path, steps, losses, seconds, test_acc, seed=SEED, lr=LR, batch=BATCH, every=100):
    last_100 = float(np.mean(losses[-100:])) if losses else None
    first_n = min(100, len(losses))
    recorded = [round(float(v), 6) for v in losses[every - 1::every]]
    log = {
        "seed": seed,
        "batch_seed": seed + 1,
        "lr": lr,
        "batch": batch,
        "steps": steps,
        "sizes": SIZES,
        "seconds": round(float(seconds), 3),
        "first_100_loss": round(float(np.mean(losses[:first_n])), 6) if losses else None,
        "last_100_loss": round(last_100, 6) if last_100 is not None else None,
        "test_accuracy": round(float(test_acc), 4) if test_acc is not None else None,
        "loss_every": every,
        "loss": recorded,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(log, ensure_ascii=False, indent=1) + "\n")
    return log


def parse_args(argv=None):
    quick = os.environ.get("QUICK") == "1"
    parser = argparse.ArgumentParser(description="训练第6讲的 3 层网络权重")
    parser.add_argument("--out", type=Path, default=None,
                        help="权重保存路径。默认：QUICK 时为 output/quick_weights.npz，"
                             "完整训练为 output/retrained_weights.npz（不会覆盖仓库里已提交的权重）")
    parser.add_argument("--log", type=Path, default=None,
                        help="训练日志 JSON 路径。默认：QUICK 时为 output/train_log_quick.json，"
                             "完整训练为 results/train_log.json")
    parser.add_argument("--steps", type=int, default=None, help="训练步数，默认 QUICK 50、完整 12000")
    args = parser.parse_args(argv)
    args.quick = quick
    args.steps = args.steps if args.steps is not None else (50 if quick else 12000)
    if args.out is None:
        args.out = HERE / "output" / ("quick_weights.npz" if quick else "retrained_weights.npz")
    if args.log is None:
        args.log = HERE / ("output/train_log_quick.json" if quick else "results/train_log.json")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.out.resolve() == SHIPPED_WEIGHTS.resolve():
        raise SystemExit(f"拒绝覆盖仓库里已提交的权重：{SHIPPED_WEIGHTS}。请把 --out 指到别的路径。")
    start = time.time()
    params, losses = train(args.steps)
    seconds = time.time() - start
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **{k: v.astype(np.float32) for k, v in params.items()})
    test_acc = evaluate_test_accuracy(params)
    log = write_log(args.log, args.steps, losses, seconds, test_acc)
    print(f"训练 {args.steps} 步，用时 {seconds:.1f} 秒")
    print("前 / 后 100 步平均 loss:", log["first_100_loss"], log["last_100_loss"])
    print("这次训练的测试集精度:", log["test_accuracy"])
    print("权重已保存:", args.out, f"({args.out.stat().st_size / 1024:.0f} KB)")
    print("训练日志已保存:", args.log)


if __name__ == "__main__":
    main()
