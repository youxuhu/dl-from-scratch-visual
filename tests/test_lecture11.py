"""第11讲：优化器公式、赛跑数值、激活值分布、双胞胎神经元、MultiLayerNet 梯度确认。"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from common.multi_layer_net import MultiLayerNet, init_scale
from common.optimizer import SGD, AdaGrad, Adam, Momentum, Nesterov, RMSprop

HERE = Path(__file__).resolve().parent.parent / "lecture11"
sys.path.insert(0, str(HERE))


def load(name):
    spec = importlib.util.spec_from_file_location(f"l11_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


naive = load("optimizer_compare_naive")
hist = load("weight_init_activation_histogram")
twins = load("zero_init")


def one_step(opt, w=1.0, g=0.5):
    p = {"w": np.array([w])}
    opt.update(p, {"w": np.array([g])})
    return float(p["w"][0])


def test_single_step_formulas():
    assert one_step(SGD(lr=0.1)) == pytest.approx(1.0 - 0.05)
    assert one_step(Momentum(lr=0.1)) == pytest.approx(0.95)          # v 从 0 开始
    assert one_step(AdaGrad(lr=0.1)) == pytest.approx(0.9, abs=1e-6)  # g/sqrt(g²) = 1
    assert one_step(Adam(lr=0.1)) == pytest.approx(0.9, abs=1e-5)     # 偏置修正后第一步约等于 lr
    assert one_step(RMSprop(lr=0.1)) == pytest.approx(1.0 - 0.1 * 0.5 / np.sqrt(0.01 * 0.25), abs=1e-5)
    assert one_step(Nesterov(lr=0.1)) == pytest.approx(1.0 - 0.81 * 0.05 - 1.9 * 0.05)  # 原书写法


def test_optimizers_reach_minimum_of_bowl():
    for opt in [SGD(0.1), Momentum(0.05), AdaGrad(0.5), Adam(0.1), RMSprop(0.05), Nesterov(0.05)]:
        p = {"w": np.array([3.0, -2.0])}
        for _ in range(500):
            opt.update(p, {"w": 2 * p["w"]})
        assert np.all(np.abs(p["w"]) < 0.05), type(opt).__name__


def test_race_numbers():
    end = {k: v[-1] for k, v in naive.paths.items()}
    np.testing.assert_allclose(end["SGD"], [-0.3504, 0.0848], atol=1e-4)
    np.testing.assert_allclose(end["Momentum"], [0.8964, 0.0881], atol=1e-4)
    np.testing.assert_allclose(end["AdaGrad"], [-0.1202, 0.0], atol=1e-4)
    np.testing.assert_allclose(end["Adam"], [0.1360, 0.3182], atol=1e-4)
    m = naive.first_states["Momentum"]
    np.testing.assert_allclose([s["v"] for s in m], [(0.07, -0.4), (0.1323, -0.68), (0.187, -0.796)], atol=1e-4)
    h = naive.first_states["AdaGrad"]
    np.testing.assert_allclose(h[0]["h"], (0.49, 16.0))
    np.testing.assert_allclose(naive.first_states["Adam"][0]["m"], (-0.07, 0.4))
    assert all(len(p) == 31 for p in naive.paths.values())


def test_activation_distributions():
    r = hist.results
    assert all(row["near0"] > 0.15 and row["near1"] > 0.15 for row in r[("sigmoid", "std=1")])
    assert all(row["std"] < 0.03 for row in r[("sigmoid", "std=0.01")])
    assert all(0.1 < row["std"] < 0.25 for row in r[("sigmoid", "Xavier")])
    assert r[("relu", "std=0.01")][2]["near0"] == 1.0
    xav = [row["mean"] for row in r[("relu", "Xavier")]]
    assert xav[0] > xav[-1] * 3
    he = [row["mean"] for row in r[("relu", "He")]]
    assert max(he) / min(he) < 1.3


def test_twin_neurons_never_split():
    W = twins.W1_same
    np.testing.assert_allclose(W[:, 0], W[:, 1])
    assert not np.allclose(W, 0.5)       # 确实更新了，只是两列一起变


def test_init_scale():
    assert init_scale("he", 100) == pytest.approx(np.sqrt(0.02))
    assert init_scale("xavier", 100) == pytest.approx(0.1)
    assert init_scale(0.01, 100) == 0.01


@pytest.mark.parametrize("act,init", [("relu", "he"), ("sigmoid", "xavier")])
def test_multilayer_gradient_check(act, init):
    rng = np.random.default_rng(3)
    net = MultiLayerNet(7, [6, 5], 3, activation=act, weight_init_std=init, seed=4)
    x = rng.random((4, 7))
    t = np.array([0, 2, 1, 2])
    numeric = net.numerical_gradient(x, t)
    analytic = net.gradient(x, t)
    for k in numeric:
        assert np.mean(np.abs(numeric[k] - analytic[k])) < 1e-5, k   # 交叉熵里的 eps=1e-7 带来 1e-6 量级的偏差


def test_compare_mnist_runs_on_toy_data():
    mod = load("optimizer_compare_mnist")
    from common.util import toy_digits
    (x, t), _ = toy_digits(300, 10)
    losses, _ = mod.compare(x, t, max_iterations=15, batch_size=32, report_every=0)
    assert set(losses) == {"SGD", "Momentum", "AdaGrad", "Adam"}
    assert all(len(v) == 15 for v in losses.values())
