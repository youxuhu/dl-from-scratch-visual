"""第8讲：数值微分、梯度、梯度下降、SimpleNet、NumericalTwoLayerNet。"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lecture08"))

import derivative  # noqa: E402
import gradient_demo  # noqa: E402
import weight_gradient as simple_net  # noqa: E402
from common import mnist  # noqa: E402
from common.functions import softmax  # noqa: E402
from lecture08.numerical_two_layer_net import NumericalTwoLayerNet  # noqa: E402

needs_mnist = pytest.mark.skipif(not mnist.CACHE.exists(), reason="MNIST 还没下载")


def test_derivative_table():
    assert derivative.TRUE_SLOPE == pytest.approx(0.1)
    fwd = [round(r[1], 6) for r in derivative.table]
    ctr = [round(r[2], 6) for r in derivative.table]
    assert fwd == [0.15, 0.105, 0.1005, 0.100005]
    assert ctr == [0.1, 0.1, 0.1, 0.1]
    assert derivative.tiny_result == 0.0 and derivative.tiny_float32 == 0.0
    assert derivative.tangent_line(0.0) == pytest.approx(-0.8)


def test_gradient_and_descent():
    np.testing.assert_allclose(gradient_demo.grad_at_point, [4.0, -4.0], atol=1e-6)
    _, arrows = gradient_demo.gradient_field()
    np.testing.assert_allclose(arrows[-1], [-4.0, -8.0], atol=1e-6)
    end, path = gradient_demo.runs[0.1]
    np.testing.assert_allclose(path[1], [-1.6, 1.8])
    assert np.abs(end).max() < 1e-4
    end, path = gradient_demo.runs[0.6]
    np.testing.assert_allclose(path[1], [0.4, -4.2])
    assert abs(end[1]) > 1e7                  # x1 方向发散
    end, path = gradient_demo.runs[1e-4]
    np.testing.assert_allclose(path[1], [-1.9996, 2.9988])
    np.testing.assert_allclose(end, [-1.98009769, 2.94058425], atol=1e-6)


def test_simple_net():
    np.testing.assert_allclose(np.round(simple_net.scores, 3), [0.49, -0.36, 0.43])
    assert round(simple_net.loss_before, 4) == 1.7125
    np.testing.assert_allclose(np.round(simple_net.dW, 4),
                               [[0.3377, -0.6557, 0.318], [-0.211, 0.4098, -0.1988]])
    assert round(simple_net.loss_after, 4) == 1.2888
    # dW 和解析式 x^T (softmax - onehot) 一致
    p = softmax(simple_net.scores)
    p[simple_net.t] -= 1
    np.testing.assert_allclose(simple_net.dW, np.outer(simple_net.x, p), atol=1e-6)


def test_numerical_two_layer_net_interface_and_gradient_check():
    net = NumericalTwoLayerNet(4, 3, 2, weight_init_std=0.5, seed=0)
    assert set(net.params) == {"W1", "b1", "W2", "b2"}
    rng = np.random.default_rng(1)
    x = rng.normal(size=(5, 4))
    t = np.array([0, 1, 1, 0, 1])
    assert net.predict(x).shape == (5, 2)
    assert net.loss(x, t) == pytest.approx(net.loss(x, np.eye(2)[t]))
    assert 0.0 <= net.accuracy(x, t) <= 1.0
    g = net.numerical_gradient(x, t)
    assert {k: v.shape for k, v in g.items()} == {k: v.shape for k, v in net.params.items()}
    # 和手推的解析梯度对比（梯度确认）
    p = net.params
    a1 = x @ p["W1"] + p["b1"]
    z1 = 1 / (1 + np.exp(-a1))
    y = softmax(z1 @ p["W2"] + p["b2"])
    dy = y.copy()
    dy[np.arange(5), t] -= 1
    dy /= 5
    da1 = (dy @ p["W2"].T) * z1 * (1 - z1)
    analytic = {"W2": z1.T @ dy, "b2": dy.sum(0), "W1": x.T @ da1, "b1": da1.sum(0)}
    for k in analytic:
        np.testing.assert_allclose(g[k], analytic[k], atol=1e-7, err_msg=k)


def test_default_init_std_and_seed():
    a = NumericalTwoLayerNet(784, 50, 10, seed=3)
    b = NumericalTwoLayerNet(784, 50, 10, seed=3)
    np.testing.assert_array_equal(a.params["W1"], b.params["W1"])
    assert 0.008 < a.params["W1"].std() < 0.012
    assert not a.params["b1"].any()


@needs_mnist
def test_quick_training_runs():
    import train_numerical
    assert train_numerical.shrink_to_7x7(np.zeros((2, 784))).shape == (2, 49)
    net, log = train_numerical.train(iters=5)
    assert len(log["loss"]) == 5 and np.isfinite(log["loss"]).all()
