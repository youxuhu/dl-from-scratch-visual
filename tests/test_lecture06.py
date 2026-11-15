"""第6讲：3 层前向传播、softmax、MNIST 推理精度。"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lecture06"))

import mnist_inference  # noqa: E402
import softmax_demo  # noqa: E402
import three_layer  # noqa: E402
import train_weights  # noqa: E402
from common import mnist  # noqa: E402
from common.gradient import numerical_gradient  # noqa: E402

needs_mnist = pytest.mark.skipif(not mnist.CACHE.exists(), reason="MNIST 还没下载")


def test_three_layer_forward():
    s = three_layer.steps
    np.testing.assert_allclose(np.round(s["A1"], 3), [0.1, -0.15, 0.5])
    np.testing.assert_allclose(np.round(s["Z1"], 3), [0.525, 0.463, 0.622])
    np.testing.assert_allclose(np.round(s["A2"], 3), [0.376, 0.032])
    np.testing.assert_allclose(np.round(s["Z2"], 3), [0.593, 0.508])
    np.testing.assert_allclose(np.round(s["Y"], 3), [0.262, -0.042])
    assert three_layer.shape_chain == [(2,), (2, 3), (3,), (3, 2), (2,), (2, 2), (2,)]


def test_softmax_numbers():
    np.testing.assert_allclose(np.round(softmax_demo.exp_a, 3), [3.320, 22.198, 1.492])
    assert round(float(softmax_demo.total), 3) == 27.01
    np.testing.assert_allclose(np.round(softmax_demo.p, 3), [0.123, 0.822, 0.055])
    bad, shifted, good = softmax_demo.overflow_demo()
    assert np.isnan(bad).all()
    np.testing.assert_allclose(shifted, [0, -2, -5])
    np.testing.assert_allclose(np.round(good, 3), [0.876, 0.118, 0.006])


def test_handwritten_backprop_matches_numerical():
    """train_weights 里手写的反向传播和数值梯度一致（用很小的网络和 3 个样本）。"""
    rng = np.random.default_rng(0)
    params = {"W1": rng.normal(0, 0.5, (4, 3)), "b1": rng.normal(0, 0.1, 3),
              "W2": rng.normal(0, 0.5, (3, 5)), "b2": rng.normal(0, 0.1, 5),
              "W3": rng.normal(0, 0.5, (5, 2)), "b3": rng.normal(0, 0.1, 2)}
    x = rng.normal(0, 1, (3, 4))
    t = np.array([0, 1, 1])
    _, g = train_weights.loss_and_grads(params, x, t)
    for k in params:
        num = numerical_gradient(lambda _: train_weights.loss_and_grads(params, x, t)[0], params[k])
        np.testing.assert_allclose(g[k], num, atol=1e-6, err_msg=k)


def test_shipped_weights_shape():
    net = mnist_inference.init_network()
    assert net["W1"].shape == (784, 50) and net["W2"].shape == (50, 100) and net["W3"].shape == (100, 10)


@needs_mnist
def test_mnist_accuracy():
    x, t = mnist_inference.get_data()
    net = mnist_inference.init_network()
    assert int(np.argmax(mnist_inference.predict(net, x[0]))) == 7 == int(t[0])
    y = mnist_inference.predict(net, x[:100])
    assert y.shape == (100, 10)
    assert int(np.sum(np.argmax(y, axis=1) == t[:100])) == 100
    assert mnist_inference.accuracy_batched(net, x, t) == pytest.approx(0.9743)
    assert mnist_inference.accuracy_one_by_one(net, x[:1000], t[:1000]) == \
        mnist_inference.accuracy_batched(net, x[:1000], t[:1000])
