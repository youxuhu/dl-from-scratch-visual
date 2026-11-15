"""common/functions.py、gradient.py、mnist.py 的基本测试。"""
import numpy as np
import pytest

from common.functions import (cross_entropy_error, identity, mean_squared_error, relu,
                              sigmoid, sigmoid_grad, softmax, step_function)
from common.gradient import gradient_descent, numerical_diff, numerical_gradient
from common import mnist


def test_activations():
    x = np.array([-2.0, 0.0, 3.0])
    assert step_function(x).tolist() == [0, 0, 1]
    assert step_function(x).dtype.kind == "i"
    np.testing.assert_allclose(sigmoid(np.array([0.0])), [0.5])
    np.testing.assert_allclose(relu(x), [0.0, 0.0, 3.0])
    assert identity(x) is x
    assert np.isfinite(sigmoid(np.array([-1000.0, 1000.0]))).all()


def test_sigmoid_grad_matches_numeric():
    x = np.array([-1.5, 0.2, 2.0])
    num = numerical_gradient(lambda v: float(np.sum(sigmoid(v))), x.copy())
    np.testing.assert_allclose(sigmoid_grad(x), num, atol=1e-7)


def test_softmax_1d_2d_and_overflow():
    p = softmax(np.array([1.0, 2.0, 3.0]))
    assert p.sum() == pytest.approx(1.0)
    big = softmax(np.array([2000.0, 1990.0]))
    assert np.isfinite(big).all()
    X = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    P = softmax(X)
    np.testing.assert_allclose(P.sum(axis=1), [1.0, 1.0])
    np.testing.assert_allclose(P[0], p)
    np.testing.assert_allclose(P[1], [1 / 3] * 3)


def test_losses_onehot_and_labels():
    y = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    t_idx = np.array([0, 1])
    t_hot = np.eye(3)[t_idx]
    expect = -(np.log(0.7 + 1e-7) + np.log(0.8 + 1e-7)) / 2
    assert cross_entropy_error(y, t_idx) == pytest.approx(expect)
    assert cross_entropy_error(y, t_hot) == pytest.approx(expect)
    assert cross_entropy_error(y[0], t_hot[0]) == pytest.approx(-np.log(0.7 + 1e-7))
    assert cross_entropy_error(y[0], 0) == pytest.approx(-np.log(0.7 + 1e-7))
    mse = 0.5 * ((0.3 ** 2 + 0.2 ** 2 + 0.1 ** 2) + (0.1 ** 2 + 0.2 ** 2 + 0.1 ** 2)) / 2
    assert mean_squared_error(y, t_hot) == pytest.approx(mse)
    assert mean_squared_error(y, t_idx) == pytest.approx(mse)


def test_numerical_gradient_any_shape_restores_input():
    W = np.array([[1.0, -2.0, 0.5], [3.0, 0.0, -1.0]])
    before = W.copy()
    g = numerical_gradient(lambda w: float(np.sum(w ** 2)), W)
    np.testing.assert_allclose(g, 2 * before, atol=1e-6)
    np.testing.assert_array_equal(W, before)
    # 非连续视图也要能用
    Wt = W.T
    g2 = numerical_gradient(lambda w: float(np.sum(w ** 3)), Wt)
    np.testing.assert_allclose(g2, 3 * before.T ** 2, rtol=1e-6, atol=1e-6)


def test_numerical_diff_and_descent():
    assert numerical_diff(lambda x: x ** 2, 3.0) == pytest.approx(6.0)
    end, path = gradient_descent(lambda v: float(np.sum(v ** 2)), [3.0, -4.0], lr=0.1, step_num=100)
    assert np.abs(end).max() < 1e-8
    assert path.shape == (101, 2)


def test_mnist_one_hot_helper():
    oh = mnist.to_one_hot([2, 0], 4)
    assert oh.tolist() == [[0, 0, 1, 0], [1, 0, 0, 0]]


@pytest.mark.skipif(not mnist.CACHE.exists(), reason="MNIST 还没下载（第一次运行任意训练脚本会自动下载）")
def test_mnist_shapes():
    (xtr, ttr), (xte, tte) = mnist.load_mnist(normalize=True, flatten=True, one_hot_label=False)
    assert xtr.shape == (60000, 784) and xte.shape == (10000, 784)
    assert ttr.shape == (60000,) and xtr.max() <= 1.0
    (_, ttr2), _ = mnist.load_mnist(one_hot_label=True)
    assert ttr2.shape == (60000, 10)
    (x3, _), _ = mnist.load_mnist(flatten=False, normalize=False)
    assert x3.shape == (60000, 1, 28, 28) and x3.dtype == np.uint8
