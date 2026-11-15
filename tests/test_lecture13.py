"""第13讲：卷积、填充、输出大小、3 维卷积、池化的数值，以及卷积层、池化层的梯度确认。"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from common.gradient import numerical_gradient
from common.layers import Convolution, Pooling
from common.util import conv_output_size

HERE = Path(__file__).resolve().parent.parent / "lecture13"


def load(name):
    spec = importlib.util.spec_from_file_location(f"l13_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cn = load("conv_naive")
osz = load("output_size")
c3 = load("conv3d_naive")
pn = load("pooling_naive")


def test_conv_numbers():
    assert cn.first_products == [2, 0, 0, 2, 3, 0, 0, 2, 1]
    assert cn.out.tolist() == [[10, 8], [2, 11]]
    assert cn.out_bias.tolist() == [[13, 11], [5, 14]]
    assert cn.out_pad1.tolist() == [[6, 10, 4, 3], [2, 10, 8, 5], [0, 2, 11, 7], [1, 5, 2, 5]]
    assert cn.cols.shape == (4, 9)
    assert cn.by_matmul.tolist() == [10, 8, 2, 11]
    np.testing.assert_allclose(cn.by_layer, cn.out_bias)


def test_output_size():
    got = list(osz.sizes.values())
    assert got == [2, 4, 3, 24, 28]
    with pytest.raises(ValueError):
        conv_output_size(4, 3, 2, 1)
    assert osz.below - osz.above == 28


def test_conv3d():
    assert [p.tolist() for p in c3.planes] == [[[10, 8], [2, 11]], [[0, 0], [0, 0]], [[4, 3], [3, 4]]]
    assert c3.summed.tolist() == [[14, 11], [5, 15]]
    assert c3.stacked.shape == (2, 2, 2)
    np.testing.assert_allclose(c3.by_layer, c3.stacked)
    assert c3.batch_out_shape == (10, 30, 24, 24)


def test_pooling():
    assert pn.pooled.tolist() == [[7, 6], [8, 9]]
    assert pn.pooled_shifted.tolist() == [[7, 6], [8, 3]]
    assert pn.unchanged == 3
    np.testing.assert_allclose(pn.by_layer, pn.pooled)


@pytest.mark.parametrize("stride,pad", [(1, 0), (1, 1), (2, 1)])
def test_convolution_gradient_check(stride, pad):
    rng = np.random.default_rng(stride * 10 + pad)
    x = rng.standard_normal((2, 2, 5, 5))
    W = rng.standard_normal((3, 2, 3, 3))
    b = rng.standard_normal(3)
    layer = Convolution(W, b, stride, pad)
    weight = rng.standard_normal(layer.forward(x).shape)
    f = lambda _: float(np.sum(layer.forward(x) * weight))  # noqa: E731
    layer.forward(x)
    dx = layer.backward(weight)
    np.testing.assert_allclose(dx, numerical_gradient(f, x), atol=1e-6)
    np.testing.assert_allclose(layer.dW, numerical_gradient(f, W), atol=1e-6)
    np.testing.assert_allclose(layer.db, numerical_gradient(f, b), atol=1e-6)


def test_pooling_gradient_check():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((2, 3, 4, 4))     # 随机数基本不会有并列最大值
    layer = Pooling(2, 2, stride=2)
    weight = rng.standard_normal((2, 3, 2, 2))
    f = lambda _: float(np.sum(layer.forward(x) * weight))  # noqa: E731
    layer.forward(x)
    np.testing.assert_allclose(layer.backward(weight), numerical_gradient(f, x), atol=1e-6)
