"""第10讲：各层的数值、逐层梯度确认、TwoLayerNet 整体梯度确认。"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from common.gradient import numerical_gradient
from common.layers import Affine, Relu, Sigmoid, SoftmaxWithLoss

HERE = Path(__file__).resolve().parent.parent / "lecture10"
sys.path.insert(0, str(HERE))


def load(name):
    spec = importlib.util.spec_from_file_location(f"l10_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


act = load("activation_layers")
aff = load("affine_layer")
smx = load("softmax_loss_grad")
tln = load("two_layer_net_bp")


def test_relu_values():
    assert act.relu_mask.tolist() == [[False, True], [True, False]]
    np.testing.assert_allclose(act.relu_out, [[1, 0], [0, 3]])
    np.testing.assert_allclose(act.relu_dx, [[0.4, 0], [0, 0.6]])


def test_sigmoid_nodes():
    assert act.sig_y == pytest.approx(0.731059, abs=1e-6)
    assert act.sig_backward["d(1+exp)"] == pytest.approx(-0.534447, abs=1e-6)
    assert act.sig_backward["dx"] == pytest.approx(0.196612, abs=1e-6)
    assert act.sig_simple == pytest.approx(act.sig_backward["dx"])
    assert act.sig_layer_dx == pytest.approx(act.sig_simple)


def test_affine_values():
    np.testing.assert_allclose(aff.Y, [0.6, -0.2, 0.15])
    np.testing.assert_allclose(aff.dX, [0.15, 0.11])
    np.testing.assert_allclose(aff.dW_one, [[0.3, -0.2, 0.1], [0.15, -0.1, 0.05]])
    np.testing.assert_allclose(aff.db2, [0.2, 0.2, 0.3])
    np.testing.assert_allclose(aff.dX2, [[0.15, 0.11], [-0.16, -0.04]])
    assert aff.dX.shape == aff.X.shape and aff.dW2.shape == aff.W.shape


def test_softmax_loss_values():
    assert smx.bad_loss == pytest.approx(2.9417, abs=1e-4)
    assert smx.good_loss == pytest.approx(0.0166, abs=1e-4)
    assert smx.bad_grad[3] == pytest.approx(-0.947, abs=1e-3)
    assert smx.good_grad[3] == pytest.approx(-0.016, abs=1e-3)
    assert abs(smx.bad_grad.sum()) < 1e-12        # y 和 t 各自加起来都是 1
    assert np.all(np.delete(smx.bad_grad, 3) > 0)


def _scalar_check(layer, x, rng):
    """L = Σ(输出 · 随机权重)，用它对输入做梯度确认。"""
    weight = rng.standard_normal(layer.forward(x.copy()).shape)
    analytic = layer.backward(weight)
    numeric = numerical_gradient(lambda _: float(np.sum(layer.forward(x) * weight)), x)
    np.testing.assert_allclose(analytic, numeric, atol=1e-6)


def test_layer_gradient_checks():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 4)) + 0.05       # 避开 ReLU 在 0 处的折点
    _scalar_check(Relu(), x.copy(), rng)
    _scalar_check(Sigmoid(), x.copy(), rng)
    W, b = rng.standard_normal((4, 5)), rng.standard_normal(5)
    layer = Affine(W, b)
    _scalar_check(layer, x.copy(), rng)
    weight = rng.standard_normal((3, 5))
    layer.forward(x)
    layer.backward(weight)
    numeric_W = numerical_gradient(lambda _: float(np.sum(layer.forward(x) * weight)), W)
    np.testing.assert_allclose(layer.dW, numeric_W, atol=1e-6)
    np.testing.assert_allclose(layer.db, weight.sum(axis=0))
    swl = SoftmaxWithLoss()
    t = np.array([0, 3, 1])
    swl.forward(x, t)
    numeric = numerical_gradient(lambda _: swl.forward(x, t), x)
    # cross_entropy_error 里有 eps=1e-7，数值梯度会带一点点偏差，容差放宽到 1e-4
    np.testing.assert_allclose(swl.backward(1), numeric, atol=1e-4)


def test_two_layer_net_gradient_check():
    rng = np.random.default_rng(1)
    net = tln.TwoLayerNet(12, 6, 4, weight_init_std=0.5, seed=2)
    x = rng.random((5, 12))
    t = np.eye(4)[[0, 1, 2, 3, 1]]
    numeric = net.numerical_gradient(x, t)
    analytic = net.gradient(x, t)
    for k in numeric:
        assert np.mean(np.abs(numeric[k] - analytic[k])) < 1e-5, k


def test_gradient_check_script_runs_on_toy_data():
    gc = load("gradient_check")
    from common.util import toy_digits
    (x, t), _ = toy_digits(3, 1)
    diffs, t_num, t_bp = gc.check(x.astype(float), t, seed=0)
    assert all(v < 1e-6 for v in diffs.values())
    assert t_num > t_bp
