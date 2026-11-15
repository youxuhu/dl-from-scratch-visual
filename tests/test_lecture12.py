"""第12讲：BN 数值与梯度确认、Dropout、权值衰减、Trainer、验证数据切分。"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from common.gradient import numerical_gradient
from common.layers import BatchNormalization, Dropout
from common.multi_layer_net import MultiLayerNet
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.trainer import Trainer
from common.util import shuffle_dataset, toy_digits

HERE = Path(__file__).resolve().parent.parent / "lecture12"


def load(name):
    spec = importlib.util.spec_from_file_location(f"l12_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bn = load("batchnorm_steps")
dd = load("dropout_demo")


def test_batchnorm_steps():
    assert bn.mu == 2.0 and bn.var == 5.0
    assert bn.std == pytest.approx(2.2361, abs=1e-4)
    np.testing.assert_allclose(bn.xn, [0.447, -1.342, 1.342, -0.447], atol=1e-3)
    np.testing.assert_allclose(bn.out, [1.394, -2.183, 3.183, -0.394], atol=1e-3)
    np.testing.assert_allclose(bn.layer_out, bn.out)
    assert bn.running_mean_after_one == pytest.approx(0.2)
    assert bn.running_var_after_one == pytest.approx(0.5)


def test_batchnorm_layer_gradient_check():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5, 3))
    gamma, beta = rng.random(3) + 0.5, rng.standard_normal(3)
    layer = BatchNormalization(gamma, beta)
    weight = rng.standard_normal((5, 3))
    f = lambda _: float(np.sum(layer.forward(x, train_flg=True) * weight))  # noqa: E731
    layer.forward(x, train_flg=True)
    dx = layer.backward(weight)
    np.testing.assert_allclose(dx, numerical_gradient(f, x), atol=1e-6)
    np.testing.assert_allclose(layer.dgamma, numerical_gradient(f, gamma), atol=1e-6)
    np.testing.assert_allclose(layer.dbeta, numerical_gradient(f, beta), atol=1e-6)


def test_batchnorm_inference_uses_running_stats():
    layer = BatchNormalization(np.ones(1), np.zeros(1), momentum=0.0)
    layer.forward(np.array([[3.0], [-1.0], [5.0], [1.0]]), train_flg=True)
    out = layer.forward(np.array([[2.0]]), train_flg=False)
    assert out[0, 0] == pytest.approx(0.0)     # 2.0 正好是记住的均值


def test_extend_net_with_batchnorm_gradient_check():
    rng = np.random.default_rng(1)
    net = MultiLayerNetExtend(6, [5, 4], 3, use_batchnorm=True, seed=2)
    x = rng.random((6, 6))
    t = np.array([0, 1, 2, 0, 1, 2])
    numeric = net.numerical_gradient(x, t)
    analytic = net.gradient(x, t)
    assert set(numeric) == set(analytic)
    for k in numeric:
        assert np.mean(np.abs(numeric[k] - analytic[k])) < 1e-5, k


def test_weight_decay_gradient_check():
    rng = np.random.default_rng(3)
    net = MultiLayerNet(5, [4], 3, weight_decay_lambda=0.1, seed=4)
    x = rng.random((4, 5))
    t = np.array([0, 2, 1, 1])
    numeric = net.numerical_gradient(x, t)
    analytic = net.gradient(x, t)
    for k in numeric:
        assert np.mean(np.abs(numeric[k] - analytic[k])) < 1e-5, k
    # 梯度里确实多了 λW
    plain = MultiLayerNet(5, [4], 3, seed=4)
    diff = analytic["W1"] - plain.gradient(x, t)["W1"]
    np.testing.assert_allclose(diff, 0.1 * net.params["W1"], atol=1e-12)


def test_dropout_behaviour():
    assert [int(m.sum()) for m in dd.masks] == [6, 6, 5, 2, 8]
    np.testing.assert_allclose(dd.test_out, dd.hidden * 0.5)
    layer = Dropout(0.3, rng=np.random.default_rng(0))
    x = np.ones((1000, 10))
    y = layer.forward(x, train_flg=True)
    assert 0.25 < 1 - y.mean() < 0.35
    np.testing.assert_array_equal(layer.backward(np.ones_like(x)), layer.mask)


def test_trainer_on_toy_data():
    (x, t), (xt, tt) = toy_digits(400, 100)
    net = MultiLayerNetExtend(784, [50], 10, use_batchnorm=True, seed=0)
    trainer = Trainer(net, x, t, xt, tt, epochs=3, mini_batch_size=50,
                      optimizer="Adam", optimizer_param={"lr": 0.01}, verbose=False, seed=0)
    acc = trainer.train()
    assert len(trainer.train_acc_list) == 3
    assert trainer.train_loss_list[-1] < trainer.train_loss_list[0]
    assert acc > 0.5


def test_shuffle_and_validation_split():
    x = np.arange(10).reshape(10, 1)
    t = np.arange(10)
    xs, ts = shuffle_dataset(x, t, rng=np.random.default_rng(0))
    np.testing.assert_array_equal(xs[:, 0], ts)
    hp = load("hyperparameter_optimization")
    train, val = hp.split_validation(np.zeros((500, 3)), np.zeros(500))
    assert train[0].shape[0] == 400 and val[0].shape[0] == 100
    lr_r, wd_r = hp.narrowed([(0.9, 1e-3, 1e-6), (0.8, 1e-2, 1e-5), (0.7, 1e-4, 1e-7)], top=2)
    assert lr_r == pytest.approx((-3, -2)) and wd_r == pytest.approx((-6, -5))


def test_batch_norm_gradient_check_script():
    """batch_norm_gradient_check.py 的 check：小网络、合成数据，平均绝对误差都应 < 1e-5。"""
    import importlib.util
    from common.util import toy_digits
    path = Path(__file__).resolve().parent.parent / "lecture12" / "batch_norm_gradient_check.py"
    spec = importlib.util.spec_from_file_location("l12_batch_norm_gradient_check", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    (x, t), _ = toy_digits(3, 1)
    diffs = mod.check(x.astype(float), t, hidden_size_list=(6, 5))
    assert set(diffs) >= {"W1", "b1", "gamma1", "beta1", "W3", "b3"}
    assert max(diffs.values()) < 1e-5
