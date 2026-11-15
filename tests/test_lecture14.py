"""第14讲：im2col/col2im、卷积层内部形状、SimpleConvNet 梯度确认与训练。"""
import importlib.util
from pathlib import Path

import numpy as np

from common.simple_convnet import SimpleConvNet
from common.trainer import Trainer
from common.util import col2im, im2col, toy_digits

HERE = Path(__file__).resolve().parent.parent / "lecture14"


def load(name):
    spec = importlib.util.spec_from_file_location(f"l14_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


demo = load("im2col_demo")
cpl = load("conv_pool_layers")
nets = load("lenet_alexnet")


def test_im2col_numbers():
    assert demo.rows.shape == (4, 9)
    assert demo.rows[0].tolist() == [2, 1, 0, 1, 3, 2, 0, 2, 1]
    assert demo.result.tolist() == [10, 8, 2, 11]
    assert demo.appear_count.tolist() == [[1, 2, 2, 1], [2, 4, 4, 2], [2, 4, 4, 2], [1, 2, 2, 1]]
    assert demo.memory_ratio == 2.25
    assert demo.col1.shape == (9, 75) and demo.col2.shape == (90, 75)


def test_col2im_is_adjoint_of_im2col():
    rng = np.random.default_rng(0)
    for stride, pad in [(1, 0), (2, 1), (1, 2)]:
        x = rng.standard_normal((2, 3, 7, 7))
        cols = im2col(x, 3, 3, stride, pad)
        c = rng.standard_normal(cols.shape)
        back = col2im(c, x.shape, 3, 3, stride, pad)
        assert np.isclose(np.sum(cols * c), np.sum(x * back))


def test_layer_internal_shapes():
    assert cpl.shapes["col"] == (18, 75)
    assert cpl.shapes["col_W"] == (75, 4)
    assert cpl.shapes["y(N, FN, OH, OW)"] == (2, 4, 3, 3)
    assert cpl.moved.shape == (2, 4, 3, 3)
    assert cpl.moved[1, 2, 0, 1] == cpl.demo[1, 0, 1, 2]
    assert cpl.row_max.tolist() == [7, 5, 6, 3, 8, 6, 9, 4]
    assert cpl.pooled[0, 0].tolist() == [[7, 6], [8, 9]]
    assert cpl.pooled[0, 1].tolist() == [[5, 3], [6, 4]]


def test_simple_convnet_shapes_and_gradient_check():
    assert SimpleConvNet().shapes == [(1, 28, 28), (30, 24, 24), (30, 12, 12), (100,), (10,)]
    rng = np.random.default_rng(1)
    net = SimpleConvNet((1, 6, 6), {"filter_num": 2, "filter_size": 3}, hidden_size=4,
                        output_size=3, weight_init_std=0.5, seed=2)
    x = rng.random((2, 1, 6, 6))
    t = np.array([0, 2])
    numeric = net.numerical_gradient(x, t)
    analytic = net.gradient(x, t)
    for k in numeric:
        assert np.mean(np.abs(numeric[k] - analytic[k])) < 1e-5, k


def test_save_load_roundtrip(tmp_path):
    a = SimpleConvNet(seed=0)
    a.save_params(tmp_path / "p.pkl")
    b = SimpleConvNet(seed=1)
    b.load_params(tmp_path / "p.pkl")
    x = np.random.default_rng(0).random((3, 1, 28, 28))
    np.testing.assert_allclose(a.predict(x), b.predict(x))


def test_training_on_toy_digits():
    (x, t), (xt, tt) = toy_digits(200, 50, flatten=False)
    net = SimpleConvNet(conv_param={"filter_num": 8}, hidden_size=30, seed=0)
    trainer = Trainer(net, x, t, xt, tt, epochs=4, mini_batch_size=50, optimizer="Adam",
                      optimizer_param={"lr": 0.005}, verbose=False, seed=0)
    trainer.train()
    assert np.mean(trainer.train_loss_list[-3:]) < trainer.train_loss_list[0]


def test_lenet_alexnet():
    assert nets.lenet_params == 2572
    assert nets.lenet_trace[-1] == ("subsample", 16, 5)
    assert nets.alex_trace[1] == ("conv", 96, 55)
    assert nets.alex_trace[-1] == ("pool", 256, 6)


def test_edge_filter_prefers_vertical_strokes():
    vf = load("visualize_filter")
    digit = np.zeros((28, 28))
    digit[4:24, 13:15] = 1.0            # 一根竖线
    v, h = vf.edge_response(digit)
    assert v.sum() > 3 * h.sum()
    assert vf.roughness(np.random.default_rng(0).standard_normal((5, 1, 5, 5))).mean() > 0.3


def test_gradient_check_script_book_setting():
    """gradient_check.py：原书的 1×10×10 小网络，各参数平均绝对误差 < 1e-6。"""
    gc = load("gradient_check")
    diffs = gc.check(seed=0)
    assert set(diffs) == {"W1", "b1", "W2", "b2", "W3", "b3"}
    assert max(diffs.values()) < 1e-6
