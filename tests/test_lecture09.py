"""第9讲：计算图的正向数值、反向梯度，以及数值梯度和反向传播的梯度确认。"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from common.gradient import numerical_gradient

HERE = Path(__file__).resolve().parent.parent / "lecture09"


def load(name):
    """按文件路径加载本讲脚本，模块名带讲次前缀，避免和别的讲同名文件冲突。"""
    spec = importlib.util.spec_from_file_location(f"l09_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


naive = load("layer_naive")
apple = load("buy_apple")
fruit = load("buy_apple_orange")
chain = load("chain_rule")


def test_mul_add_layers():
    m = naive.MulLayer()
    assert m.forward(3, 5) == 15
    assert m.backward(1) == (5, 3)
    a = naive.AddLayer()
    assert a.forward(3, 5) == 8
    assert a.backward(2) == (2, 2)


def test_buy_apple():
    assert apple.apple_price == 200
    assert apple.price == pytest.approx(220)
    assert apple.dtax == pytest.approx(200)
    assert apple.dapple_num == pytest.approx(110)
    assert apple.dapple == pytest.approx(2.2)
    total, slope = apple.nudge_check(0.01)
    assert total == pytest.approx(220.022)
    assert slope == pytest.approx(2.2)


def test_buy_apple_orange():
    assert fruit.forward_values["fruit_total"] == 650
    assert fruit.price == pytest.approx(715)
    expect = {"dApple": 2.2, "dApple_num": 110, "dOrange": 3.3, "dOrange_num": 165, "dTax": 650}
    for k, v in expect.items():
        assert fruit.backward_values[k] == pytest.approx(v), k


def test_chain_rule():
    assert (chain.t, chain.z, chain.dz_dx) == (4.0, 16.0, 8.0)
    assert chain.numeric_dz_dx() == pytest.approx(8.0, abs=1e-6)


def test_gradient_check_shopping_graph():
    """梯度确认：数值梯度 vs 反向传播。输入顺序 [苹果单价, 苹果个数, 橘子单价, 橘子个数, 税率]。"""
    def total(v):
        a = naive.MulLayer().forward(v[0], v[1])
        o = naive.MulLayer().forward(v[2], v[3])
        return naive.MulLayer().forward(naive.AddLayer().forward(a, o), v[4])

    v = np.array([100.0, 2.0, 150.0, 3.0, 1.1])
    numeric = numerical_gradient(lambda _: total(v), v)
    analytic = np.array([fruit.dapple, fruit.dapple_num, fruit.dorange, fruit.dorange_num, fruit.dtax])
    np.testing.assert_allclose(numeric, analytic, rtol=1e-6)
