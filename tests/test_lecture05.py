"""第5讲：激活函数、线性叠加、一层的矩阵乘法。"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lecture05"))

import activations  # noqa: E402
import layer_dot  # noqa: E402
import plot_activations  # noqa: E402
import why_nonlinear  # noqa: E402


def test_step_two_versions():
    assert activations.step_scalar(2.5) == 1 and activations.step_scalar(-0.1) == 0
    assert activations.step_scalar_on_array() == "ValueError"
    assert activations.mask.tolist() == [False, False, True]
    assert activations.y_step.tolist() == [0, 0, 1]


def test_probe_table():
    t = activations.table
    assert t["step"].tolist() == [0, 0, 0, 1, 1]
    np.testing.assert_allclose(t["sigmoid"], [0.119, 0.378, 0.5, 0.731, 0.953])
    np.testing.assert_allclose(t["relu"], [0, 0, 0, 1, 3])


def test_plot():
    assert len(plot_activations.xs) == 121
    assert round(float(plot_activations.curves["sigmoid"][0]), 4) == 0.0025
    assert plot_activations.draw().exists()


def test_linear_collapses():
    assert why_nonlinear.three_layers == why_nonlinear.one_layer == 12.0
    np.testing.assert_allclose(why_nonlinear.stacked, [4.0, 1.5])
    np.testing.assert_allclose(why_nonlinear.merged, why_nonlinear.stacked)
    np.testing.assert_allclose(why_nonlinear.W_all, [[7.0, 2.5], [3.0, 1.0]])
    assert round(float(why_nonlinear.s_of_2x), 3) == 0.881
    assert round(float(why_nonlinear.two_s_of_x), 3) == 1.462


def test_layer_dot():
    assert layer_dot.Y.tolist() == [4, 7, -1]
    assert layer_dot.Y.shape == (3,)
    assert layer_dot.columns() == ["1×2 + 2×1 = 4", "1×(-1) + 2×4 = 7", "1×3 + 2×(-2) = -1"]
    assert np.ndim(layer_dot.X) == 1 and layer_dot.W.shape == (2, 3)
