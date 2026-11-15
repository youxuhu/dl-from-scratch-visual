"""第2讲：检查视频里出现的数值和图片输出。"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lecture02"))

import array_image  # noqa: E402
import loss_curve  # noqa: E402
import sin_cos  # noqa: E402
import subplots  # noqa: E402


def test_sample_points():
    assert sin_cos.n_points == 35
    np.testing.assert_allclose(sin_cos.x[:4], [0.0, 0.2, 0.4, 0.6])
    assert (sin_cos.n_pos, sin_cos.n_neg) == (18, 16)
    assert round(float(sin_cos.y_sin[8]), 4) == 0.9996


def test_sin_cos_png():
    assert sin_cos.draw().exists()


def test_grid_image():
    assert array_image.brightest == (2, 2)
    assert array_image.n_bright == 9
    img = array_image.roundtrip_png()
    assert img.shape == (5, 5, 4)
    # imshow 灰度：值 1.0 的格子最亮，值 0.0 的最暗
    assert img[2, 2, 0] == pytest.approx(1.0)
    assert img[0, 0, 0] == pytest.approx(0.0)


def test_loss_curve_numbers():
    np.testing.assert_allclose(np.round(loss_curve.loss[:3], 3), [2.237, 2.317, 2.145])
    assert round(float(loss_curve.smooth[0]), 3) == 2.137
    assert round(float(loss_curve.smooth[-1]), 3) == 0.151
    assert len(loss_curve.smooth) == loss_curve.STEPS - loss_curve.WINDOW + 1


def test_subplots():
    path, n_axes = subplots.draw()
    assert n_axes == 3 and path.exists()
