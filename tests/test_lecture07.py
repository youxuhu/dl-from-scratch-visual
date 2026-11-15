"""第7讲：损失函数、mini-batch、精度和损失的对比、背答案模型。"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lecture07"))

import cross_entropy_curve  # noqa: E402
import losses  # noqa: E402
from common import mnist  # noqa: E402

needs_mnist = pytest.mark.skipif(not mnist.CACHE.exists(), reason="MNIST 还没下载")


def test_mse_steps():
    assert losses.y_good.sum() == pytest.approx(1.0) and losses.y_bad.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(np.round(losses.diff_good, 2),
                               [0.02, 0.05, 0.03, -0.3, 0.04, 0.06, 0.02, 0.03, 0.03, 0.02])
    assert round(float(losses.sq_good.sum()), 4) == 0.1016
    assert round(losses.mse_good, 4) == 0.0508
    assert round(losses.mse_bad, 4) == 0.6088


def test_cross_entropy():
    assert round(losses.ce_good, 4) == 0.3567
    assert round(losses.ce_bad, 4) == 2.3026
    from common.functions import cross_entropy_error
    assert cross_entropy_error(losses.y_good, 3) == pytest.approx(losses.ce_good)


def test_curve_points():
    np.testing.assert_allclose(np.round(cross_entropy_curve.penalty, 3), [0.105, 0.511, 1.204, 2.303])
    assert cross_entropy_curve.draw().exists()


@needs_mnist
def test_minibatch():
    import minibatch
    assert minibatch.batch_mask[:5].tolist() == [40984, 707, 58170, 5489, 30465]
    assert minibatch.x_batch.shape == (100, 784) and minibatch.t_batch.shape == (100, 10)
    np.testing.assert_allclose(np.round(minibatch.picked_probs()[:3], 4), [0.9541, 0.9999, 0.9957])
    assert round(minibatch.loss_onehot, 4) == 0.0553
    assert minibatch.loss_labels == pytest.approx(minibatch.loss_onehot)


@needs_mnist
def test_accuracy_is_steps_loss_is_smooth():
    import accuracy_vs_loss as avl
    assert round(avl.w0, 4) == 0.6326
    np.testing.assert_allclose(np.unique(avl.accs), [0.98, 0.985, 0.99, 0.995, 1.0])
    assert len(np.unique(np.round(avl.losses, 6))) == 120
    assert round(float(avl.losses.min()), 4) == 0.0136 and round(float(avl.losses.max()), 4) == 0.0479
    d_acc, d_loss = avl.slopes_at_zero()
    assert d_acc == 0.0 and round(d_loss, 5) == -0.00115


@needs_mnist
def test_memorizer():
    import memorizer
    assert memorizer.train_acc == 1.0
    assert memorizer.test_acc == pytest.approx(0.089)
