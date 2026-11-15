"""第3讲：训练日志表和 Iris 样本上的数值。"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lecture03"))

import iris_demo  # noqa: E402
import training_log  # noqa: E402


def test_log_table_basics():
    df = training_log.df
    assert df.shape == (8, 3)
    assert list(df.columns) == ["epoch", "loss", "acc"]
    assert df.head(3)["loss"].tolist() == [2.0734, 1.1835, 0.6912]
    assert str(df["epoch"].dtype) == "int64"


def test_log_stats_and_filter():
    s = training_log.stats
    assert round(s.loc["mean", "loss"], 4) == 0.7427
    assert round(s.loc["mean", "acc"], 4) == 0.8079
    assert s.loc["max", "acc"] == pytest.approx(0.9157)
    assert training_log.good["epoch"].tolist() == [6, 7, 8]
    assert int(training_log.best_row["epoch"]) == 8


def test_log_to_numpy_and_missing():
    assert training_log.arr.shape == (8, 2)
    np.testing.assert_allclose(training_log.arr[0], [2.0734, 0.4120])
    gap = training_log.with_gap()
    assert int(gap["acc"].isnull().sum()) == 1
    assert len(gap.dropna()) == 7
    assert round(float(gap["acc"].fillna(gap["acc"].mean())[3]), 4) == 0.7989


def test_learning_curve_png():
    assert training_log.learning_curve().exists()


def test_iris():
    assert iris_demo.iris.shape == (12, 5)
    assert iris_demo.counts.tolist() == [4, 4, 4]
    np.testing.assert_allclose(iris_demo.group_mean.values, [1.4, 4.525, 5.65])
    assert len(iris_demo.long_petal) == 4
    assert iris_demo.X.shape == (12, 2)
    assert iris_demo.T.shape == (12, 3)
    assert iris_demo.T[0].tolist() == [1, 0, 0] and iris_demo.T[-1].tolist() == [0, 0, 1]
    assert iris_demo.T.sum(axis=1).tolist() == [1] * 12
