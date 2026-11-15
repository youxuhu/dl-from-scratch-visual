"""第4讲：感知机逻辑门、两层异或、线性可分搜索。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lecture04"))

import decision_lines  # noqa: E402
import gates  # noqa: E402
import separability  # noqa: E402
import xor_two_layer  # noqa: E402


def test_and_theta():
    assert [gates.and_with_theta(*p) for p in gates.INPUTS] == [0, 0, 0, 1]


def test_truth_tables_and_sums():
    assert gates.truth_table("AND") == [(0, 0, -0.6, 0), (1, 0, -0.2, 0), (0, 1, -0.2, 0), (1, 1, 0.2, 1)]
    assert gates.truth_table("NAND") == [(0, 0, 0.6, 1), (1, 0, 0.2, 1), (0, 1, 0.2, 1), (1, 1, -0.2, 0)]
    assert gates.truth_table("OR") == [(0, 0, -0.2, 0), (1, 0, 0.2, 1), (0, 1, 0.2, 1), (1, 1, 0.6, 1)]


def test_xor_trace():
    assert xor_two_layer.trace() == [(0, 0, 1, 0, 0), (1, 0, 1, 1, 1), (0, 1, 1, 1, 1), (1, 1, 0, 1, 0)]
    assert [xor_two_layer.XOR(*p) for p in gates.INPUTS] == [0, 1, 1, 0]


def test_separability_counts():
    assert len(separability.GRID) ** 3 == 9261
    counts = {k: separability.count_solutions(v) for k, v in separability.TARGETS.items()}
    assert counts == {"AND": 224, "NAND": 159, "OR": 385, "XOR": 0}


def test_decision_lines():
    _, ys = decision_lines.line_points("AND")
    np.testing.assert_allclose(ys, [1.5, 0.0])
    _, ys = decision_lines.line_points("OR")
    np.testing.assert_allclose(ys, [0.5, -1.0])
    assert decision_lines.draw().exists()
