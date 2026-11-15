"""第4讲：感知机逻辑门、两层异或、线性可分搜索。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lecture04"))

import decision_lines  # noqa: E402
import gates  # noqa: E402
import separability  # noqa: E402
import xor_two_layer  # noqa: E402
import and_gate, nand_gate, or_gate, xor_gate  # noqa: E402


def test_and_theta():
    assert [gates.and_with_theta(*p) for p in gates.INPUTS] == [0, 0, 0, 1]


def test_truth_tables_and_sums():
    assert gates.truth_table("AND") == [(0, 0, -0.7, 0), (1, 0, -0.2, 0), (0, 1, -0.2, 0), (1, 1, 0.3, 1)]
    assert gates.truth_table("NAND") == [(0, 0, 0.7, 1), (1, 0, 0.2, 1), (0, 1, 0.2, 1), (1, 1, -0.3, 0)]
    assert gates.truth_table("OR") == [(0, 0, -0.2, 0), (1, 0, 0.3, 1), (0, 1, 0.3, 1), (1, 1, 0.8, 1)]


def test_xor_trace():
    assert xor_two_layer.trace() == [(0, 0, 1, 0, 0), (1, 0, 1, 1, 1), (0, 1, 1, 1, 1), (1, 1, 0, 1, 0)]
    assert [xor_two_layer.XOR(*p) for p in gates.INPUTS] == [0, 1, 1, 0]


def test_separability_counts():
    assert len(separability.GRID) ** 3 == 9261
    counts = {k: separability.count_solutions(v) for k, v in separability.TARGETS.items()}
    assert counts == {"AND": 224, "NAND": 159, "OR": 385, "XOR": 0}


def test_decision_lines():
    _, ys = decision_lines.line_points("AND")
    np.testing.assert_allclose(ys, [1.4, -0.1])
    _, ys = decision_lines.line_points("OR")
    np.testing.assert_allclose(ys, [0.4, -1.1])
    assert decision_lines.draw().exists()


def test_book_gate_files():
    """ch02 同名脚本的函数和 gates 里的一致。"""
    for x1, x2 in gates.INPUTS:
        assert and_gate.AND(x1, x2) == gates.AND(x1, x2)
        assert nand_gate.NAND(x1, x2) == gates.NAND(x1, x2)
        assert or_gate.OR(x1, x2) == gates.OR(x1, x2)
        assert xor_gate.XOR(x1, x2) == xor_two_layer.XOR(x1, x2)
