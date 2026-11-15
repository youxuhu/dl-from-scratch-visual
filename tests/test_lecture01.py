"""检查第1讲视频里显示的数值和脚本真实算出来的一致。"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import axis_sum
import broadcasting
import indexing
import matmul
import numpy_arrays
import python_basics
from common import describe, matmul_steps

ROOT = Path(__file__).resolve().parent.parent


def test_python_basics():
    assert python_basics.doubled_list == [3, 1, 4, 3, 1, 4]
    assert python_basics.total == 7
    assert python_basics.square(5) == 25
    c = python_basics.Counter("白猫")
    c.add(3)
    c.add(4)
    assert c.report() == "白猫 一共数到 7"


def test_elementwise():
    assert numpy_arrays.y.tolist() == [6, 2, 8]
    assert describe(numpy_arrays.x) == "shape=(3,) dtype=int64"
    assert describe(numpy_arrays.a) == "shape=(3,) dtype=float64"
    np.testing.assert_allclose(numpy_arrays.add, [2.0, -1.5, 6.0])
    np.testing.assert_allclose(numpy_arrays.mul, [0.75, -1.0, 8.0])
    np.testing.assert_allclose(numpy_arrays.div, [3.0, -4.0, 2.0])
    assert numpy_arrays.A.shape == (2, 3)


def test_broadcasting():
    assert broadcasting.M_plus_v.tolist() == [[14, 19], [10, 22]]
    assert broadcasting.M_times_3.tolist() == [[12, -3], [0, 6]]
    assert broadcasting.M_plus_col.tolist() == [[5, 0], [-1, 1]]


def test_axis():
    assert axis_sum.col_sum.tolist() == [3, 3, 4]
    assert axis_sum.row_sum.tolist() == [7, 3]
    assert axis_sum.A.sum() == 10
    assert axis_sum.row_max.tolist() == [5, 3]
    assert axis_sum.row_argmax.tolist() == [2, 1]


def test_matmul():
    assert matmul.C.tolist() == [[5, 1], [4, 2]]
    assert matmul.C.shape == (2, 2)
    steps = [s for _, _, s in matmul_steps(matmul.A, matmul.B)]
    assert steps == [
        "1×3 + 0×2 + 2×1 = 5",
        "1×1 + 0×1 + 2×0 = 1",
        "(-1)×3 + 3×2 + 1×1 = 4",
        "(-1)×1 + 3×1 + 1×0 = 2",
    ]
    assert matmul.shape_error_message() == "shapes (2,3) and (2,2) not aligned: 3 (dim 1) != 2 (dim 0)"


def test_row_vector():
    np.testing.assert_allclose(matmul.Y, [0.6, -0.2, 0.15])
    assert matmul.Y.shape == (3,)
    np.testing.assert_allclose(matmul.Y2, [[0.6, -0.2, 0.15], [1.3, 0.5, -0.4]])
    assert matmul.Y2.shape == (2, 3)


def test_indexing():
    assert indexing.first == 23
    assert indexing.middle.tolist() == [8, 31, 17]
    assert indexing.mask.tolist() == [True, False, True, False, False, True]
    assert indexing.hot.tolist() == [23, 31, 26]
    assert indexing.hot_flags.tolist() == [1, 0, 1, 0, 0, 1]
    assert indexing.hot_days == 3
    assert indexing.f_int.tolist() == [2, -1, 0]
    assert indexing.row1.tolist() == [1, 3, -1]
    assert indexing.elem == -1
    assert indexing.flat.tolist() == [2, 0, 5, 1, 3, -1]
    assert indexing.positive.tolist() == [2, 5, 1, 3]


def test_matmul_steps_rejects_bad_shape():
    with pytest.raises(ValueError):
        matmul_steps(np.ones((2, 3)), np.ones((2, 2)))


@pytest.mark.parametrize("folder", ["lecture00", "lecture01"])
def test_scripts_run(folder):
    for f in sorted((ROOT / folder).glob("*.py")):
        r = subprocess.run([sys.executable, f.name], cwd=f.parent, capture_output=True, text=True,
                           env={"MPLBACKEND": "Agg", "PATH": ""})
        assert r.returncode == 0, f"{f.name}: {r.stderr}"
