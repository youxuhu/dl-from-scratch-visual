"""检查第0讲视频里的数值。"""
import numpy as np

import pixel_seven
import preview


def test_preview():
    np.testing.assert_allclose(np.dot(preview.X, preview.W), [0.7, -1.4])
    np.testing.assert_allclose(preview.Y, [0.3, -1.2])


def test_pixel_seven():
    assert pixel_seven.neat.shape == (28, 28)
    assert int(pixel_seven.neat.sum()) == 32
    assert pixel_seven.x.shape == (784,)
    rows = {k: int(v[5].sum()) for k, v in pixel_seven.sevens.items()}
    assert rows == {"neat": 14, "slanted": 13, "short_top": 6, "barred": 14}
    assert [pixel_seven.looks_like_seven(v) for v in pixel_seven.sevens.values()] == [True, True, False, True]
    assert int(pixel_seven.tee[5].sum()) == 14
    assert pixel_seven.looks_like_seven(pixel_seven.tee) is True
