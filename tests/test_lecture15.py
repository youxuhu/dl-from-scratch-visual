"""第15讲：感受野与参数个数、DeepConvNet 结构与梯度确认、半精度、数据扩充。"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

from common.deep_convnet import DeepConvNet
from common.layers import Dropout

HERE = Path(__file__).resolve().parent.parent / "lecture15"


def load(name):
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))      # half_float_network 等要导入同目录的 _trained
    spec = importlib.util.spec_from_file_location(f"l15_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rf = load("receptive_field")
st = load("deep_convnet")


def test_receptive_field():
    assert (rf.params_5x5, rf.params_two_3x3) == (25, 18)
    assert rf.covered == 25
    assert rf.grad_in[0, 0].astype(int).tolist()[2] == [3, 6, 9, 6, 3]


def test_deep_convnet_structure():
    assert st.filter_nums == [16, 16, 32, 32, 64, 64]
    assert st.pads == [1, 1, 1, 2, 1, 1]
    assert st.net.shapes[-3:] == [(64, 4, 4), (50,), (10,)]
    assert st.layer_names.count("Convolution") == 6
    assert st.layer_names.count("Pooling") == 3
    assert st.layer_names[-6:] == ["Affine", "Relu", "Dropout", "Affine", "Dropout", "SoftmaxWithLoss"]
    assert st.param_count == 123552


def test_deep_convnet_gradient_check():
    """小号 DeepConvNet（16×16 输入、每层 2 个通道）的数值梯度 vs 反向传播。

    Dropout 比例设成 0 让正向确定；偏置设成小正数，避开 ReLU 在 0 处的折点。
    填充出来的边框区域数值完全相同，池化时可能并列最大，扰动 h 会让最大值换位置，
    所以固定一个没有这种并列的随机种子。阈值：每个参数的 最大绝对误差 / 最大梯度绝对值 < 1e-4。
    """
    net = DeepConvNet((1, 16, 16), conv_plan=[(2, 1)] * 3 + [(2, 2)] + [(2, 1)] * 2,
                      hidden_size=4, output_size=3, seed=1)
    rng = np.random.default_rng(1)
    for k, v in net.params.items():
        if k.startswith("b"):
            v[...] = rng.uniform(0.05, 0.2, v.shape)
    for layer in net.layers:
        if isinstance(layer, Dropout):
            layer.dropout_ratio = 0.0
    x = np.random.default_rng(1).random((2, 1, 16, 16))
    t = np.array([0, 2])
    analytic = net.gradient(x, t)
    numeric = net.numerical_gradient(x, t)
    for k in analytic:
        rel = np.abs(analytic[k] - numeric[k]).max() / (np.abs(analytic[k]).max() + 1e-12)
        assert rel < 1e-4, (k, rel)


def test_half_float_keeps_params_and_matches_on_toy():
    hf = load("half_float_network")
    from common.util import toy_digits
    (x, t), _ = toy_digits(60, 1, flatten=False)
    net = DeepConvNet(seed=0)
    before = {k: v.copy() for k, v in net.params.items()}
    a64 = hf.accuracy_with_dtype(net, x, t, np.float64)
    a16 = hf.accuracy_with_dtype(net, x, t, np.float16)
    assert abs(a64 - a16) <= 0.05
    for k in before:
        np.testing.assert_array_equal(net.params[k], before[k])


def test_data_augmentation():
    da = load("data_augmentation")
    img = np.zeros((28, 28))
    img[10:18, 12:14] = 1.0
    out = da.augment(img)
    assert all(v.shape == (28, 28) for v in out.values())
    np.testing.assert_array_equal(out["右移3"][:, 15:17], img[:, 12:14])
    assert out["右移3"].sum() == img.sum()
    np.testing.assert_array_equal(da.rotate(img, 0), img)
    assert np.abs(da.rotate(img, 90).sum() - img.sum()) <= 2


def test_misclassified_finder():
    mc = load("misclassified_mnist")

    class Fixed:
        def predict(self, x, train_flg=False):
            out = np.zeros((len(x), 10))
            out[:, 3] = 1.0
            return out

    preds, wrong = mc.find_mistakes(Fixed(), np.zeros((5, 1, 28, 28)), np.array([3, 1, 3, 4, 3]))
    assert preds.tolist() == [3] * 5
    assert wrong.tolist() == [1, 3]
