"""神经网络的各种层。

改编自《深度学习入门：基于Python的理论与实现》官方源码 common/layers.py
（MIT 许可，Copyright (c) 2016 Koki Saitoh）。类名、构造参数和属性名与原书一致，
注释改写成中文，并补充了第9讲的 MulLayer、AddLayer（原书在 ch05/layer_naive.py）。

每个层都有 forward 和 backward 两个方法：
forward 算输出并记下反向要用的东西，backward 接收上游传来的梯度 dout，
返回对本层输入的梯度；有参数的层把参数梯度存在 dW、db 等属性里。
"""
import numpy as np

from common.functions import cross_entropy_error, sigmoid, softmax
from common.util import col2im, im2col


class MulLayer:
    """乘法层 z = x * y（第9讲）。反向时把两个输入「翻转」：dx = dout * y，dy = dout * x。"""

    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x, self.y = x, y
        return x * y

    def backward(self, dout):
        return dout * self.y, dout * self.x


class AddLayer:
    """加法层 z = x + y（第9讲）。两个局部导数都是 1，上游梯度原样分给两边。"""

    def forward(self, x, y):
        return x + y

    def backward(self, dout):
        return dout * 1, dout * 1


class Relu:
    """ReLU 层：正向把 x<=0 的位置变 0，反向在同样的位置把梯度挡成 0。"""

    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        y = np.array(x, dtype=float, copy=True)
        y[self.mask] = 0
        return y

    def backward(self, dout):
        grad = np.array(dout, dtype=float, copy=True)
        grad[self.mask] = 0
        return grad


class Sigmoid:
    """Sigmoid 层：反向传播只用到正向的输出 y，梯度是 dout * y * (1 - y)。"""

    def __init__(self):
        self.y = None

    def forward(self, x):
        self.y = sigmoid(x)
        return self.y

    def backward(self, dout):
        return dout * self.y * (1.0 - self.y)


class Affine:
    """全连接层 Y = X·W + b，X 每行一个样本。

    输入如果是图片那样的多维数组，先压成 (N, -1)，反向时再还原形状。
    """

    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.original_x_shape = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.original_x_shape = x.shape
        self.x = x.reshape(x.shape[0], -1) if x.ndim > 2 else x
        return np.dot(self.x, self.W) + self.b

    def backward(self, dout):
        # dL/dX = dL/dY · Wᵀ，dL/dW = Xᵀ · dL/dY，偏置沿批方向求和
        dx = np.dot(dout, self.W.T)
        if self.x.ndim == 1:
            self.dW = np.outer(self.x, dout)
            self.db = np.array(dout, dtype=float)
        else:
            self.dW = np.dot(self.x.T, dout)
            self.db = dout.sum(axis=0)
        return dx.reshape(self.original_x_shape)


class SoftmaxWithLoss:
    """softmax 加交叉熵误差合成一层，反向传播化简成 (y - t) / 批大小。

    t 可以是 one hot，也可以是整数标签。
    """

    def __init__(self):
        self.y = None
        self.t = None
        self.loss = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, t)
        return self.loss

    def backward(self, dout=1):
        y = self.y if self.y.ndim == 2 else self.y[np.newaxis, :]
        n = y.shape[0]
        target = np.asarray(self.t)
        if target.size == y.size:
            diff = y - target.reshape(y.shape)
        else:
            diff = y.copy()
            diff[np.arange(n), target.astype(int).reshape(n)] -= 1
        grad = diff * (dout / n)
        return grad if self.y.ndim == 2 else grad[0]


class Dropout:
    """Dropout：训练时按 dropout_ratio 随机关掉神经元；测试时全开，输出乘 (1 - dropout_ratio)。"""

    def __init__(self, dropout_ratio=0.5, rng=None):
        self.dropout_ratio = dropout_ratio
        self.rng = np.random.default_rng() if rng is None else rng
        self.mask = None

    def forward(self, x, train_flg=True):
        if not train_flg:
            return x * (1.0 - self.dropout_ratio)
        self.mask = self.rng.random(x.shape) > self.dropout_ratio
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask


class BatchNormalization:
    """Batch Normalization：按 mini-batch 标准化，再乘 gamma、加 beta。

    训练时用这一批的均值和方差，同时按 momentum 滑动更新 running_mean、running_var；
    推理时（train_flg=False）改用滑动平均值。输入是 4 维时按 (N, C*H*W) 处理。
    """

    eps = 10e-7

    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.running_mean = running_mean
        self.running_var = running_var
        self.cache = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        shape = x.shape
        flat = x.reshape(shape[0], -1)
        if self.running_mean is None:
            self.running_mean = np.zeros(flat.shape[1])
            self.running_var = np.zeros(flat.shape[1])
        if train_flg:
            mu = flat.mean(axis=0)
            xc = flat - mu
            var = np.mean(xc ** 2, axis=0)
            std = np.sqrt(var + self.eps)
            xn = xc / std
            self.cache = (xc, std, xn)
            m = self.momentum
            self.running_mean = m * self.running_mean + (1 - m) * mu
            self.running_var = m * self.running_var + (1 - m) * var
        else:
            xn = (flat - self.running_mean) / np.sqrt(self.running_var + self.eps)
        out = self.gamma * xn + self.beta
        return out.reshape(shape)

    def backward(self, dout):
        shape = dout.shape
        g = dout.reshape(shape[0], -1)
        xc, std, xn = self.cache
        n = g.shape[0]
        self.dbeta = g.sum(axis=0)
        self.dgamma = np.sum(g * xn, axis=0)
        dxn = g * self.gamma
        # 标准化 xn = (x - mu) / std 的反向，整理成一行公式
        dx = (dxn - dxn.mean(axis=0) - xn * np.mean(dxn * xn, axis=0)) / std
        assert dx.shape == (n, xc.shape[1])
        return dx.reshape(shape)


class Convolution:
    """卷积层：W 形状 (FN, C, FH, FW)，b 形状 (FN,)。

    正向用 im2col 把输入展开成行矩阵，滤波器拉成列，一次矩阵乘法算完所有窗口。
    """

    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        self.x = None
        self.col = None
        self.col_W = None
        self.dW = None
        self.db = None

    def forward(self, x):
        fn, _, fh, fw = self.W.shape
        n, _, h, w = x.shape
        oh = (h + 2 * self.pad - fh) // self.stride + 1
        ow = (w + 2 * self.pad - fw) // self.stride + 1
        cols = im2col(x, fh, fw, self.stride, self.pad)      # (N*OH*OW, C*FH*FW)
        kernel = self.W.reshape(fn, -1).T                     # (C*FH*FW, FN)
        res = np.dot(cols, kernel) + self.b                   # (N*OH*OW, FN)
        self.x, self.col, self.col_W = x, cols, kernel
        return res.reshape(n, oh, ow, fn).transpose(0, 3, 1, 2)

    def backward(self, dout):
        fn, c, fh, fw = self.W.shape
        g = dout.transpose(0, 2, 3, 1).reshape(-1, fn)        # (N*OH*OW, FN)
        self.db = g.sum(axis=0)
        self.dW = np.dot(self.col.T, g).T.reshape(fn, c, fh, fw)
        dcols = np.dot(g, self.col_W.T)
        return col2im(dcols, self.x.shape, fh, fw, self.stride, self.pad)


class Pooling:
    """Max 池化层：每个窗口取最大值，没有参数，通道数不变。

    和原书一样 stride 默认是 1，常用的 2×2、步幅 2 要写成 Pooling(2, 2, stride=2)。
    """

    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        self.x = None
        self.arg_max = None

    def forward(self, x):
        n, c, h, w = x.shape
        oh = (h + 2 * self.pad - self.pool_h) // self.stride + 1
        ow = (w + 2 * self.pad - self.pool_w) // self.stride + 1
        cols = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        # 每行原本是 C 个窗口首尾相接，拆成一行一个窗口
        per_window = cols.reshape(-1, self.pool_h * self.pool_w)
        self.arg_max = per_window.argmax(axis=1)
        self.x = x
        best = per_window.max(axis=1)
        return best.reshape(n, oh, ow, c).transpose(0, 3, 1, 2)

    def backward(self, dout):
        size = self.pool_h * self.pool_w
        g = dout.transpose(0, 2, 3, 1).ravel()
        spread = np.zeros((g.size, size))
        spread[np.arange(g.size), self.arg_max] = g
        n, c, _, _ = self.x.shape
        spread = spread.reshape(-1, c * size)
        return col2im(spread, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
