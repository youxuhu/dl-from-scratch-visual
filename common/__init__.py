"""各讲共用的函数。

functions.py  激活函数、softmax、损失函数
gradient.py   数值微分、数值梯度、梯度下降
mnist.py      MNIST 下载与读取
arrays.py     第1讲用的数组小工具
"""
from .arrays import describe, matmul_steps

__all__ = ["describe", "matmul_steps"]
