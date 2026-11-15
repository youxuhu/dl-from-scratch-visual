"""MNIST 加载器（独立实现）。

第一次运行时从网上下载 4 个 gz 文件，解析后存成 dataset/mnist.npz，
以后直接读缓存。dataset/ 已写进 .gitignore，不进仓库。
"""
import gzip
import os
import urllib.request
from pathlib import Path

import numpy as np

DATASET_DIR = Path(os.environ.get("MNIST_DIR", Path(__file__).resolve().parent.parent / "dataset"))
CACHE = DATASET_DIR / "mnist.npz"

MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]
FILES = {
    "x_train": "train-images-idx3-ubyte.gz",
    "t_train": "train-labels-idx1-ubyte.gz",
    "x_test": "t10k-images-idx3-ubyte.gz",
    "t_test": "t10k-labels-idx1-ubyte.gz",
}


def _download(name):
    target = DATASET_DIR / name
    if target.exists():
        return target
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    last_err = None
    for base in MIRRORS:
        try:
            print(f"下载 {base}{name} ...")
            tmp = target.with_suffix(".part")
            urllib.request.urlretrieve(base + name, tmp)
            tmp.rename(target)
            return target
        except Exception as e:  # 换下一个镜像
            last_err = e
    raise RuntimeError(f"MNIST 下载失败：{name}，{last_err}")


def _parse(path):
    with gzip.open(path, "rb") as fh:
        raw = fh.read()
    magic = int.from_bytes(raw[:4], "big")
    if magic == 2051:      # 图片：头 16 字节
        n = int.from_bytes(raw[4:8], "big")
        return np.frombuffer(raw, np.uint8, offset=16).reshape(n, 28 * 28)
    if magic == 2049:      # 标签：头 8 字节
        return np.frombuffer(raw, np.uint8, offset=8)
    raise ValueError(f"不认识的文件头 {magic}：{path}")


def _load_raw():
    if CACHE.exists():
        with np.load(CACHE) as z:
            return {k: z[k] for k in FILES}
    arrays = {k: _parse(_download(fname)) for k, fname in FILES.items()}
    np.savez_compressed(CACHE, **arrays)
    return arrays


def to_one_hot(labels, num_classes=10):
    labels = np.asarray(labels).astype(int)
    out = np.zeros((labels.size, num_classes), dtype=np.float32)
    out[np.arange(labels.size), labels] = 1.0
    return out


def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """读取 MNIST，返回 (x_train, t_train), (x_test, t_test)。

    normalize: True 时像素变成 0.0 到 1.0 的 float32，否则保持 0 到 255 的 uint8
    flatten:   True 时每张图是长度 784 的一行，否则形状是 (1, 28, 28)
    one_hot_label: True 时标签是 (N, 10) 的 one hot，否则是 (N,) 的整数标签
    训练集 60000 张，测试集 10000 张。
    """
    d = _load_raw()
    out = {}
    for split in ("train", "test"):
        x = d["x_" + split]
        t = d["t_" + split].astype(int)
        if normalize:
            x = x.astype(np.float32) / 255.0
        if not flatten:
            x = x.reshape(-1, 1, 28, 28)
        if one_hot_label:
            t = to_one_hot(t)
        out[split] = (x, t)
    return out["train"], out["test"]
