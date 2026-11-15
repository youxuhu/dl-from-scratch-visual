"""第0讲：讲法预告。一个 1×2 的行向量乘 2×2 的权重，再加偏置。"""
import numpy as np

X = np.array([1.0, 2.0])
W = np.array([[0.5, -0.2], [0.1, -0.6]])
b = np.array([-0.4, 0.2])
Y = np.dot(X, W) + b

# 课程最后会自己写出来的两个类，这里只露个名字
COMING_SOON = ["TwoLayerNet", "SimpleConvNet"]


def main():
    print("np.dot(X, W) =", np.round(np.dot(X, W), 2))
    print("Y =", np.round(Y, 2))
    print("后面会写出来：", ", ".join(COMING_SOON))


if __name__ == "__main__":
    main()
