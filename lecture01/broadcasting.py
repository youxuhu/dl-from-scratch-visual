"""第1讲 第三部分：广播。"""
import numpy as np

M = np.array([[4, -1], [0, 2]])
v = np.array([10, 20])
M_plus_v = M + v
M_times_3 = M * 3

col = np.array([[1], [-1]])
M_plus_col = M + col


def main():
    print("M + v =")
    print(M_plus_v)
    print("M * 3 =")
    print(M_times_3)
    print("M + col =")
    print(M_plus_col)


if __name__ == "__main__":
    main()
