"""感知机做逻辑门：与、与非、或。权重和偏置是自己选的一组数。"""
import numpy as np

INPUTS = [(0, 0), (1, 0), (0, 1), (1, 1)]

# 第一版：不用偏置，加权和和阈值 theta 比
def and_with_theta(x1, x2):
    w1, w2, theta = 0.4, 0.4, 0.6
    s = x1 * w1 + x2 * w2
    if s <= theta:
        return 0
    return 1


# 第二版：把 theta 挪到左边，变成偏置 b = -theta
PARAMS = {
    "AND": (np.array([0.4, 0.4]), -0.6),
    "NAND": (np.array([-0.4, -0.4]), 0.6),
    "OR": (np.array([0.4, 0.4]), -0.2),
}


def weighted_sum(gate, x1, x2):
    w, b = PARAMS[gate]
    x = np.array([x1, x2])
    return float(np.sum(w * x) + b)


def perceptron(gate, x1, x2):
    total = weighted_sum(gate, x1, x2)
    if total <= 0:
        return 0
    return 1


def AND(x1, x2):
    return perceptron("AND", x1, x2)


def NAND(x1, x2):
    return perceptron("NAND", x1, x2)


def OR(x1, x2):
    return perceptron("OR", x1, x2)


def truth_table(gate):
    """返回 [(x1, x2, 加权和, 输出), ...]，加权和保留 2 位小数。"""
    return [(x1, x2, round(weighted_sum(gate, x1, x2), 2), perceptron(gate, x1, x2))
            for x1, x2 in INPUTS]


def main():
    print("用阈值的与门:", [and_with_theta(*p) for p in INPUTS])
    for gate in PARAMS:
        w, b = PARAMS[gate]
        print(f"{gate}: w = {w.tolist()}, b = {b}")
        for x1, x2, s, y in truth_table(gate):
            print(f"  ({x1}, {x2})  加权和 = {s:+.2f}  输出 {y}")


if __name__ == "__main__":
    main()
