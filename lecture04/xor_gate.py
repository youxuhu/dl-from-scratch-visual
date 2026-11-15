"""异或门：两层感知机。改编自 oreilly-japan/deep-learning-from-scratch 的 ch02/xor_gate.py
（MIT, Copyright (c) 2016 Koki Saitoh）。XOR = AND(NAND, OR)。"""
from and_gate import AND
from nand_gate import NAND
from or_gate import OR


def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y


if __name__ == "__main__":
    print(" x1 x2 | s1(NAND) s2(OR) | y(XOR)")
    for xs in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        s1 = NAND(*xs)
        s2 = OR(*xs)
        y = XOR(*xs)
        print(f"  {xs[0]}  {xs[1]} |    {s1}        {s2}    |   {y}")
