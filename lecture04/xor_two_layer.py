"""异或：一层做不到，叠两层就行。实现见 xor_gate.py（对齐鱼书 ch02/xor_gate.py）。"""
from gates import INPUTS
from xor_gate import XOR


def trace():
    """每组输入的中间结果：[(x1, x2, s1, s2, y), ...]"""
    from nand_gate import NAND
    from or_gate import OR
    from and_gate import AND
    rows = []
    for x1, x2 in INPUTS:
        s1, s2 = NAND(x1, x2), OR(x1, x2)
        rows.append((x1, x2, s1, s2, AND(s1, s2)))
    return rows


def main():
    print(" x1 x2 | s1(NAND) s2(OR) | y(XOR)")
    for x1, x2, s1, s2, y in trace():
        print(f"  {x1}  {x2} |    {s1}        {s2}    |   {y}")


if __name__ == "__main__":
    main()
