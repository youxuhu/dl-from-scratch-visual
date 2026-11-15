"""异或：一层做不到，叠两层就行。第一层并排放与非门和或门，第二层放与门。"""
from gates import AND, INPUTS, NAND, OR


def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    return AND(s1, s2)


def trace():
    """每组输入的中间结果：[(x1, x2, s1, s2, y), ...]"""
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
