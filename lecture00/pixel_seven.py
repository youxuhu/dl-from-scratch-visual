"""第0讲：用 28×28 的格子画几个「7」，再看看手写规则为什么靠不住。

只用 NumPy，不弹窗口。运行后在终端里用字符把格子打印出来。
"""
import numpy as np

SIZE = 28


def draw_seven(top_start, top_end, foot_col, bar=False):
    """在 28×28 的全零格子上画一个 7：一条横线加一条斜线。

    top_start, top_end: 第 5 行横线从哪一列画到哪一列
    foot_col: 斜线落到第 23 行时在哪一列
    bar: 是否在中间加一道小横杠（欧洲写法）
    """
    img = np.zeros((SIZE, SIZE))
    img[5, top_start:top_end + 1] = 1.0
    rows = np.arange(5, 24)
    cols = np.round(np.linspace(top_end, foot_col, len(rows))).astype(int)
    img[rows, cols] = 1.0
    if bar:
        img[14, cols[9] - 3:cols[9] + 4] = 1.0
    return img


neat = draw_seven(7, 20, 11)
slanted = draw_seven(9, 21, 6)
short_top = draw_seven(15, 20, 12)
barred = draw_seven(7, 20, 11, bar=True)
# 一个「T」：根本不是 7，但横线很长
tee = np.zeros((SIZE, SIZE))
tee[5, 7:21] = 1.0
tee[5:24, 14] = 1.0

sevens = {"neat": neat, "slanted": slanted, "short_top": short_top, "barred": barred}


def looks_like_seven(img):
    """一条手写规则：第 5 行至少亮 10 个格子，就算 7。"""
    return int(img[5].sum()) >= 10


x = neat.flatten()


def to_text(img):
    return "\n".join("".join("#" if v > 0 else "." for v in row) for row in img)


def main():
    print(to_text(neat))
    print("亮着的像素个数：", int(neat.sum()))
    print("拉直以后 x.shape =", x.shape)
    for name, img in sevens.items():
        print(f"{name}: 第5行亮 {int(img[5].sum())} 格，规则判断 = {looks_like_seven(img)}")
    print(f"tee: 第5行亮 {int(tee[5].sum())} 格，规则判断 = {looks_like_seven(tee)}")


if __name__ == "__main__":
    main()
