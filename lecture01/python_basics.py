"""第1讲 第一部分：够用就好的 Python 语法。"""

# 列表：一排独立的小盒子
scores = [3, 1, 4]
doubled_list = scores * 2

# 字典：用名字找值
cat = {"name": "白猫", "age": 2}

# for 和 if
total = 0
for s in scores:
    if s > 2:
        total += s


# 函数
def square(x):
    return x * x


# 类：把数据和动作打包在一起
class Counter:
    def __init__(self, name):
        self.name = name
        self.count = 0

    def add(self, n):
        self.count += n

    def report(self):
        return f"{self.name} 一共数到 {self.count}"


def main():
    print("scores * 2 =", doubled_list)
    print("cat['name'] =", cat["name"])
    print("大于 2 的分数之和 total =", total)
    print("square(5) =", square(5))
    c = Counter("白猫")
    c.add(3)
    c.add(4)
    print(c.report())


if __name__ == "__main__":
    main()
