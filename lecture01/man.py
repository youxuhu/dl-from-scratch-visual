"""一个最简单的类。改编自 oreilly-japan/deep-learning-from-scratch 的 ch01/man.py
（MIT, Copyright (c) 2016 Koki Saitoh）。类名 Man 与书中一致。"""


class Man:
    def __init__(self, name):
        self.name = name
        print("Initialized!")

    def hello(self):
        print("Hello " + self.name + "!")

    def goodbye(self):
        print("Good-bye " + self.name + "!")


def main():
    m = Man("David")
    m.hello()
    m.goodbye()


if __name__ == "__main__":
    main()
