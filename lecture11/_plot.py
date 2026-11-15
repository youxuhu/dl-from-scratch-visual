"""本讲脚本共用：有 matplotlib 就把图存到 lecture11/output/，没有就跳过，不影响数值输出。"""
from pathlib import Path

OUT = Path(__file__).resolve().parent / "output"


def get_pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("提示：没装 matplotlib，跳过画图。")
        return None
    OUT.mkdir(exist_ok=True)
    return plt
