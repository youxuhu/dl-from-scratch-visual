"""sigmoid 和阶跃函数画在一起。改编自鱼书 ch03/sig_step_compare.py
（MIT, Copyright (c) 2016 Koki Saitoh）。函数从 common/functions.py 导入。"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common.functions import sigmoid, step_function  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"


def main():
    x = np.arange(-5.0, 5.0, 0.1)
    OUT.mkdir(exist_ok=True)
    plt.plot(x, sigmoid(x), label="sigmoid")
    plt.plot(x, step_function(x), "k--", label="step")
    plt.ylim(-0.1, 1.1)
    plt.legend()
    path = OUT / "sig_step_compare.png"
    plt.savefig(path, dpi=100)
    plt.close()
    print("采样点个数:", len(x))
    print("图已保存:", path)


if __name__ == "__main__":
    main()
