# 第1讲 Python 与 NumPy 基础

## 这讲讲什么
够用就好的 Python 语法（列表、字典、for、if、函数、类），然后是 NumPy：数组、shape 和 dtype、逐元素运算、广播、axis、`np.dot` 矩阵乘法（行向量写法 `np.dot(X, W) + b`）、索引、布尔掩码、`astype(int)`、flatten。

## 文件
| 文件 | 内容 |
|---|---|
| python_basics.py | 列表 `* 2` 是拼接、字典、for/if、函数、自己写的 Counter 类 |
| numpy_arrays.py | np.array、`x * 2`、shape、dtype、逐元素加减乘除 |
| broadcasting.py | 矩阵加向量、矩阵乘标量、矩阵加列向量 |
| axis_sum.py | sum/max/argmax 沿 axis=0 和 axis=1 |
| matmul.py | np.dot 逐格展开、形状不匹配的报错、行向量写法、一次喂两个样本 |
| indexing.py | 索引、切片、布尔掩码、astype(int) 截断、flatten |

common/arrays.py 里的 `describe`、`matmul_steps` 是本讲用到的共用小工具。第1讲台词（L01.json）还没写，写完后在这里补上段落对应表。

## 怎么运行
```bash
cd lecture01
python matmul.py   # 其余脚本同理
```

## 视频里出现的数值（都由脚本算出）
- `[3, 1, 4] * 2` = `[3, 1, 4, 3, 1, 4]`；`np.array([3, 1, 4]) * 2` = `[6 2 8]`，dtype int64。
- a = [1.5, -2.0, 4.0]，b = [0.5, 0.5, 2.0]：a+b = [2.0, -1.5, 6.0]，a*b = [0.75, -1.0, 8.0]，a/b = [3.0, -4.0, 2.0]。
- M = [[4, -1], [0, 2]]：M + [10, 20] = [[14, 19], [10, 22]]；M * 3 = [[12, -3], [0, 6]]；M + [[1], [-1]] = [[5, 0], [-1, 1]]。
- A = [[2, 0, 5], [1, 3, -1]]：shape (2, 3)；sum(axis=0) = [3 3 4]，sum(axis=1) = [7 3]，max(axis=1) = [5 3]，argmax(axis=1) = [2 1]。
- A = [[1, 0, 2], [-1, 3, 1]]，B = [[3, 1], [2, 1], [1, 0]]：np.dot(A, B) = [[5, 1], [4, 2]]，第一格 1×3 + 0×2 + 2×1 = 5；B 换成 2×2 时报错 `shapes (2,3) and (2,2) not aligned: 3 (dim 1) != 2 (dim 0)`。
- X = [1.0, 0.5]，W = [[0.2, -0.4, 0.1], [0.6, 0.2, -0.3]]，b = [0.1, 0.1, 0.2]：Y = [0.6, -0.2, 0.15]；两行输入 [[1.0, 0.5], [0.0, 2.0]] 得到 [[0.6, -0.2, 0.15], [1.3, 0.5, -0.4]]，shape (2, 3)。
- temps = [23, 8, 31, 17, 5, 26]：temps > 20 → [T F T F F T]，temps[mask] = [23 31 26]，mask.astype(int) = [1 0 1 0 0 1]，共 3 天；[2.7, -1.6, 0.4].astype(int) = [2 -1 0]（向 0 截断）。
