# tools

## run_full.py：台式机完整训练

视频台词里写成【待填:Lxx_xxx】的数字（完整训练才有的精度、曲线等），都由 `tools/run_full.py` 在台式机上一次跑出来，
写进 `results/full_results.json` 的 `values` 里，键名和占位符一一对应。

### 用法（在仓库根目录运行，Windows / macOS / Linux 都一样）

**台式机开机后的顺序：**

```bash
python tools/run_full.py --figures-only       # 1. 先出确定性的图（约 5 秒），L00–L05、L07 等马上可以开渲
python tools/run_full.py --jobs 4             # 2. 再跑完整训练（N 按核数选，见下文「并行」）
python tools/run_full.py --check-keys         # 3. 每个占位符都显示「已有值」就齐了
python tools/run_full.py --check-images       # 4. 台词引用的每张图都显示「完整模式 … 生成」就齐了
```

全部命令：

```bash
python tools/run_full.py --list          # 看全部任务、每个任务产出的键、是否已完成
python tools/run_full.py --check-keys    # 扫 dl-video-p2/out/*/scripts/*.json 里的【待填:…】，报告缺哪些键
python tools/run_full.py --check-images  # 扫台词里 RepoImage 引用的仓库图：在不在、哪个任务生成、是不是完整模式生成的
python tools/run_full.py --quick         # 先冒烟：QUICK=1 全流程跑一遍（2 线程串行约 80 秒，--jobs 4 约 26 秒），结果写到 results_quick/
python tools/run_full.py --figures-only  # 只跑确定性出图任务（见下文），几秒跑完
python tools/run_full.py --no-figures    # 跳过所有 FIG_* 出图任务
python tools/run_full.py --jobs 4        # 最多 4 个任务同时跑（各自一个子进程）
python tools/run_full.py                 # 完整训练：跑全部还没完成的任务
python tools/run_full.py --only L14      # 只跑第14讲（L14_train、L14_filter、FIG_L14_*）；也可写任务名，可写多个；--only FIG 是全部出图任务
python tools/run_full.py --force --only L10_train   # 已完成的任务也重跑
python tools/run_full.py --threads 8     # 每个子进程的线程数（OPENBLAS/OMP/MKL_NUM_THREADS），默认 总核数 // jobs，至少 1
python tools/run_full.py -v              # 把各脚本的输出同时打到屏幕
```

- **断点续跑：** 每个任务跑完就把结果原子写入 `results/full_results.json`（先写临时文件再 rename）。中途关机或 Ctrl+C 后直接重跑同一条命令，已完成的任务会跳过。
- **日志：** 每个任务一份，存在 `results/logs/<任务>.log`（utf-8）。日志开头记录开始时间、模式、线程数、仓库提交号、Python/NumPy 版本和机器信息，中间是脚本的全部 stdout+stderr，结尾是结束时间和用时。
- **依赖：** `L14_filter` 和 `FIG_L14_apply_filter` 依赖 `L14_train`；`L15_misclassified` 和 `L15_half_float` 依赖 `L15_train`。训练任务会记下参数文件（`lecture14/output/simple_convnet_params.pkl`、`lecture15/output/deep_convnet_params.pkl`）的 sha256。依赖它的任务开跑前会核对：文件被别的运行覆盖过（比如之后又跑了 `--quick`），就拒绝运行，提示先重跑训练任务。
- **并行（`--jobs N`）：** 主进程开 N 个子进程同时跑任务，每个子进程的线程数默认是 `总核数 // N`（至少 1），`--threads` 可覆盖。依赖照样遵守：`L14_filter` 等 `L14_train` 跑完，`L15_misclassified`、`L15_half_float` 等 `L15_train` 跑完（会出 `filters.png`、`misclassified.png` 这类依赖参数的图的，就是这几个任务）。可开跑的任务按预计用时从长到短启动，所以 `L15_train` 总是第一个开。`full_results.json` 只由主进程写（子进程把结果交给主进程），不会并发覆盖。一个任务失败只影响依赖它的任务，其它照跑，最后汇总成功、失败各几个。
  - 选 N：L15_train 一个任务就要 3 小时左右，总时长基本由它决定。建议 N = 3 或 4：L15_train 自己留一半核左右，其余任务在别的子进程里跑完。N 太大时每个子进程线程太少，L15_train 反而变慢。
- **出图任务（`FIG_*`）：** 各讲 `output/` 下的图，按生成脚本一个脚本一个任务，共 22 个（20 个确定性、1 个依赖训练参数、1 个自己训练）。完整模式下全部重跑，让每张图都来自当前代码。训练任务自己出的图（`optimizer_mnist.png`、`filters.png`、`misclassified.png`）随训练任务一起生成，不再单列。
  - **确定性出图（20 个）：** 不依赖训练，`--figures-only` 只跑这些。逐个读过脚本：只用 numpy、matplotlib、common 里的函数、仓库里提交的数据（lecture03/data/*.csv）和 MNIST 原始数据，不读 pkl、不读训练结果，缺文件时也不会自己训练。2026-11-15 在移走全部 pkl 的新克隆里实测：QUICK 跑一次、完整模式跑两次，png 的 sha256 全部一致，所以台式机上跑出来的和这里一样。它们是：lecture02 的 array_image、loss_curve、sin_cos、sin_cos_graph、subplots；lecture03 的 training_log；lecture04 的 decision_lines；lecture05 的 plot_activations、relu、sig_step_compare、sigmoid、step_function；lecture07 的 cross_entropy_curve；lecture08 的 gradient_1d、gradient_2d、gradient_demo、gradient_method；lecture11 的 optimizer_compare_naive、weight_init_activation_histogram；lecture15 的 data_augmentation。
    - 注意：array_image、data_augmentation 要读 MNIST，第一次跑会先下载。
  - **要等训练参数的出图（1 个）：** `FIG_L14_apply_filter` 依赖 `L14_train`。apply_filter.py 有 `lecture14/output/simple_convnet_params.pkl` 就读，没有就当场训练一小会儿，结果不确定，所以不进 `--figures-only`。开跑前按 `L14_filter` 的规则核对 pkl 的 sha256，跑完核对输出写的是「读取已保存的参数」，否则判失败。
  - **要训练的出图（1 个）：** `FIG_L08_train_neuralnet` 对齐鱼书 ch04 的 train_neuralnet.py，用解析梯度跑 10000 步，约 1 分钟，只留下 `lecture08/output/train_neuralnet.png`。QUICK 只跑 20 步，图会变，所以不进 `--figures-only`。第8讲视频不用它，也不产键。
- **产物记录：** 每个任务跑完，把它在本讲 `output/` 里写过的文件都记进 `full_results.json` 的 `artifacts`，每个文件记 sha256、生成任务、脚本、模式和时间。产物按子进程实际以写方式打开过的文件来认，所以并行时也不会记错。`--check-images` 就靠这张表判断一张图是不是完整模式生成的、之后有没有被改过，并写出生成它的任务属于哪一类：确定性出图（`--figures-only` 就能生成）、要等训练参数、或者要训练。
  - `lecture08/output/train_log.json` 由 `L08_numerical`（train_numerical.py 不开 QUICK）重新生成，也记在这张表里。
  - `lecture08/results/train_log.json` 是已提交的一次完整运行记录，loss 保留 5 位小数。没有脚本会写 `results/`，run_full 也不碰它。
- **数据检查：** 完整模式下如果 MNIST 下载失败、脚本换成了合成数据，该任务记为失败，不写结果。
- **不改讲次脚本：** 每个任务在子进程（`sys.executable`）里导入对应脚本，调用它自己的 `main()`，步数、数据量、种子都由脚本决定。数值从脚本函数的返回值（套一层记录器）、`Trainer` 实例的精度曲线和打印输出里取。
- **结果格式：** `values` 里一个键一个值，曲线类的键（`*_curve`）是列表。`tasks` 里记每个任务的状态、用时、数据来源、版本信息。
- **QUICK 和完整分开存：** QUICK 的结果在 `results_quick/`，不进 git；完整结果在 `results/`，其中 `results/logs/` 不进 git。两种模式不会写进同一个文件。

### 任务和预计耗时

下表的「本机 QUICK」是 2026-11-15 在共享 Linux 机上实测的（8 核，只用 2 线程，Python 3.13、NumPy 2.5）。
「完整训练（估计）」是按迭代次数和评估数据量外推出来的，**只是粗估**。台式机核多、内存带宽大时会快不少，
DeepConvNet 这类卷积网络受 CPU 影响最大。

| 任务 | 脚本 | 本机 QUICK | 完整训练（估计） |
|---|---|---|---|
| L06_inference | lecture06/mnist_inference.py（现成权重，只推理） | 0.6 秒 | 几秒 |
| L08_numerical | lecture08/train_numerical.py（数值梯度 1500 步 + 原尺寸梯度耗时估算） | 3.3 秒 | 约 8 分钟（已有一次实测：训练 289 秒） |
| L10_train | lecture10/train_neuralnet.py（10000 步） | 0.8 秒 | 约 1 分钟 |
| L11_optimizer | lecture11/optimizer_compare_mnist.py（4 种优化器 × 2000 次） | 2.8 秒 | 约 1 分钟 |
| L11_weight_init | lecture11/weight_init_compare.py（3 种初始值 × 2000 次） | 2.4 秒 | 约 1 分钟 |
| L12_batchnorm | lecture12/batch_norm_test.py（16 个标准差 × 20 epoch） | 0.8 秒 | 约 1 分钟 |
| L12_weight_decay | lecture12/overfit_weight_decay.py（201 epoch × 2） | 2.5 秒 | 约 1 分钟 |
| L12_dropout | lecture12/overfit_dropout.py（301 epoch × 2） | 6.3 秒 | 约 1 分钟 |
| L12_hparam | lecture12/hyperparameter_optimization.py（2 轮 × 100 组 × 50 epoch） | 9.1 秒 | 约 3 到 5 分钟 |
| L14_train | lecture14/train_convnet.py（SimpleConvNet，20 epoch） | 9.5 秒 | 约 30 分钟（按 QUICK 的 200 倍迭代外推） |
| L14_filter | lecture14/visualize_filter.py | 0.9 秒 | 几秒 |
| L15_train | lecture15/train_deepnet.py（DeepConvNet，20 epoch） | 21.2 秒 | 约 3 小时（按 QUICK 的 500 倍迭代外推） |
| L15_misclassified | lecture15/misclassified_mnist.py（10000 张推理） | 1.8 秒 | 约 1 分钟 |
| L15_half_float | lecture15/half_float_network.py（64 位、16 位各推理 10000 张） | 2.0 秒 | 约 1 到 2 分钟 |
| FIG_L08_train_neuralnet | lecture08/train_neuralnet.py（解析梯度 10000 步，只出图） | 1.3 秒 | 约 1 分钟 |
| 20 个确定性出图任务 | 见上文 | 共约 10 秒（--jobs 4 约 4 秒） | 同左（和 QUICK 完全一样） |
| FIG_L14_apply_filter | lecture14/apply_filter.py（读 L14_train 的参数） | 1 秒 | 几秒（排在 L14_train 之后） |
| **合计** | | **串行约 80 秒，--jobs 4 约 26 秒** | **串行约 3.5 到 4.5 小时；--jobs 3~4 时约等于 L15_train 一个任务的时间（约 3 小时）** |

不想并行时，也可以分批跑：先 `--only L06 L08 L10 L11 L12`（半小时内跑完），再 `--only L14`，最后 `--only L15`。

### 路径和环境变量（Windows 台式机也一样）

- 代码仓库自己的路径都按 `tools/run_full.py` 的位置推导，不写死目录。
- `--check-keys`、`--check-images` 要读制作仓库 dl-video-production 的 `out/*/scripts/*.json`：
  - 先看环境变量 `DLVIS_SCRIPTS_ROOT`，否则找与代码仓库并排的 `../dl-video-production/out`，也可以用 `--scripts-root` 指定。
- 制作仓库的生成器（out/C/gen_L*.py）用环境变量 `DLVIS_REPO` 找代码仓库，没有就找并排的 `../dl-from-scratch-visual`。

### 键的说明

- **`L08_numgrad_full_seconds`：** 不开 QUICK 跑 lecture08/train_numerical.py 时，最后一行「原尺寸 784 → 50 → 10 有 N 个参数，算一次数值梯度估计要 X 秒」里的 X，按打印的一位小数取。注意这是估计值：脚本抽样一部分参数计时，再按参数总数放大。同一行的 N 存成 `L08_full_grad_params`。
- **`L08_num_seconds`（必填键）：** 不开 QUICK 跑 train_numerical.py 时打印的「训练 1500 步（49 → 20 → 10，…），用时 X 秒」里的 X，即 1500 步数值梯度训练的实测总耗时，一位小数。已提交的 lecture08/results/train_log.json 里那次是 289.2 秒。它和 `L08_numgrad_full_seconds` 都是必填键，`--check-keys` 会单列这两个必填键；台词里暂时没有它的占位符时只做提示，不算缺。
- **精度类的键（`*_acc`）：** 存的是 0 到 1 的小数（如 0.9734）。填进台词时一律念成百分数（「百分之九十七点三」），不要念小数。
