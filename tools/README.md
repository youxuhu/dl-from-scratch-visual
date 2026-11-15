# tools

## run_full.py：台式机完整训练

视频台词里写成【待填:Lxx_xxx】的数字（完整训练才有的精度、曲线等），都由 `tools/run_full.py` 在台式机上一次跑出来，
写进 `results/full_results.json` 的 `values` 里，键名和占位符一一对应。

### 用法（在仓库根目录运行，Windows / macOS / Linux 都一样）

```bash
python tools/run_full.py --list          # 看全部任务、每个任务产出的键、是否已完成
python tools/run_full.py --check-keys    # 扫 dl-video-p2/out/*/scripts/*.json 里的【待填:…】，报告缺哪些键
python tools/run_full.py --quick         # 先冒烟：QUICK=1 全流程跑一遍（约 1 分钟），结果写到 results_quick/
python tools/run_full.py                 # 完整训练：跑全部还没完成的任务
python tools/run_full.py --only L14      # 只跑第14讲（L14_train、L14_filter）；也可写任务名，可写多个
python tools/run_full.py --force --only L10_train   # 已完成的任务也重跑
python tools/run_full.py --threads 8     # 线程数（OPENBLAS/OMP/MKL_NUM_THREADS），默认用全部核
python tools/run_full.py -v              # 把各脚本的输出同时打到屏幕
```

- **断点续跑：** 每个任务跑完就把结果原子写入 `results/full_results.json`（先写临时文件再 rename）。中途关机或 Ctrl+C 后直接重跑同一条命令，已完成的任务会跳过。
- **日志：** 每个任务一份，存在 `results/logs/<任务>.log`（utf-8）。日志开头记录开始时间、模式、线程数、仓库提交号、Python/NumPy 版本和机器信息，中间是脚本的全部 stdout+stderr，结尾是结束时间和用时。
- **依赖：** `L14_filter` 依赖 `L14_train`；`L15_misclassified` 和 `L15_half_float` 依赖 `L15_train`。训练任务会记下参数文件（`lecture14/output/simple_convnet_params.pkl`、`lecture15/output/deep_convnet_params.pkl`）的 sha256。依赖它的任务开跑前会核对：文件被别的运行覆盖过（比如之后又跑了 `--quick`），就拒绝运行，提示先重跑训练任务。
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
| L08_train_neuralnet | lecture08/train_neuralnet.py（解析梯度 10000 步） | 1.3 秒 | 约 1 分钟 |
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
| **合计** | | **约 65 秒** | **约 3.5 到 4.5 小时，大头是 L15_train** |

建议先跑 `python tools/run_full.py --only L06 L08 L10 L11 L12`（半小时内跑完），再跑 `--only L14`，最后跑 `--only L15`。
跑完后用 `python tools/run_full.py --check-keys`，每个占位符后面都显示「已有值」就齐了。
