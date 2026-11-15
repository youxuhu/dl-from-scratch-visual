"""台式机一键跑完整训练（非 QUICK），把视频台词里【待填:Lxx_xxx】需要的数值写进 results/full_results.json。

用法（在仓库根目录）：
    python tools/run_full.py                 # 跑全部还没完成的任务（断点续跑）
    python tools/run_full.py --list          # 列出任务、产出的键、完成状态
    python tools/run_full.py --only L14      # 只跑第14讲的任务（可写任务名或讲次前缀，可多个）
    python tools/run_full.py --force --only L10_train   # 已完成也重跑
    python tools/run_full.py --quick         # QUICK=1 冒烟测试，结果写到 results_quick/
    python tools/run_full.py --check-keys    # 扫视频台词里的【待填:…】，报告哪些键没有任务产出
    python tools/run_full.py --threads 4     # OPENBLAS/OMP/MKL 线程数（默认用全部核）

做法：每个任务在单独的子进程里（sys.executable）导入对应讲次的脚本，原样调用脚本自己的 main()，
设置（步数、数据量、种子）完全由脚本决定，这里不复制参数；需要的数值从脚本函数的返回值
（运行前给函数套一层记录器）和打印输出里取。不修改任何讲次脚本。
每个任务完成后立刻把结果原子写入 JSON（先写临时文件再 os.replace），重跑时跳过已完成的任务。
日志：<结果目录>/logs/<任务>.log（utf-8，含开始结束时间、Python/NumPy 版本、机器信息、全部 stdout+stderr）。
"""
import argparse
import datetime as _dt
import hashlib
import io
import json
import os
import platform
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_SCRIPTS_ROOT = REPO.parent.parent / "dl-video-p2" / "out"
PLACEHOLDER = re.compile(r"【待填:([A-Za-z0-9_]+)】")


def now():
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


# --------------------------------------------------------------------------- 任务表
# name: (讲次目录, 脚本, 依赖任务, 产出的键, 说明, 本机 QUICK→完整 的粗略工作量倍数)
OPT = ("sgd", "momentum", "adagrad", "adam")
INIT = ("std001", "xavier", "he")

TASKS = {
    "L06_inference": dict(
        dir="lecture06", script="mnist_inference.py", deps=[],
        keys=["L06_one_by_one_acc", "L06_batch_acc"],
        desc="第6讲：用仓库里现成的权重推理 10000 张测试图（不训练）", scale=10),
    "L08_numerical": dict(
        dir="lecture08", script="train_numerical.py", deps=[],
        keys=["L08_num_loss_curve", "L08_num_train_acc_curve", "L08_num_test_acc_curve",
              "L08_num_final_train_acc", "L08_num_final_test_acc", "L08_num_seconds",
              "L08_full_grad_params", "L08_full_grad_seconds_est"],
        desc="第8讲：数值梯度训练 7×7 缩小版 1500 步，并估算原尺寸一次数值梯度的耗时", scale=75),
    "L08_train_neuralnet": dict(
        dir="lecture08", script="train_neuralnet.py", deps=[],
        keys=["L08_train_acc_curve", "L08_test_acc_curve", "L08_final_train_acc",
              "L08_final_test_acc", "L08_final_loss"],
        desc="第8讲：对齐鱼书的 train_neuralnet，解析梯度 10000 步", scale=500),
    "L10_train": dict(
        dir="lecture10", script="train_neuralnet.py", deps=[],
        keys=["L10_test_acc", "L10_train_acc", "L10_train_acc_curve", "L10_test_acc_curve",
              "L10_loss_first", "L10_loss_last"],
        desc="第10讲：反向传播训练 TwoLayerNet 10000 步", scale=40),
    "L11_optimizer": dict(
        dir="lecture11", script="optimizer_compare_mnist.py", deps=[],
        keys=[f"L11_{k}_acc" for k in OPT] + [f"L11_{k}_final_loss" for k in OPT]
        + [f"L11_{k}_loss_curve" for k in OPT],
        desc="第11讲：MNIST 上比较 SGD/Momentum/AdaGrad/Adam，2000 次迭代", scale=18),
    "L11_weight_init": dict(
        dir="lecture11", script="weight_init_compare.py", deps=[],
        keys=[f"L11_init_{k}_acc" for k in INIT] + [f"L11_init_{k}_final_loss" for k in INIT]
        + [f"L11_init_{k}_loss_curve" for k in INIT],
        desc="第11讲：MNIST 上比较 std=0.01、Xavier、He 三种初始值，2000 次迭代", scale=14),
    "L12_batchnorm": dict(
        dir="lecture12", script="batch_norm_test.py", deps=[],
        keys=["L12_bn_win", "L12_bn_total", "L12_bn_small_std", "L12_bn_small_acc", "L12_nobn_small_acc",
              "L12_bn_table"],
        desc="第12讲：16 种初始值标准差下有无 BN 的训练精度（1000 张，20 个 epoch）", scale=67),
    "L12_weight_decay": dict(
        dir="lecture12", script="overfit_weight_decay.py", deps=[],
        keys=["L12_wd0_train", "L12_wd0_test", "L12_wd0_gap", "L12_wd0_wsum",
              "L12_wd01_train", "L12_wd01_test", "L12_wd01_gap", "L12_wd01_wsum",
              "L12_wd0_train_curve", "L12_wd0_test_curve", "L12_wd01_train_curve", "L12_wd01_test_curve"],
        desc="第12讲：300 张训练图制造过拟合，λ = 0 和 0.1，201 个 epoch", scale=4),
    "L12_dropout": dict(
        dir="lecture12", script="overfit_dropout.py", deps=[],
        keys=["L12_drop0_train", "L12_drop0_test", "L12_drop_gap0",
              "L12_drop02_train", "L12_drop02_test", "L12_drop_gap02",
              "L12_drop0_train_curve", "L12_drop0_test_curve", "L12_drop02_train_curve", "L12_drop02_test_curve"],
        desc="第12讲：同样设置下有无 Dropout(0.2)，301 个 epoch", scale=3),
    "L12_hparam": dict(
        dir="lecture12", script="hyperparameter_optimization.py", deps=[],
        keys=["L12_hp_best_val", "L12_hp_best_lr", "L12_hp_best_wd", "L12_hp_first_top5",
              "L12_hp_lr_log_range", "L12_hp_wd_log_range"],
        desc="第12讲：超参数随机搜索，每轮 100 组、每组 50 个 epoch，两轮", scale=21),
    "L14_train": dict(
        dir="lecture14", script="train_convnet.py", deps=[],
        keys=["L14_test_acc", "L14_train_acc_curve", "L14_test_acc_curve"],
        desc="第14讲：SimpleConvNet，60000 张，20 个 epoch（参数存到 lecture14/output/）", scale=200),
    "L14_filter": dict(
        dir="lecture14", script="visualize_filter.py", deps=["L14_train"],
        keys=["L14_filter_roughness_before", "L14_filter_roughness_after"],
        desc="第14讲：读取 L14_train 存下的参数，比较训练前后第 1 层滤波器的平滑程度", scale=1),
    "L15_train": dict(
        dir="lecture15", script="train_deepnet.py", deps=[],
        keys=["L15_test_acc", "L15_train_acc_curve", "L15_test_acc_curve"],
        desc="第15讲：DeepConvNet，60000 张，20 个 epoch（参数存到 lecture15/output/）", scale=500),
    "L15_misclassified": dict(
        dir="lecture15", script="misclassified_mnist.py", deps=["L15_train"],
        keys=["L15_misclassified_count", "L15_misclassified_examples", "L15_misclassified_acc"],
        desc="第15讲：用 L15_train 的参数在 10000 张测试图上挑出认错的样本", scale=33),
    "L15_half_float": dict(
        dir="lecture15", script="half_float_network.py", deps=["L15_train"],
        keys=["L15_fp64_acc", "L15_fp16_acc"],
        desc="第15讲：用 L15_train 的参数比较 float64 和 float16 的精度（10000 张）", scale=33),
}
PARAM_FILES = {"L14_train": "lecture14/output/simple_convnet_params.pkl",
               "L15_train": "lecture15/output/deep_convnet_params.pkl"}


# --------------------------------------------------------------------------- 子进程里跑的部分
class Tee(io.TextIOBase):
    """把脚本打印的内容同时写到真正的 stdout（父进程写进日志）和一个缓冲区（用来解析数值）。"""

    def __init__(self, real):
        self.real, self.buf = real, io.StringIO()

    def write(self, s):
        self.real.write(s)
        self.real.flush()
        self.buf.write(s)
        return len(s)

    def flush(self):
        self.real.flush()


def _load(task):
    import importlib.util
    spec = TASKS[task]
    path = REPO / spec["dir"] / spec["script"]
    mod_name = f"_runfull_{spec['dir']}_{path.stem}"
    m = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(m)
    sys.modules[mod_name] = mod
    m.loader.exec_module(mod)
    return mod


def _record(mod, name, store):
    """给 mod.name 套一层：每次调用把 (参数, 返回值) 追加到 store。脚本的 main() 通过全局名字调用，会走到这层。"""
    orig = getattr(mod, name)

    def wrapper(*a, **k):
        out = orig(*a, **k)
        store.append((a, k, out))
        return out
    setattr(mod, name, wrapper)


def _floats(xs):
    return [float(v) for v in xs]


def _grab(text, pattern, cast=float, default=None):
    m = re.findall(pattern, text)
    if not m:
        if default is not None:
            return default
        raise RuntimeError(f"输出里找不到 {pattern!r}")
    return cast(m[-1])


def worker(task, out_path):
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / TASKS[task]["dir"]))
    os.chdir(REPO / TASKS[task]["dir"])
    tee = Tee(sys.stdout)
    sys.stdout = tee
    import numpy as np
    print(f"[run_full] 任务 {task}，QUICK={os.environ.get('QUICK', '')}，"
          f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS')}，numpy {np.__version__}")

    # 记录每次 get_mnist 的数据来源（完整训练必须是 MNIST，不能是下载失败后的合成数据）
    import common.util as cu
    sources = []
    orig_get = cu.get_mnist

    def get_mnist(*a, **k):
        out = orig_get(*a, **k)
        sources.append(out[2])
        return out
    cu.get_mnist = get_mnist
    # 记录每个 Trainer 实例，训练完从里面取精度曲线
    import common.trainer as ct
    trainers = []
    orig_init = ct.Trainer.__init__

    def trainer_init(self, *a, **k):
        orig_init(self, *a, **k)
        trainers.append(self)
    ct.Trainer.__init__ = trainer_init

    calls = {}
    mod = _load(task)
    hooks = {"L08_numerical": ["train", "time_full_size_gradient"], "L10_train": ["train"],
             "L11_optimizer": ["compare"], "L11_weight_init": ["compare"], "L12_batchnorm": ["run"],
             "L12_weight_decay": ["run"], "L12_dropout": ["run"], "L12_hparam": ["search", "narrowed"],
             "L14_filter": ["roughness"], "L15_misclassified": ["find_mistakes", "get_mnist"]}
    for name in hooks.get(task, []):
        calls[name] = []
        _record(mod, name, calls[name])
    mod.main()
    text = tee.buf.getvalue()
    r = EXTRACT[task](text, calls, trainers, np)
    meta = {"data_sources": sorted(set(sources))}
    missing = [k for k in TASKS[task]["keys"] if k not in r]
    if missing:
        raise RuntimeError(f"任务没有产出这些键：{missing}")
    Path(out_path).write_text(json.dumps({"values": r, "meta": meta}, ensure_ascii=False), encoding="utf-8")


def ex_L06(text, calls, trainers, np):
    return {"L06_one_by_one_acc": _grab(text, r"一张一张算（前 \d+ 张）精度: ([0-9.]+)"),
            "L06_batch_acc": _grab(text, r"100 张一批（全部 10000 张）精度: ([0-9.]+)")}


def ex_L08_num(text, calls, trainers, np):
    _, _, (net, log) = calls["train"][-1]
    r = {"L08_num_loss_curve": _floats(log["loss"]), "L08_num_train_acc_curve": _floats(log["train_acc"]),
         "L08_num_test_acc_curve": _floats(log["test_acc"]), "L08_num_final_train_acc": float(log["final_train_acc"]),
         "L08_num_final_test_acc": float(log["final_test_acc"]), "L08_num_seconds": float(log["seconds"])}
    if calls["time_full_size_gradient"]:
        n_total, est = calls["time_full_size_gradient"][-1][2]
        r["L08_full_grad_params"], r["L08_full_grad_seconds_est"] = int(n_total), float(est)
    else:  # QUICK 时脚本不做这一步
        r["L08_full_grad_params"], r["L08_full_grad_seconds_est"] = None, None
    return r


def ex_L08_nn(text, calls, trainers, np):
    pairs = re.findall(r"train acc, test acc \| ([0-9.]+), ([0-9.]+)", text)
    if not pairs:
        raise RuntimeError("输出里没有 train acc, test acc 行")
    tr, te = [float(a) for a, _ in pairs], [float(b) for _, b in pairs]
    return {"L08_train_acc_curve": tr, "L08_test_acc_curve": te, "L08_final_train_acc": tr[-1],
            "L08_final_test_acc": te[-1], "L08_final_loss": _grab(text, r"最后一步 loss: ([0-9.]+)")}


def ex_L10(text, calls, trainers, np):
    _, _, (net, hist, source) = calls["train"][-1]
    return {"L10_test_acc": float(hist["test_acc"][-1]), "L10_train_acc": float(hist["train_acc"][-1]),
            "L10_train_acc_curve": _floats(hist["train_acc"]), "L10_test_acc_curve": _floats(hist["test_acc"]),
            "L10_loss_first": float(hist["loss"][0]), "L10_loss_last": float(hist["loss"][-1])}


def _compare_table(text, calls, names, keys, prefix, acc_re):
    _, _, (losses, nets) = calls["compare"][-1]
    r = {}
    for name, key in zip(names, keys):
        r[f"{prefix}{key}_acc"] = _grab(text, acc_re.format(name=re.escape(name)))
        r[f"{prefix}{key}_final_loss"] = _grab(
            text, r"\s*{name}: .*?平滑后最终损失 ([0-9.]+)".format(name=re.escape(name)))
        r[f"{prefix}{key}_loss_curve"] = _floats(losses[name])
    return r


def ex_L11_opt(text, calls, trainers, np):
    return _compare_table(text, calls, ["SGD", "Momentum", "AdaGrad", "Adam"], OPT, "L11_",
                          r"\s*{name}: 平滑后最终损失 [0-9.]+，测试精度 ([0-9.]+)")


def ex_L11_init(text, calls, trainers, np):
    return _compare_table(text, calls, ["std=0.01", "Xavier", "He"], INIT, "L11_init_",
                          r"\s*{name}: 起始损失 [0-9.]+，平滑后最终损失 [0-9.]+，测试精度 ([0-9.]+)")


def ex_L12_bn(text, calls, trainers, np):
    table = []
    for a, k, curves in calls["run"]:
        std = float(a[2] if len(a) > 2 else k["std"])
        table.append({"std": std, "bn": float(curves["BN"][-1]), "nobn": float(curves["无BN"][-1]),
                      "bn_curve": _floats(curves["BN"]), "nobn_curve": _floats(curves["无BN"])})
    small = min(table, key=lambda row: row["std"])
    return {"L12_bn_win": sum(row["bn"] > row["nobn"] for row in table), "L12_bn_total": len(table),
            "L12_bn_small_std": small["std"], "L12_bn_small_acc": small["bn"], "L12_nobn_small_acc": small["nobn"],
            "L12_bn_table": table}


def ex_L12_wd(text, calls, trainers, np):
    r = {}
    for a, k, (tr, te, wsize) in calls["run"]:
        lam = float(a[4] if len(a) > 4 else k["lam"])
        tag = {0.0: "wd0", 0.1: "wd01"}[lam]
        r.update({f"L12_{tag}_train": float(tr[-1]), f"L12_{tag}_test": float(te[-1]),
                  f"L12_{tag}_gap": float(tr[-1] - te[-1]), f"L12_{tag}_wsum": float(wsize),
                  f"L12_{tag}_train_curve": _floats(tr), f"L12_{tag}_test_curve": _floats(te)})
    return r


def ex_L12_drop(text, calls, trainers, np):
    r = {}
    for a, k, (tr, te) in calls["run"]:
        flag = bool(a[4] if len(a) > 4 else k["use_dropout"])
        tag = "02" if flag else "0"
        r.update({f"L12_drop{tag}_train": float(tr[-1]), f"L12_drop{tag}_test": float(te[-1]),
                  f"L12_drop_gap{tag}": float(tr[-1] - te[-1]),
                  f"L12_drop{tag}_train_curve": _floats(tr), f"L12_drop{tag}_test_curve": _floats(te)})
    return r


def ex_L12_hp(text, calls, trainers, np):
    first = calls["search"][0][2]
    second = calls["search"][-1][2]
    acc, lr, wd = second[0]
    lr_r, wd_r = calls["narrowed"][-1][2]
    return {"L12_hp_best_val": float(acc), "L12_hp_best_lr": float(lr), "L12_hp_best_wd": float(wd),
            "L12_hp_first_top5": [{"val_acc": float(a), "lr": float(b), "wd": float(c)} for a, b, c in first[:5]],
            "L12_hp_lr_log_range": _floats(lr_r), "L12_hp_wd_log_range": _floats(wd_r)}


def _trainer_curves(prefix, text, trainers):
    if not trainers:
        raise RuntimeError("没有捕获到 Trainer")
    t = trainers[-1]
    return {f"{prefix}_test_acc": _grab(text, r"测试精度 ([0-9.]+)；参数已保存"),
            f"{prefix}_train_acc_curve": _floats(t.train_acc_list), f"{prefix}_test_acc_curve": _floats(t.test_acc_list)}


def ex_L14_train(text, calls, trainers, np):
    return _trainer_curves("L14", text, trainers)


def _need_saved_params(text, marker):
    if marker not in text:
        raise RuntimeError("脚本没有读取训练好的参数（参数文件不存在？），结果不能用")


def ex_L14_filter(text, calls, trainers, np):
    _need_saved_params(text, "读取已保存的参数")
    before, after = calls["roughness"][0][2], calls["roughness"][1][2]
    return {"L14_filter_roughness_before": float(before.mean()), "L14_filter_roughness_after": float(after.mean())}


def ex_L15_train(text, calls, trainers, np):
    return _trainer_curves("L15", text, trainers)


def ex_L15_mis(text, calls, trainers, np):
    _need_saved_params(text, "读取 train_deepnet.py 保存的参数")
    _, _, (preds, wrong) = calls["find_mistakes"][-1]
    t_test = calls["get_mnist"][-1][2][1][1]
    return {"L15_misclassified_count": int(len(wrong)),
            "L15_misclassified_acc": float(1 - len(wrong) / len(t_test)),
            "L15_misclassified_examples": [{"index": int(i), "pred": int(preds[i]), "label": int(t_test[i])}
                                           for i in wrong[:20]]}


def ex_L15_half(text, calls, trainers, np):
    _need_saved_params(text, "读取 train_deepnet.py 保存的参数")
    return {"L15_fp64_acc": _grab(text, r"64 位浮点数：精度 ([0-9.]+)"),
            "L15_fp16_acc": _grab(text, r"16 位浮点数：精度 ([0-9.]+)")}


EXTRACT = {"L06_inference": ex_L06, "L08_numerical": ex_L08_num, "L08_train_neuralnet": ex_L08_nn,
           "L10_train": ex_L10, "L11_optimizer": ex_L11_opt, "L11_weight_init": ex_L11_init,
           "L12_batchnorm": ex_L12_bn, "L12_weight_decay": ex_L12_wd, "L12_dropout": ex_L12_drop,
           "L12_hparam": ex_L12_hp, "L14_train": ex_L14_train, "L14_filter": ex_L14_filter,
           "L15_train": ex_L15_train, "L15_misclassified": ex_L15_mis, "L15_half_float": ex_L15_half}


# --------------------------------------------------------------------------- 父进程：调度、续跑、日志
def load_results(path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"meta": {}, "tasks": {}, "values": {}}


def save_results(path, data):
    """原子写：先写同目录下的临时文件，再 os.replace 覆盖。中途断电也不会留下半个 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def machine_info():
    try:
        import numpy as np
        npv = np.__version__
    except ImportError:
        npv = "未安装"
    return {"python": sys.version.split()[0], "python_exe": sys.executable, "numpy": npv,
            "platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor() or "", "cpu_count": os.cpu_count(), "node": platform.node()}


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True,
                              text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def select(only):
    if not only:
        return list(TASKS)
    picked = []
    for item in only:
        for part in item.split(","):
            part = part.strip()
            hit = [t for t in TASKS if t == part or t.startswith(part + "_") or t.split("_")[0] == part]
            if not hit:
                raise SystemExit(f"不认识的任务或讲次：{part}（用 --list 看全部任务）")
            picked += [t for t in hit if t not in picked]
    return [t for t in TASKS if t in picked]   # 保持任务表顺序，依赖总在前面


def run_task(task, args, results, results_path, log_dir):
    spec = TASKS[task]
    for dep in spec["deps"]:
        d = results["tasks"].get(dep, {})
        if d.get("status") != "done":
            raise RuntimeError(f"依赖任务 {dep} 还没完成，先跑它（--only {dep}）")
        rel = PARAM_FILES.get(dep)
        if rel:
            f = REPO / rel
            if not f.exists() or sha256(f) != d.get("param_sha256"):
                raise RuntimeError(f"{rel} 不是 {dep} 这次运行存下的参数（文件不存在或被改过），请 --force --only {dep}")
    env = dict(os.environ)
    env.pop("QUICK", None)
    if args.quick:
        env["QUICK"] = "1"
    for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        env[v] = str(args.threads)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["MPLBACKEND"] = "Agg"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{task}.log"
    tmp_out = results_path.with_name(f".{task}.worker.json")
    info = machine_info()
    started, t0 = now(), time.time()
    with open(log_path, "w", encoding="utf-8", newline="\n") as log:
        log.write(f"任务 {task}：{spec['desc']}\n脚本 {spec['dir']}/{spec['script']}\n开始 {started}\n"
                  f"模式 {'QUICK=1 冒烟' if args.quick else '完整训练'}；线程数 {args.threads}；仓库提交 {git_commit()}\n"
                  f"机器 {json.dumps(info, ensure_ascii=False)}\n{'-' * 60}\n")
        log.flush()
        cmd = [sys.executable, str(Path(__file__).resolve()), "--_worker", task, "--_out", str(tmp_out)]
        proc = subprocess.Popen(cmd, cwd=REPO / spec["dir"], env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in proc.stdout:
            log.write(line)
            log.flush()
            if args.verbose:
                sys.stdout.write(f"  {line}")
        code = proc.wait()
        seconds = time.time() - t0
        log.write(f"{'-' * 60}\n结束 {now()}；用时 {seconds:.1f} 秒；退出码 {code}\n")
    if code != 0 or not tmp_out.exists():
        raise RuntimeError(f"子进程失败（退出码 {code}），看日志 {log_path}")
    payload = json.loads(tmp_out.read_text(encoding="utf-8"))
    tmp_out.unlink()
    srcs = payload["meta"]["data_sources"]
    if not args.quick and any(s != "MNIST" for s in srcs):
        raise RuntimeError(f"数据来源不是 MNIST：{srcs}（下载失败时会换成合成数据），结果不能用")
    rec = {"status": "done", "mode": "quick" if args.quick else "full", "started": started, "finished": now(),
           "seconds": round(seconds, 1), "log": str(log_path.relative_to(REPO)) if log_path.is_relative_to(REPO)
           else str(log_path), "keys": spec["keys"], "data_sources": srcs, "threads": args.threads,
           "python": info["python"], "numpy": info["numpy"], "platform": info["platform"], "commit": git_commit()}
    if task in PARAM_FILES:
        rec["param_file"] = PARAM_FILES[task]
        rec["param_sha256"] = sha256(REPO / PARAM_FILES[task])
    results["tasks"][task] = rec
    results["values"].update(payload["values"])
    results["meta"].update({"mode": rec["mode"], "updated": now()})
    save_results(results_path, results)
    return seconds


def scan_placeholders(root):
    """返回 {键: [(执行员, 文件), ...]}。"""
    found = {}
    for f in sorted(Path(root).glob("*/scripts/*.json")):
        for key in sorted(set(PLACEHOLDER.findall(f.read_text(encoding="utf-8")))):
            found.setdefault(key, []).append((f.parent.parent.name, f.name))
    return found


def check_keys(root, results):
    produced = {k: t for t, s in TASKS.items() for k in s["keys"]}
    found = scan_placeholders(root)
    print(f"扫描 {root}/*/scripts/*.json：共 {len(found)} 个不同的占位符键")
    missing = {k: v for k, v in found.items() if k not in produced}
    for key, where in sorted(found.items()):
        owner = "、".join(sorted({f"{a}/{b}" for a, b in where}))
        state = produced.get(key, "缺：没有任务产出")
        val = "已有值" if key in results.get("values", {}) and results["values"][key] is not None else "还没值"
        print(f"  {key:32s} {owner:28s} {state:22s} {val if key in produced else ''}")
    unused = sorted(k for k in produced if k not in found)
    print(f"缺 {len(missing)} 个键" + ("：" + "、".join(sorted(missing)) if missing else ""))
    print(f"run_full 额外产出、台词里暂未引用的键 {len(unused)} 个（可供其他执行员对齐）：" + "、".join(unused))
    return 1 if missing else 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--only", nargs="+", help="只跑这些任务；写任务名（L14_train）或讲次（L14），可逗号分隔")
    p.add_argument("--list", action="store_true", help="列出任务、产出的键和完成状态")
    p.add_argument("--force", action="store_true", help="已完成的任务也重跑")
    p.add_argument("--quick", action="store_true", help="QUICK=1 冒烟测试，默认写到 results_quick/")
    p.add_argument("--check-keys", action="store_true", help="扫台词里的【待填:…】，报告哪些键没有任务产出")
    p.add_argument("--scripts-root", type=Path, default=DEFAULT_SCRIPTS_ROOT,
                   help=f"台词 JSON 的根目录（下面是 A/B/C/scripts/*.json），默认 {DEFAULT_SCRIPTS_ROOT}")
    p.add_argument("--results-dir", type=Path, default=None, help="结果目录，默认 results/（--quick 时 results_quick/）")
    p.add_argument("--threads", type=int, default=os.cpu_count() or 1, help="OPENBLAS/OMP/MKL 线程数，默认全部核")
    p.add_argument("--keep-going", action="store_true", help="某个任务失败后继续跑后面的（默认也继续，只跳过依赖它的）")
    p.add_argument("-v", "--verbose", action="store_true", help="把脚本输出同时打到屏幕")
    p.add_argument("--_worker", help=argparse.SUPPRESS)
    p.add_argument("--_out", help=argparse.SUPPRESS)
    args = p.parse_args(argv)

    if args._worker:
        try:
            worker(args._worker, args._out)
        except Exception:
            traceback.print_exc()
            return 1
        return 0

    rdir = args.results_dir or (REPO / ("results_quick" if args.quick else "results"))
    results_path = rdir / "full_results.json"
    results = load_results(results_path)
    if results.get("meta", {}).get("mode") and results["meta"]["mode"] != ("quick" if args.quick else "full"):
        raise SystemExit(f"{results_path} 里是 {results['meta']['mode']} 模式的结果，不和本次混写；换 --results-dir")

    if args.check_keys:
        return check_keys(args.scripts_root, results)
    tasks = select(args.only)
    if args.list:
        for t in tasks:
            s, st = TASKS[t], results["tasks"].get(t, {})
            state = f"已完成 {st['finished']}（{st['seconds']} 秒）" if st.get("status") == "done" else "未完成"
            print(f"{t:20s} {state}\n    {s['desc']}\n    依赖：{'、'.join(s['deps']) or '无'}\n    键：{'、'.join(s['keys'])}")
        return 0

    print(f"结果文件 {results_path}；模式 {'QUICK 冒烟' if args.quick else '完整训练'}；线程 {args.threads}")
    failed, total = [], time.time()
    for t in tasks:
        if not args.force and results["tasks"].get(t, {}).get("status") == "done":
            print(f"跳过 {t}（已完成，--force 可重跑）")
            continue
        if any(d in failed for d in TASKS[t]["deps"]):
            print(f"跳过 {t}（依赖的任务失败了）")
            failed.append(t)
            continue
        print(f"开始 {t}：{TASKS[t]['desc']}  [{now()}]", flush=True)
        try:
            sec = run_task(t, args, results, results_path, rdir / "logs")
            print(f"完成 {t}，用时 {sec:.1f} 秒")
        except Exception as e:
            failed.append(t)
            results["tasks"][t] = {"status": "failed", "error": str(e), "finished": now()}
            save_results(results_path, results)
            print(f"失败 {t}：{e}")
    print(f"全部结束，用时 {time.time() - total:.1f} 秒；失败 {len(failed)} 个" + (f"：{failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
