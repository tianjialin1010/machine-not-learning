#!/usr/bin/env python3
"""多工业数据集评测 v2 · 修正版

背景：0.1.5 附带的 docs/industrial-report-review.md 对 9/28 的 v1 报告
提了 7 条方法论质疑。本脚本落实其中可实验验证的四项修正：

  M1【时序分组划分】SKAB / TEP / SECOM 是连续采样数据，v1 用随机行划分，
     同一时间段或同一次试验的数据可能同时进入训练与测试（泄漏风险）。
     v2 改为「在时序单元内按 60/20/20 顺序切分」，并保留 v1 口径做对照。

  M2【容量对照】v1 把 SECOM（474 维）表现差归因于隐层容量不足，但没做对照。
     v2 扫三档容量，看学习曲线，检验该归因是否成立。

  M3【消融】v1 只证明「接口可适配」。v2 对温度缩放、期望取值做消融，
     量化各组件对指标的贡献。

  M4【延迟口径】v1 把批量摊薄延迟与单条延迟混在一起比较。
     v2 同时给出两种口径并显式标注，不做跨口径比较。

用法：python3 multi_dataset_bench_v2.py [--quick]
输出：multi_bench_v2/multi_dataset_v2.json（同时打印摘要）
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

DS = os.path.expanduser("~/jev-service/ind_ds")
AI4I = os.path.expanduser("~/jev-service/data/ai4i2020.csv")
OUT = os.path.expanduser("~/jev-service/multi_bench_v2")
SEED = 20260921
EPOCHS = 600                     # 与 v1 保持一致，确保可比
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 容量档位：(hidden sizes)
CAPACITIES = {
    "small": (64, 32),          # v1 使用的容量
    "medium": (256, 128),
    "large": (512, 256, 128),
}


# ------------------------------------------------------------------ 基础工具
def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def ece_binary(p, y, bins=10):
    p = np.clip(p, 0, 1)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum():
            tot += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return float(tot)


def fit_temp(z, y, binary=True, iters=500, lr=0.05):
    T = 1.0
    for _ in range(iters):
        if binary:
            p = sigmoid(z / T)
            g = np.mean((y - p) * z / (T ** 2))
        else:
            P = softmax(z / T)
            oh = np.zeros_like(P)
            oh[np.arange(len(y)), y] = 1.0
            g = np.sum((oh - P) * z / (T ** 2)) / len(y)
        T = float(np.clip(T - lr * g, 0.05, 10.0))
    return T


class Head(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(64, 32)):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_head(Xtr, ytr, out_dim, hidden=(64, 32), dev=DEVICE, seed=SEED):
    torch.manual_seed(seed)
    m = Head(Xtr.shape[1], out_dim, hidden).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=250, gamma=0.5)
    lossf = nn.CrossEntropyLoss()
    Xt = torch.tensor(np.ascontiguousarray(Xtr), dtype=torch.float32, device=dev)
    yt = torch.tensor(ytr, dtype=torch.long, device=dev)
    m.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        lossf(m(Xt), yt).backward()
        opt.step()
        sch.step()
    m.eval()
    return m


def infer(m, X, dev=DEVICE):
    with torch.no_grad():
        return m(torch.tensor(np.ascontiguousarray(X, dtype=np.float32),
                              device=dev)).cpu().numpy()


# ------------------------------------------------------------------ 数据集
def load_ai4i():
    """AI4I 2020：无时序语义（每条是独立工况快照）。特征工程与 v1 完全一致。"""
    rows = list(csv.DictReader(open(AI4I, encoding="utf-8-sig")))
    tl = ["No Failure", "TWF", "HDF", "PWF", "OSF", "RNF"]

    def enc(r):
        air, proc = float(r["Air temperature [K]"]), float(r["Process temperature [K]"])
        rpm, tq = float(r["Rotational speed [rpm]"]), float(r["Torque [Nm]"])
        w = float(r["Tool wear [min]"])
        return [air, proc, rpm, tq, w, proc - air,
                rpm * tq * 2 * math.pi / 60 / 1000, w * tq / 100,
                (proc - air) / max(rpm / 1000, 1e-6)]

    X = np.array([enc(r) for r in rows], dtype=np.float32)
    y_n = np.array([int(r["Machine failure"]) for r in rows])
    y_c = np.array([tl.index(next((k for k in tl[1:] if int(r[k]) == 1),
                                  "No Failure")) for r in rows])
    return dict(key="ai4i", name="AI4I 2020", domain="机加工 / 数控机床",
                X=X, y_noul=y_n, groups=None, temporal=False,
                y_choice=y_c, choice_labels=tl,
                note=f"{len(X)} 条独立工况快照，9 维（5 原始 + 4 派生）；"
                     f"正例率 {y_n.mean()*100:.2f}%")


def load_steel():
    """Steel Plates：27 维特征 + 7 类 one-hot 标签，无正常类（只能评多分类）。"""
    arr = np.loadtxt(f"{DS}/x_steel/Faults.NNA")
    X = arr[:, :27].astype(np.float32)
    Y = arr[:, 27:34]
    labels = ["Pastry", "Z_Scratch", "K_Scatch", "Stains",
              "Dirtiness", "Bumps", "Other_Faults"]
    y_c = Y.argmax(1)
    return dict(key="steel", name="Steel Plates", domain="冶金 / 钢板表面",
                X=X, y_noul=None, groups=None, temporal=False,
                y_choice=y_c, choice_labels=labels,
                note=f"{len(X)} 条全为缺陷样本（无正常类），{X.shape[1]} 维，7 类")


def load_secom():
    """SECOM：半导体制造，按采集顺序排列，有隐含时间序。"""
    d = f"{DS}/x_secom"
    raw = np.genfromtxt(f"{d}/secom.data", dtype=float)        # NaN = 缺失
    lab = np.genfromtxt(f"{d}/secom_labels.data", dtype=str, usecols=0)
    y = (lab.astype(float) > 0).astype(int)
    col_med = np.nanmedian(raw, axis=0)
    col_med = np.where(np.isnan(col_med), 0.0, col_med)
    idx = np.where(np.isnan(raw))
    raw[idx] = np.take(col_med, idx[1])
    keep = raw.std(0) > 0
    X = raw[:, keep].astype(np.float32)
    n = len(X)
    return dict(key="secom", name="SECOM", domain="半导体 / 晶圆制造",
                X=X, y_noul=y, y_choice=None, choice_labels=None,
                groups=[(0, i) for i in range(n)], temporal=True,
                note=f"590 维传感器（列中位数填充缺失、去常数后保留 {int(keep.sum())} 维），"
                     f"{n} 条按采集顺序排列；正例率 {y.mean()*100:.1f}%")


def load_skab():
    """SKAB：水泵机组，多个独立试验文件，每个文件内部为连续时序。"""
    files = sorted(glob.glob(f"{DS}/x_skab/SKAB-master/data/**/*.csv",
                             recursive=True))
    cols = ["Accelerometer1RMS", "Accelerometer2RMS", "Current", "Pressure",
            "Temperature", "Thermocouple", "Voltage", "Volume Flow RateRMS"]
    Xs, ys, groups = [], [], []
    for gi, f in enumerate(files):
        try:
            with open(f, newline="") as fh:
                head = fh.readline()
            delim = ";" if head.count(";") >= head.count(",") else ","
            with open(f, newline="") as fh:
                rd = list(csv.DictReader(fh, delimiter=delim))
            if not rd:
                continue
            names = set(rd[0].keys())
            if "anomaly" not in names or not set(cols) <= names:
                continue
            Xs.append(np.array([[float(r[c]) for c in cols] for r in rd],
                               dtype=np.float32))
            ys.append(np.array([int(float(r["anomaly"])) for r in rd]))
            groups += [(gi, i) for i in range(len(rd))]
        except Exception:                                          # noqa: BLE001
            continue
    if not Xs:
        raise RuntimeError("SKAB 未找到可用 CSV")
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    return dict(key="skab", name="SKAB", domain="水务 / 水泵机组",
                X=X, y_noul=y, groups=groups, temporal=True,
                note=f"{len(Xs)} 个试验文件（组内连续时序），{X.shape[1]} 维；"
                     f"正例率 {y.mean()*100:.1f}%")


def load_tep():
    """TEP：化工过程，22 个工况文件，每文件 960 个连续采样点。"""
    base = f"{DS}/x_tep/tennessee-eastman-profBraatz-master"
    Xs, ys, groups = [], [], []
    for gi in range(22):
        f = f"{base}/d{gi:02d}_te.dat"
        if not os.path.exists(f):
            continue
        a = np.loadtxt(f)
        if a.ndim == 1:
            a = a[None, :]
        Xs.append(a.astype(np.float32))
        ys.append(np.full(a.shape[0], gi))
        groups += [(gi, i) for i in range(a.shape[0])]
    X = np.vstack(Xs)
    y_c = np.concatenate(ys)
    labels = ["Normal"] + [f"Fault{i}" for i in range(1, 22)]
    return dict(key="tep", name="Tennessee Eastman", domain="化工 / 连续流程",
                X=X, y_noul=(y_c > 0).astype(int), y_choice=y_c,
                choice_labels=labels,
                groups=groups, temporal=True,
                note="22 个工况（1 正常 + 21 故障），每工况 960 点连续采样，52 维")


LOADERS = [load_ai4i, load_steel, load_secom, load_skab, load_tep]
TEMPORAL_LOADERS = [load_secom, load_skab, load_tep]


# ------------------------------------------------------------------ 划分策略
def random_split(n, rng):
    """v1 口径：全样本随机打乱后 60/20/20。"""
    idx = rng.permutation(n)
    return (idx[:int(n * .6)], idx[int(n * .6):int(n * .8)], idx[int(n * .8):])


def temporal_split(groups, ratios=(0.6, 0.2, 0.2)):
    """v2 口径：在**每个时序单元内部**按顺序切分。

    这样训练只见到每个试验/工况的前段，测试见到后段——模拟真实部署
    （用历史数据训练，预测未来），彻底避免同段数据跨集泄漏。
    """
    bucket = defaultdict(list)
    for i, (g, o) in enumerate(groups):
        bucket[g].append((o, i))
    tr, ca, te = [], [], []
    for _, items in bucket.items():
        items.sort()
        idxs = [i for _, i in items]
        m = len(idxs)
        if m < 5:                     # 单元过小则全部给训练
            tr += idxs
            continue
        a, b = int(m * ratios[0]), int(m * (ratios[0] + ratios[1]))
        tr += idxs[:a]
        ca += idxs[a:b]
        te += idxs[b:]
    return (np.array(sorted(tr)), np.array(sorted(ca)), np.array(sorted(te)))


def group_split(groups, ratios=(0.6, 0.2, 0.2)):
    """v2+ 口径：按**整组**划分（前 60% 的试验/工况做训练…）。

    对应「跨试验泛化」——用已完成的试验训练，预测全新一次试验。
    比组内切分更贴近真实部署，但要求每个子集都有正负样本，
    否则训练无正例或测试无法评估（调用方需自行检查）。
    """
    idx_by_group = defaultdict(list)
    for i, (g, _) in enumerate(groups):
        idx_by_group[g].append(i)
    gids = sorted(idx_by_group)
    m = len(gids)
    a, b = int(m * ratios[0]), int(m * (ratios[0] + ratios[1]))
    tr, ca, te = [], [], []
    for j, gid in enumerate(gids):
        bucket = tr if j < a else (ca if j < b else te)
        bucket += sorted(idx_by_group[gid])
    return (np.array(sorted(tr)), np.array(sorted(ca)), np.array(sorted(te)))


def split_usable(split, y):
    """每个非空子集都必须同时含正负样本，否则该划分不可评估。"""
    for s in split:
        if len(s) == 0:
            return False
        if y is not None and len(set(y[s].tolist())) < 2:
            return False
    return True


# ------------------------------------------------------------------ 评测单元
def balance_idx(y, idx, rng):
    """在 idx 内下采样到正负 1:1，返回新的索引数组。

    注意：这是「判别力口径」，会改变部署先验，不能当作真实部署指标。
    v1 报告的主表用的就是这个口径，v2 保留它以做版本间对照。
    """
    pos = idx[y[idx] == 1]
    neg = idx[y[idx] == 0]
    k = min(len(pos), len(neg))
    if k == 0:
        return idx
    sel = np.concatenate([rng.choice(pos, k, replace=False),
                          rng.choice(neg, k, replace=False)])
    rng.shuffle(sel)
    return sel


def run_noul(X, y, split, hidden, calibrate=True, balanced=False,
             seed=SEED, dev=DEVICE):
    """训练 Noul 头并评测。

    balanced=True 时对三个子集各自下采样到 1:1（判别力口径，与 v1 主表一致）；
    balanced=False 时保持划分原样（贴近真实部署的先验）。
    calibrate=False 用于消融（不做温度缩放）。
    seed 控制训练初始化与下采样抽样，供重复实验使用。
    """
    tr, ca, te = split
    if balanced:
        rng = np.random.default_rng(seed + 1)
        tr = balance_idx(y, tr, rng)
        ca = balance_idx(y, ca, rng)
        te = balance_idx(y, te, rng)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    Xs = np.ascontiguousarray((X - mu) / sd, dtype=np.float32)
    m = train_head(Xs[tr], y[tr], 2, hidden, dev, seed=seed)
    z_te = infer(m, Xs[te], dev)
    z_ca = infer(m, Xs[ca], dev)
    T = fit_temp(z_ca[:, 1] - z_ca[:, 0], y[ca], binary=True) if calibrate else 1.0
    p = sigmoid((z_te[:, 1] - z_te[:, 0]) / T)
    yt = y[te]
    pred = (p >= .5).astype(int)
    blk = {
        "n_train": int(len(tr)), "n_calib": int(len(ca)), "n_test": int(len(te)),
        "test_pos_rate": round(float(yt.mean()), 4),
        "temperature": round(float(T), 4),
        "accuracy": round(float(accuracy_score(yt, pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(yt, pred)), 4),
        "precision": round(float(precision_score(yt, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(yt, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(yt, pred, zero_division=0)), 4),
        "mcc": round(float(matthews_corrcoef(yt, pred)), 4)
               if len(set(pred)) > 1 else 0.0,
        "ece": round(ece_binary(p, yt), 4),
        "brier": round(float(brier_score_loss(yt, p)), 4),
    }
    try:
        blk["roc_auc"] = round(float(roc_auc_score(yt, p)), 4)
        blk["pr_auc"] = round(float(average_precision_score(yt, p)), 4)
    except ValueError:
        blk["roc_auc"] = blk["pr_auc"] = None
    conf = np.maximum(p, 1 - p)
    blk["auto_rate_at_conf0.9"] = round(float((conf >= .9).mean()), 4)
    # 关键安全指标：ECE 低不等于可安全阈值路由（对应 0.1.5 核对的第 3 条质疑）
    #   高置信漏放 = 真值是故障，却被以 ≥0.9 置信度判为正常
    n_pos = max(int((yt == 1).sum()), 1)
    n_neg = max(int((yt == 0).sum()), 1)
    blk["high_conf_miss_rate"] = round(
        float(((p <= 0.1) & (yt == 1)).sum()) / n_pos, 4)
    blk["high_conf_false_alarm_rate"] = round(
        float(((p >= 0.9) & (yt == 0)).sum()) / n_neg, 4)
    return blk, p, yt, m, Xs, mu, sd


def run_choice(X, y, split, hidden, dev=DEVICE):
    """多分类头（Steel / TEP），报告宏平均 F1。"""
    tr, ca, te = split
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    Xs = np.ascontiguousarray((X - mu) / sd, dtype=np.float32)
    m = train_head(Xs[tr], y[tr], int(y.max()) + 1, hidden, dev)
    Z = infer(m, Xs[te], dev)
    pred = Z.argmax(1)
    yt = y[te]
    return {
        "n_classes": int(y.max()) + 1,
        "n_test": int(len(te)),
        "accuracy": round(float(accuracy_score(yt, pred)), 4),
        "macro_f1": round(float(f1_score(yt, pred, average="macro",
                                         zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(yt, pred, average="weighted",
                                            zero_division=0)), 4),
    }


def summarize(blocks):
    """多次重复结果 → 供报告直接使用的扁平统计（mean±std）。"""
    def ms(key):
        v = np.array([b[key] for b in blocks if b.get(key) is not None], dtype=float)
        if not len(v):
            return None, None
        return (round(float(v.mean()), 4),
                round(float(v.std(ddof=1)) if len(v) > 1 else 0.0, 4))

    out = {
        "reps": len(blocks),
        "n_test": blocks[0]["n_test"],
        "test_pos_rate": round(float(np.mean([b["test_pos_rate"] for b in blocks])), 4),
        "temperature": round(float(np.mean([b["temperature"] for b in blocks])), 4),
    }
    for name, key in (("bacc", "balanced_accuracy"), ("mcc", "mcc"),
                      ("auc", "roc_auc"), ("pr_auc", "pr_auc"),
                      ("ece", "ece"), ("brier", "brier"),
                      ("precision", "precision"), ("recall", "recall"),
                      ("f1", "f1"), ("accuracy", "accuracy"),
                      ("auto_rate", "auto_rate_at_conf0.9"),
                      ("high_conf_miss", "high_conf_miss_rate"),
                      ("high_conf_false_alarm", "high_conf_false_alarm_rate")):
        m, s = ms(key)
        out[name] = m
        out[name + "_std"] = s
    return out


def threshold_scan(p, yt, thresholds=(0.7, 0.8, 0.9, 0.95)):
    """置信度阈值扫描：自动化率与安全代价。

    直接回应 0.1.5 核对的第 3 条质疑——「低 ECE 不等于可以安全地做阈值路由」：
    只有当高置信自动处置部分的错误率足够低，阈值路由才成立。
    """
    n_pos = max(int((yt == 1).sum()), 1)
    rows = []
    for th in thresholds:
        conf = np.maximum(p, 1 - p)
        auto = conf >= th
        miss = int(((auto) & (p < 0.5) & (yt == 1)).sum())       # 自动判正常但真故障
        falarm = int(((auto) & (p >= 0.5) & (yt == 0)).sum())    # 自动判故障但真正常
        auto_n = int(auto.sum())
        rows.append({
            "threshold": th,
            "automation_rate": round(float(auto.mean()), 4),
            "auto_count": auto_n,
            "miss_count": miss,
            "false_alarm_count": falarm,
            "auto_error_rate": round((miss + falarm) / max(auto_n, 1), 4),
            "fault_miss_rate": round(miss / n_pos, 4),
        })
    return rows


def measure_latency(m, X, n_single=300, rounds=7, dev=DEVICE):
    """两种延迟口径，显式区分（M4）。"""
    x1 = np.ascontiguousarray(X[:1])
    for _ in range(30):
        infer(m, x1, dev)
    singles = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(n_single):
            infer(m, x1, dev)
        singles.append((time.perf_counter() - t0) / n_single * 1000.0)
    single_ms = float(np.median(singles))

    xb = np.ascontiguousarray(X[:min(4096, len(X))])
    for _ in range(5):
        infer(m, xb, dev)
    t0 = time.perf_counter()
    for _ in range(5):
        infer(m, xb, dev)
    batch_ms = (time.perf_counter() - t0) / 5 * 1000.0
    return {
        "single_inference_ms": round(single_ms, 5),
        "single_throughput_per_s": int(1000.0 / single_ms) if single_ms else None,
        "batch_size": int(len(xb)),
        "batch_total_ms": round(batch_ms, 3),
        "batch_amortized_ms": round(batch_ms / len(xb), 6),
        "batch_throughput_per_s": int(len(xb) / batch_ms * 1000),
        "note": "single_inference_ms 为单条端到端（含张量构造）；"
                "batch_amortized_ms 为批内摊薄值，两者不可跨口径比较",
    }


# ------------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="只跑主对比，跳过容量与消融")
    ap.add_argument("--repeats", type=int, default=5,
                    help="主对比的多种子重复次数（默认 5）")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    print(f"设备 {DEVICE} · PyTorch {torch.__version__} · 种子 {SEED}", flush=True)
    result = {"device": DEVICE, "torch": torch.__version__, "seed": SEED,
              "epochs": EPOCHS, "datasets": {}, "generated_at": time.strftime("%F %T")}

    for loader in LOADERS:
        ds = loader()
        key = ds["key"]
        print(f"\n===== {ds['name']}（{ds['domain']}）=====", flush=True)
        print(f"  {ds['note']}", flush=True)
        rng = np.random.default_rng(SEED)
        rec = {"name": ds["name"], "domain": ds["domain"], "note": ds["note"],
               "n": int(len(ds["X"])), "dim": int(ds["X"].shape[1]),
               "temporal": bool(ds["temporal"]),
               "noul_random_split": None, "noul_temporal_split": None,
               "noul_cross_unit_split": None}

        n = len(ds["X"])
        rand = random_split(n, rng)
        temp = temporal_split(ds["groups"]) if ds["groups"] else None

        # ---- M1 主对比：随机划分 / 时序划分 / 跨单元划分
        # 每个划分都产出两个口径：
        #   自然分布 = 贴近真实部署的先验
        #   均衡分布 = 判别力口径（v1 主表用的就是这个，用于版本对照）
        # 多种子重复取 mean±std —— 部分数据集的均衡测试集很小，单次结果不足为据
        if ds["y_noul"] is not None and len(set(ds["y_noul"])) > 1:
            y = ds["y_noul"]
            gs = group_split(ds["groups"]) if ds["groups"] else None
            gs_ok = gs is not None and split_usable(gs, y)
            buckets = {k: [] for k in ("random_natural", "random_balanced",
                                       "temporal_natural", "temporal_balanced",
                                       "cross_unit_natural", "cross_unit_balanced")}
            for rep in range(args.repeats):
                s_rep = SEED + rep * 1000
                split_rep = (random_split(len(ds["X"]),
                                          np.random.default_rng(s_rep))
                             if rep else rand)
                b, p, yt, m, Xs, _, _ = run_noul(
                    ds["X"], y, split_rep, CAPACITIES["small"], seed=s_rep)
                buckets["random_natural"].append(b)
                buckets["random_balanced"].append(run_noul(
                    ds["X"], y, split_rep, CAPACITIES["small"],
                    balanced=True, seed=s_rep)[0])
                if rep == 0:
                    m_r, Xs_r = m, Xs
                if temp is not None:
                    buckets["temporal_natural"].append(run_noul(
                        ds["X"], y, temp, CAPACITIES["small"], seed=s_rep)[0])
                    buckets["temporal_balanced"].append(run_noul(
                        ds["X"], y, temp, CAPACITIES["small"],
                        balanced=True, seed=s_rep)[0])
                    if gs_ok:
                        buckets["cross_unit_natural"].append(run_noul(
                            ds["X"], y, gs, CAPACITIES["small"], seed=s_rep)[0])
                        buckets["cross_unit_balanced"].append(run_noul(
                            ds["X"], y, gs, CAPACITIES["small"],
                            balanced=True, seed=s_rep)[0])

            rec["repeats"] = args.repeats
            rec["main"] = {k: summarize(v) for k, v in buckets.items() if v}
            for label, keyn in (("随机·自然", "random_natural"),
                                ("随机·均衡", "random_balanced"),
                                ("时序·自然", "temporal_natural"),
                                ("时序·均衡", "temporal_balanced"),
                                ("跨单元·自然", "cross_unit_natural"),
                                ("跨单元·均衡", "cross_unit_balanced")):
                s = rec["main"].get(keyn)
                if not s:
                    continue
                print(f"  [{label:<12}] BAcc {s['bacc']:.4f}±{s['bacc_std']:.4f}  "
                      f"MCC {s['mcc']:+.4f}  AUC {s['auc']}  ECE {s['ece']:.4f}  "
                      f"(n_test={s['n_test']}, 正例率 {s['test_pos_rate']:.3f})",
                      flush=True)
        else:
            rec["main"] = None

        # ---- M2 容量对照（时序划分优先；均衡口径，以便与 v1 主表对照）
        split_cap = temp if temp is not None else rand
        if not args.quick and ds["y_noul"] is not None \
                and len(set(ds["y_noul"])) > 1:
            rec["capacity_sweep"] = {}
            for cname, hidden in CAPACITIES.items():
                b, _, _, _, _, _, _ = run_noul(
                    ds["X"], ds["y_noul"], split_cap, hidden, balanced=True)
                rec["capacity_sweep"][cname] = b
                print(f"  [容量 {cname:<6}{str(hidden):<18}] "
                      f"BAcc {b['balanced_accuracy']:.4f}  "
                      f"MCC {b['mcc']:+.4f}  AUC {b['roc_auc']}", flush=True)

        # ---- M3 消融（容量 small；均衡口径）
        if not args.quick and ds["y_noul"] is not None \
                and len(set(ds["y_noul"])) > 1:
            rec["ablation"] = {}
            full, p_full, yt, m_full, Xs, mu, sd = run_noul(
                ds["X"], ds["y_noul"], split_cap, CAPACITIES["small"],
                calibrate=True, balanced=True, seed=SEED)
            rec["ablation"]["full_calibrated"] = full
            noc, p_noc, _, _, _, _, _ = run_noul(
                ds["X"], ds["y_noul"], split_cap, CAPACITIES["small"],
                calibrate=False, balanced=True, seed=SEED)
            rec["ablation"]["no_temperature_scaling"] = noc
            print(f"  [消融] 完整版 ECE {full['ece']:.4f} / Brier {full['brier']:.4f}"
                  f"  →  无温度缩放 ECE {noc['ece']:.4f} / Brier {noc['brier']:.4f}"
                  f"  (T={full['temperature']:.3f})", flush=True)

            # ---- 阈值扫描（回应第 3 条质疑：低 ECE ≠ 可安全阈值路由）
            rec["threshold_scan"] = threshold_scan(p_full, yt)
            print("  [阈值扫描] " + "  ".join(
                f"th={r['threshold']}: 自动 {r['automation_rate']*100:.1f}% / "
                f"漏放 {r['fault_miss_rate']*100:.1f}% / "
                f"自动内错误 {r['auto_error_rate']*100:.1f}%"
                for r in rec["threshold_scan"]), flush=True)

        # ---- 多分类（Steel / TEP）
        if ds.get("y_choice") is not None:
            split_c = temp if temp is not None else rand
            rec["choice"] = run_choice(ds["X"], ds["y_choice"], split_c,
                                       CAPACITIES["small"])
            c = rec["choice"]
            print(f"  [多分类 {c['n_classes']} 类] Acc {c['accuracy']:.4f}  "
                  f"MacroF1 {c['macro_f1']:.4f}  WeightedF1 {c['weighted_f1']:.4f}",
                  flush=True)

        # ---- M4 延迟（两口径）
        if ds["y_noul"] is not None and len(set(ds["y_noul"])) > 1:
            rec["latency"] = measure_latency(m_r, Xs_r)
            print(f"  [延迟] 单条 {rec['latency']['single_inference_ms']:.5f} ms  |  "
                  f"批量({rec['latency']['batch_size']}) 摊薄 "
                  f"{rec['latency']['batch_amortized_ms']:.6f} ms", flush=True)

        result["datasets"][key] = rec

    path = os.path.join(OUT, "multi_dataset_v2.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(f"\n结果已写入 {path}", flush=True)


if __name__ == "__main__":
    main()
