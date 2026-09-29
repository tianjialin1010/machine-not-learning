#!/usr/bin/env python3
"""Jev 架构 · 多工业数据集综合能力评测。

目标
----
在多个工业子领域上，用统一的 Jev 类型化决策框架（Noul / Choice / Score）
评测同一套决策层，得出「Jev 架构在工业场景的综合能力画像」。

数据集（5 个工业子领域）
-----------------------
  AI4I 2020      机加工        10000 样本 · Noul + Choice(5)
  Steel Plates   冶金          1941 样本  · Choice(7)  ← 原始数据无正常类
  SECOM          半导体制造    1566 样本 · Noul（590 维，缺失值多）
  SKAB           水务/水泵泵   35 文件时序 · Noul
  TEP            化工过程      22×960 样本 · Noul + Choice(21)

统一性保证
----------
  · 同一套状态编码流程（标准化 → 特征向量）
  · 同一个决策头结构 MLP(64,32) + 温度缩放校准
  · 同一划分比例 60/20/20、同一种子
  · 同一套指标：Noul(Acc/BAcc/P/R/F1/MCC/AUC/ECE/Brier) + Choice(Acc/MacroF1) + 延迟

用法：python3 multi_dataset_bench.py [输出目录]
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score, average_precision_score)

ROOT = os.path.expanduser("~/jev-service")
AI4I = f"{ROOT}/data/ai4i2020.csv"
DS = f"{ROOT}/ind_ds"
SEED = 20260921
EPOCHS = 600
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/multi_bench"
os.makedirs(OUT_DIR, exist_ok=True)


# ==================================================================== 工具
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


def brier(p, y):
    return float(np.mean((np.clip(p, 0, 1) - y) ** 2))


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


def train_head(Xtr, ytr, out_dim, dev):
    torch.manual_seed(SEED)
    m = Head(Xtr.shape[1], out_dim).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=250, gamma=0.5)
    lossf = nn.CrossEntropyLoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=dev)
    yt = torch.tensor(ytr, dtype=torch.long, device=dev)
    m.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        lossf(m(Xt), yt).backward()
        opt.step()
        sch.step()
    m.eval()
    return m


def infer(m, X, dev):
    with torch.no_grad():
        return m(torch.tensor(np.ascontiguousarray(X, dtype=np.float32),
                              device=dev)).cpu().numpy()


# ==================================================================== 数据集
def load_ai4i():
    rows = list(csv.DictReader(open(AI4I, encoding="utf-8-sig")))
    tl = ["No Failure", "TWF", "HDF", "PWF", "OSF", "RNF"]

    def enc(r):
        air, proc = float(r["Air temperature [K]"]), float(r["Process temperature [K]"])
        rpm, tq = float(r["Rotational speed [rpm]"]), float(r["Torque [Nm]"])
        w = float(r["Tool wear [min]"])
        return [air, proc, rpm, tq, w, proc - air,
                rpm * tq * 2 * math.pi / 60 / 1000, w * tq / 100,
                (proc - air) / max(rpm / 1000, 1e-6)]

    X = np.array([enc(r) for r in rows], dtype=np.float64)
    y_n = np.array([int(r["Machine failure"]) for r in rows])
    y_c = np.array([tl.index(next((k for k in tl[1:] if int(r[k]) == 1), "No Failure"))
                    for r in rows])
    return dict(key="ai4i", name="AI4I 2020", domain="机加工 / 数控机床",
                X=X, y_noul=y_n, y_choice=y_c, choice_labels=tl,
                note="故障标签 TWF/HDF/PWF/OSF/RNF")


def load_steel():
    arr = np.loadtxt(f"{DS}/x_steel/Faults.NNA")
    X = arr[:, :27]
    Y = arr[:, 27:34]
    labels = ["Pastry", "Z_Scratch", "K_Scatch", "Stains",
              "Dirtiness", "Bumps", "Other_Faults"]
    y_c = Y.argmax(1)
    y_n = (Y.sum(1) > 0).astype(int)
    return dict(key="steel", name="Steel Plates Faults", domain="冶金 / 钢板表面缺陷",
                X=X, y_noul=y_n, y_choice=y_c, choice_labels=labels,
                note="原始数据全部为缺陷样本，无正常类 → 只评测 Choice")


def load_secom():
    raw = np.genfromtxt(f"{DS}/x_secom/secom.data", dtype=float)   # NaN = 缺失
    lab = np.genfromtxt(f"{DS}/x_secom/secom_labels.data",
                        dtype=str, usecols=0)
    y = (lab.astype(float) > 0).astype(int)
    col_med = np.nanmedian(raw, axis=0)
    col_med = np.where(np.isnan(col_med), 0.0, col_med)
    idx = np.where(np.isnan(raw))
    raw[idx] = np.take(col_med, idx[1])
    # 去掉常数方差列
    keep = raw.std(0) > 0
    return dict(key="secom", name="SECOM", domain="半导体 / 晶圆制造",
                X=raw[:, keep], y_noul=y, y_choice=None, choice_labels=None,
                note=f"590 维传感器，缺失值以列中位数填充，去常数后保留 "
                     f"{int(keep.sum())} 维；正例率 {y.mean()*100:.1f}%")


def load_skab():
    files = sorted(glob.glob(f"{DS}/x_skab/SKAB-master/data/**/*.csv",
                             recursive=True))
    cols = ["Accelerometer1RMS", "Accelerometer2RMS", "Current", "Pressure",
            "Temperature", "Thermocouple", "Voltage", "Volume Flow RateRMS"]
    Xs, ys = [], []
    for f in files:
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
            Xs.append(np.array([[float(r[c]) for c in cols] for r in rd]))
            ys.append(np.array([int(float(r["anomaly"])) for r in rd]))
        except Exception:                                         # noqa: BLE001
            continue
    if not Xs:
        raise RuntimeError("SKAB 未找到可用 CSV")
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    return dict(key="skab", name="SKAB", domain="水务 / 水泵机组",
                X=X, y_noul=y, y_choice=None, choice_labels=None,
                files=len(Xs),
                note=f"{len(files)} 个 CSV 合并，8 维振动/电流/压力/温度；"
                     f"正例率 {y.mean()*100:.1f}%")


def load_tep():
    base = f"{DS}/x_tep/tennessee-eastman-profBraatz-master"
    Xs, ys = [], []
    for i in range(22):
        f = f"{base}/d{i:02d}_te.dat"
        if not os.path.exists(f):
            continue
        a = np.loadtxt(f)
        if a.ndim == 1:
            a = a[None, :]
        Xs.append(a)
        ys.append(np.full(a.shape[0], 0 if i == 0 else i))
    X = np.vstack(Xs)
    y_c = np.concatenate(ys)
    labels = ["Normal"] + [f"Fault{i}" for i in range(1, 22)]
    return dict(key="tep", name="Tennessee Eastman", domain="化工 / 连续流程",
                X=X, y_noul=(y_c > 0).astype(int), y_choice=y_c,
                choice_labels=labels,
                note="22 个工况（1 正常 + 21 故障），每工况 960 个采样点，52 维过程变量")


LOADERS = [load_ai4i, load_steel, load_secom, load_skab, load_tep]


# ==================================================================== 评测
def balance(X, y, rng):
    """下采样到正负 1:1。"""
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    k = min(len(pos), len(neg))
    if k == 0:
        return X, y
    sel = np.concatenate([rng.choice(pos, k, replace=False),
                          rng.choice(neg, k, replace=False)])
    rng.shuffle(sel)
    return X[sel], y[sel]


def noul_block(Xs, y, tr, ca, te, dev):
    """训练 + 温度缩放 + 指标，返回 (block, 预测概率)。"""
    m = train_head(Xs[tr], y[tr], 2, dev)
    z = infer(m, Xs[te], dev)
    z_ca = infer(m, Xs[ca], dev)
    T = fit_temp(z_ca[:, 1] - z_ca[:, 0], y[ca], binary=True)
    p = sigmoid((z[:, 1] - z[:, 0]) / T)
    pred = (p >= .5).astype(int)
    yt = y[te]
    blk = {
        "temperature": round(T, 4),
        "accuracy": round(float(accuracy_score(yt, pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(yt, pred)), 4),
        "precision": round(float(precision_score(yt, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(yt, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(yt, pred, zero_division=0)), 4),
        "mcc": round(float(matthews_corrcoef(yt, pred)), 4)
               if len(set(pred)) > 1 else 0.0,
        "ece": round(ece_binary(p, yt), 4),
        "brier": round(brier(p, yt), 4),
    }
    try:
        blk["roc_auc"] = round(float(roc_auc_score(yt, p)), 4)
        blk["pr_auc"] = round(float(average_precision_score(yt, p)), 4)
    except ValueError:
        blk["roc_auc"] = blk["pr_auc"] = None
    conf = np.maximum(p, 1 - p)
    blk["auto_rate_at_conf0.9"] = round(float((conf >= .9).mean()), 4)
    return blk


def evaluate(ds, dev):
    X, y_n, y_c = ds["X"], ds["y_noul"], ds["y_choice"]
    n = len(X)
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(n)
    tr, ca, te = idx[:int(n * .6)], idx[int(n * .6):int(n * .8)], idx[int(n * .8):]

    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    Xs = np.ascontiguousarray((X - mu) / sd, dtype=np.float32)

    res = {"key": ds["key"], "name": ds["name"], "domain": ds["domain"],
           "note": ds.get("note", ""), "n_total": n,
           "n_train": len(tr), "n_calib": len(ca), "n_test": len(te),
           "n_features": int(X.shape[1]),
           "fault_rate": round(float(y_n.mean()), 4)}

    # ---------------- Noul（自然分布 + 均衡分布两组口径）
    if len(set(y_n[tr])) > 1:
        res["noul_natural"] = noul_block(Xs, y_n, tr, ca, te, dev)
        # 均衡口径：训练与测试都下采样到 1:1，反映「判别能力」
        rng_b = np.random.default_rng(SEED + 1)
        Xb_tr, yb_tr = balance(X[tr], y_n[tr], rng_b)
        Xb_ca, yb_ca = balance(X[ca], y_n[ca], rng_b)
        Xb_te, yb_te = balance(X[te], y_n[te], rng_b)
        mu_b, sd_b = Xb_tr.mean(0), Xb_tr.std(0) + 1e-9
        sb_tr = ((Xb_tr - mu_b) / sd_b).astype(np.float32)
        sb_ca = ((Xb_ca - mu_b) / sd_b).astype(np.float32)
        sb_te = ((Xb_te - mu_b) / sd_b).astype(np.float32)
        ntr = np.arange(len(sb_tr))
        nca = np.arange(len(sb_tr), len(sb_tr) + len(sb_ca))
        nte = np.arange(len(sb_tr) + len(sb_ca), len(sb_tr) + len(sb_ca) + len(sb_te))
        Xall_b = np.vstack([sb_tr, sb_ca, sb_te])
        yall_b = np.concatenate([yb_tr, yb_ca, yb_te])
        res["noul_balanced"] = noul_block(Xall_b, yall_b, ntr, nca, nte, dev)
        res["noul_balanced"]["n_test"] = int(len(nte))
    else:
        res["noul_natural"] = res["noul_balanced"] = None

    # ---------------- Choice
    if y_c is not None and len(set(y_c)) > 1:
        mc = train_head(Xs[tr], y_c[tr], int(y_c.max()) + 1, dev)
        zc = infer(mc, Xs[te], dev)
        zc_ca = infer(mc, Xs[ca], dev)
        Tc = fit_temp(zc_ca, y_c[ca], binary=False)
        Pc = softmax(zc / Tc)
        predc = Pc.argmax(1)
        res["choice"] = {
            "n_classes": int(y_c.max()) + 1,
            "accuracy": round(float(accuracy_score(y_c[te], predc)), 4),
            "macro_f1": round(float(f1_score(y_c[te], predc, average="macro",
                                             zero_division=0)), 4),
            "weighted_f1": round(float(f1_score(y_c[te], predc, average="weighted",
                                                zero_division=0)), 4),
            "temperature": round(Tc, 4),
        }
    else:
        res["choice"] = None

    # ---------------- 延迟（单条，中位数）
    m_last = train_head(Xs[tr], y_n[tr], 2, dev)
    t0 = time.perf_counter()
    for _ in range(3):
        infer(mc if res.get("choice") else m_last, Xs[:200], dev)
    dt = (time.perf_counter() - t0) / 3 / 200 * 1000
    res["latency_ms_per_item"] = round(dt, 5)
    res["latency_note"] = f"batch=200 摊薄；测试集 n={len(te)}"
    return res


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 92)
    print("Jev 架构 · 多工业数据集综合能力评测")
    print("=" * 92)
    print(f"  设备: {dev}  |  输出: {OUT_DIR}\n")

    out = []
    for ld in LOADERS:
        try:
            ds = ld()
        except Exception as exc:                                  # noqa: BLE001
            print(f"  [跳过] {ld.__name__}: {type(exc).__name__}: {exc}")
            continue
        print(f"  ▸ {ds['name']:<24} {ds['domain']:<20} "
              f"n={ds['X'].shape[0]:<6} d={ds['X'].shape[1]:<4}", flush=True)
        t0 = time.perf_counter()
        try:
            r = evaluate(ds, dev)
        except Exception as exc:                                  # noqa: BLE001
            print(f"      [失败] {type(exc).__name__}: {exc}")
            out.append({"key": ds["key"], "name": ds["name"],
                        "domain": ds["domain"], "error": str(exc)})
            continue
        r["elapsed_s"] = round(time.perf_counter() - t0, 1)
        out.append(r)
        for tag, keyname in [("自然分布", "noul_natural"), ("均衡分布", "noul_balanced")]:
            b = r.get(keyname)
            if not b:
                continue
            print(f"      Noul[{tag}]  BAcc {b['balanced_accuracy']:.4f}  "
                  f"P {b['precision']:.4f}  R {b['recall']:.4f}  F1 {b['f1']:.4f}  "
                  f"MCC {b['mcc']:.4f}  AUC {b['roc_auc']}  ECE {b['ece']}")
        if r.get("choice"):
            c = r["choice"]
            print(f"      Choice({c['n_classes']}类) Acc {c['accuracy']:.4f}  "
                  f"MacroF1 {c['macro_f1']:.4f}")
        print(f"      延迟 {r['latency_ms_per_item']} ms/条  |  用时 {r['elapsed_s']}s")

    path = os.path.join(OUT_DIR, "multi_dataset_result.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"\n已写出 {path}")
    print("=" * 92)


if __name__ == "__main__":
    main()
