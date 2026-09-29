#!/usr/bin/env python3
"""哨兵 EdgeSentinel —— System One 决策层原型（D1）

把结构化设备状态编码为向量，用三个类型化决策头（Noul / Choice / Score）
一次性输出全部概率分布，再做温度缩放校准，并给出：
  1. 校准前后的可靠性曲线与 ECE / Brier
  2. 阈值—自动化率—准确率权衡表（核心卖点）
  3. 与生成式决策（实测 83.7s）的延迟对比

依赖：numpy + scikit-learn（无需 GPU、无需外网）
"""
from __future__ import annotations

import csv
import math
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

DATA = "/home/USER/jev-service/data/ai4i2020.csv"
RNG = np.random.default_rng(20260921)

# 生成式方案实测基线（本地 27B，2026-09-21 公网实测）
GENERATIVE_LATENCY_S = 83.7

TYPES = ["No Failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
SEVERITY_LABELS = ["观察", "计划", "尽快", "立即"]


# ---------------------------------------------------------------- 数据加载
def load_rows(path: str) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def to_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def encode(row: dict[str, str]) -> np.ndarray:
    """状态编码器：原始传感器读数 -> 特征向量（对应架构里的 encoder）。"""
    air = to_float(row, "Air temperature [K]")
    proc = to_float(row, "Process temperature [K]")
    rpm = to_float(row, "Rotational speed [rpm]")
    torque = to_float(row, "Torque [Nm]")
    wear = to_float(row, "Tool wear [min]")
    return np.array([
        air, proc, rpm, torque, wear,
        proc - air,                 # 温升
        rpm * torque * 2 * math.pi / 60.0 / 1000.0,  # 机械功率 kW
        wear * torque / 100.0,      # 磨损-负载耦合
        (proc - air) / max(rpm / 1000.0, 1e-6),      # 单位转速温升
    ], dtype=np.float64)


def failure_type(row: dict[str, str]) -> str:
    for key in ("TWF", "HDF", "PWF", "OSF", "RNF"):
        if int(row[key]) == 1:
            return key
    return "No Failure"


def severity(row: dict[str, str]) -> int:
    """维护优先级评分 0~3（派生标签；正式版由领域专家或 StepFun 标注）。

    0 观察 / 1 计划 / 2 尽快 / 3 立即 —— 依据磨损量、温升、负载三项工业常用判据。
    """
    air = to_float(row, "Air temperature [K]")
    proc = to_float(row, "Process temperature [K]")
    torque = to_float(row, "Torque [Nm]")
    wear = to_float(row, "Tool wear [min]")
    diff = proc - air
    if wear >= 200 or (diff >= 12 and torque >= 50):
        return 3
    if wear >= 120 or diff >= 11:
        return 2
    if wear >= 50 or diff >= 9:
        return 1
    return 0


# ---------------------------------------------------------------- 校准工具
def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def fit_temperature_binary(z: np.ndarray, y: np.ndarray) -> float:
    best_t, best_nll = 1.0, float("inf")
    for t in np.linspace(0.1, 5.0, 200):
        p = 1.0 / (1.0 + np.exp(-z / t))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        nll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        if nll < best_nll:
            best_t, best_nll = float(t), float(nll)
    return best_t


def fit_temperature_multi(Z: np.ndarray, y: np.ndarray) -> float:
    best_t, best_nll = 1.0, float("inf")
    for t in np.linspace(0.1, 5.0, 200):
        p = softmax(Z / t)
        nll = -np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1.0)))
        if nll < best_nll:
            best_t, best_nll = float(t), float(nll)
    return best_t


def ece_binary(p: np.ndarray, y: np.ndarray, bins: int = 10) -> tuple[float, list]:
    idx = np.clip((p * bins).astype(int), 0, bins - 1)
    total, rows = 0.0, []
    for b in range(bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        conf, acc = float(p[mask].mean()), float(y[mask].mean())
        total += n / len(p) * abs(acc - conf)
        rows.append((round(conf, 3), round(acc, 3), n))
    return float(total), rows


def brier_binary(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def main() -> None:
    print("=" * 66)
    print("哨兵 EdgeSentinel · System One 决策层原型")
    print("=" * 66)

    rows = load_rows(DATA)
    print(f"\n[数据] {len(rows)} 条设备记录，来自 AI4I 2020")

    X = np.vstack([encode(r) for r in rows])
    y_noul = np.array([int(r["Machine failure"]) for r in rows])
    types = ["No Failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
    y_choice = np.array([types.index(failure_type(r)) for r in rows])
    y_score = np.array([severity(r) for r in rows])

    print(f"[标签] 是否故障: 正样本 {int(y_noul.sum())} / {len(y_noul)}"
          f"  ({100 * y_noul.mean():.1f}%)")
    print(f"[标签] 故障类型: { {t: int((y_choice == i).sum()) for i, t in enumerate(types)} }")
    print(f"[标签] 严重度分布: { {s: int((y_score == s).sum()) for s in range(4)} }")

    # 划分：训练 60% / 校准 20% / 测试 20%
    n = len(rows)
    perm = RNG.permutation(n)
    i1, i2 = int(n * 0.6), int(n * 0.8)
    tr, ca, te = perm[:i1], perm[i1:i2], perm[i2:]

    scaler = StandardScaler().fit(X[tr])
    Xtr, Xca, Xte = scaler.transform(X[tr]), scaler.transform(X[ca]), scaler.transform(X[te])

    print("\n[训练] 三个决策头（MLP，输入为编码后的 9 维状态向量）")
    head_noul = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800,
                              random_state=20260921).fit(Xtr, y_noul[tr])
    head_choice = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800,
                                random_state=20260921).fit(Xtr, y_choice[tr])
    head_score = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800,
                               random_state=20260921).fit(Xtr, y_score[tr])
    ref_noul = LogisticRegression(max_iter=2000).fit(Xtr, y_noul[tr])
    print("  完成（主模型 MLP，另训练逻辑回归作对照）")

    # ---------------- 校准前 ----------------
    p_raw = head_noul.predict_proba(Xte)[:, 1]
    z_te = np.log(np.clip(p_raw, 1e-9, 1 - 1e-9) / np.clip(1 - p_raw, 1e-9, 1.0))
    ece_raw, _ = ece_binary(p_raw, y_noul[te])
    brier_raw = brier_binary(p_raw, y_noul[te])

    # ---------------- 温度缩放校准 ----------------
    p_ca_raw = head_noul.predict_proba(Xca)[:, 1]
    z_ca = np.log(np.clip(p_ca_raw, 1e-9, 1 - 1e-9) / np.clip(1 - p_ca_raw, 1e-9, 1.0))
    T_noul = fit_temperature_binary(z_ca, y_noul[ca])
    Z_ca_c = head_choice.predict_log_proba(Xca)
    T_choice = fit_temperature_multi(Z_ca_c, y_choice[ca])
    Z_ca_s = head_score.predict_log_proba(Xca)
    T_score = fit_temperature_multi(Z_ca_s, y_score[ca])

    p_cal = 1.0 / (1.0 + np.exp(-z_te / T_noul))
    ece_cal, curve = ece_binary(p_cal, y_noul[te])
    brier_cal = brier_binary(p_cal, y_noul[te])

    # ---------------- 准确率 ----------------
    acc_noul = float(((p_cal >= 0.5).astype(int) == y_noul[te]).mean())
    pc = softmax(head_choice.predict_log_proba(Xte) / T_choice)
    acc_choice = float((pc.argmax(1) == y_choice[te]).mean())
    ps = softmax(head_score.predict_log_proba(Xte) / T_score)
    acc_score = float((ps.argmax(1) == y_score[te]).mean())
    recalls = []
    for k in range(len(types)):
        mask = y_choice[te] == k
        if mask.sum():
            recalls.append(float((pc[mask].argmax(1) == k).mean()))
    macro_recall = float(np.mean(recalls)) if recalls else 0.0

    print("\n" + "=" * 66)
    print("一、校准效果（Noul：是否真实故障）")
    print("=" * 66)
    print(f"  温度系数 T = {T_noul:.3f}   (Choice T = {T_choice:.3f}, Score T = {T_score:.3f})")
    print(f"  ECE   校准前 {ece_raw:.4f}  ->  校准后 {ece_cal:.4f}")
    print(f"  Brier 校准前 {brier_raw:.4f}  ->  校准后 {brier_cal:.4f}")
    print("\n  校准后可靠性曲线（预测概率 vs 实际发生率）：")
    print("    预测   实际   样本数")
    for conf, acc, cnt in curve:
        bar = "#" * int(round(acc * 30))
        print(f"    {conf:<6.3f} {acc:<6.3f} {cnt:<6d} {bar}")

    print("\n" + "=" * 66)
    print(f"二、三种决策原语准确率（测试集 20%，n={len(te)}）")
    print("=" * 66)
    print(f"  Noul   是否故障      {acc_noul * 100:5.1f}%")
    print(f"  Choice 故障类型(6类) {acc_choice * 100:5.1f}%   （宏平均召回 {macro_recall * 100:.1f}%，类别不平衡看这个更准）")
    print(f"  Score  严重度(0~3)   {acc_score * 100:5.1f}%")

    # ---------------- 阈值权衡表 ----------------
    print("\n" + "=" * 66)
    print("三、阈值 — 自动化率 — 准确率权衡表（核心卖点）")
    print("=" * 66)
    print("  阈值   自动处理占比   自动部分准确率   人工复核占比   自动部分错误率")
    for thr in (0.60, 0.70, 0.80, 0.90, 0.95, 0.98):
        conf = np.maximum(p_cal, 1 - p_cal)
        auto = conf >= thr
        rate = float(auto.mean())
        if auto.sum() == 0:
            print(f"  {thr:<6.2f} {rate * 100:11.1f}%   (无样本满足)")
            continue
        acc = float(((p_cal[auto] >= 0.5).astype(int) == y_noul[te][auto]).mean())
        print(f"  {thr:<6.2f} {rate * 100:11.1f}% {acc * 100:14.1f}% "
              f"{(1 - rate) * 100:12.1f}% {(1 - acc) * 100:13.1f}%")

    # ---------------- 延迟 ----------------
    print("\n" + "=" * 66)
    print("四、延迟对比")
    print("=" * 66)
    sample = Xte[:500]
    t0 = time.perf_counter()
    for i in range(len(sample)):
        _ = head_noul.predict_proba(sample[i:i + 1])
        _ = head_choice.predict_proba(sample[i:i + 1])
        _ = head_score.predict_proba(sample[i:i + 1])
    dt = (time.perf_counter() - t0) / len(sample)
    print(f"  System One 决策层（一次前向，三问同出）: {dt * 1000:.3f} ms/条")
    print(f"  生成式决策（本地 27B，实测）            : {GENERATIVE_LATENCY_S * 1000:.0f} ms/条")
    print(f"  加速比                                  : {GENERATIVE_LATENCY_S / dt:.0f}x")
    print(f"  吞吐                                    : {1 / dt:.0f} 条/秒")

    print("\n" + "=" * 66)
    print("D1 完成：编码器 + 三决策头 + 校准链路全部跑通")
    print("=" * 66)


if __name__ == "__main__":
    main()
