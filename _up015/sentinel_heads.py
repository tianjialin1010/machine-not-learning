#!/usr/bin/env python3
"""哨兵决策头（System One 的模型层）。

职责边界：只做「结构化传感器数值 → 概率分布」。不做路由、不做策略、不做校准策略
——那些交给 jev 框架（calibration.py / policy.py）。这样职责单一，两端可独立演进。

对外契约（供 evaluator 调用）：
    heads = SentinelHeads.load_or_train(data_csv, device)
    heads.predict({"air_temp_c": 24.1, ...})
      -> {"p_fault": float, "type_probs": {label: p}, "sev_probs": {label: p}}

三个决策头：
    noul  : 是否真实故障        （AI4I 原生标签 Machine failure）
    type  : 故障类型 6 类       （AI4I 原生标签 Failure Type）
    sev   : 维护优先级 4 档     （派生标签，见 severity()；provenance = derived_rule）

校准：每个头单独拟合温度 T（在标定集上最小化 NLL），推理时应用。
注意：先验修正（deployment prior）**不在这里做**，交给 jev 的 apply_profile，
      因为它能拿到 channel_metadata.deployment_positive_rate 这个运行时信息。
"""
from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

SEED = 20260921
RAW_KEYS = ("air_temp_c", "process_temp_c", "rpm", "torque_nm", "tool_wear_min")
TYPES = ["No Failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
FAULT_TYPES = TYPES[1:]
SEVERITY_LABELS = ["观察", "计划", "尽快", "立即"]
FEATURE_NAMES = [
    "air_temp_c", "process_temp_c", "rpm", "torque_nm", "tool_wear_min",
    "temp_diff", "power_w", "wear_torque", "temp_diff_per_krpm",
]


# ------------------------------------------------------------------ 特征工程
def engineer(raw: dict[str, Any]) -> list[float]:
    """5 路原始传感器 → 9 维特征。对外契约只需要 5 路，其余由物理量派生。"""
    air = float(raw["air_temp_c"])
    proc = float(raw["process_temp_c"])
    rpm = float(raw["rpm"])
    torque = float(raw["torque_nm"])
    wear = float(raw["tool_wear_min"])
    diff = proc - air
    power = torque * 2.0 * math.pi * rpm / 60.0
    wear_torque = wear * torque
    diff_per_krpm = diff / max(rpm / 1000.0, 1e-6)
    return [air, proc, rpm, torque, wear, diff, power, wear_torque, diff_per_krpm]


def severity(raw: dict[str, Any]) -> int:
    """维护优先级 0~3（派生标签，provenance = derived_rule）。

    判据依据工业常用的三项：刀具磨损量、温升、负载。
    """
    diff = float(raw["process_temp_c"]) - float(raw["air_temp_c"])
    torque = float(raw["torque_nm"])
    wear = float(raw["tool_wear_min"])
    if wear >= 200 or (diff >= 12 and torque >= 50):
        return 3
    if wear >= 120 or diff >= 11:
        return 2
    if wear >= 50 or diff >= 9:
        return 1
    return 0


def sensor_plausibility(raw: dict[str, Any]) -> int:
    """传感器可信度检查：返回不合物理常识的读数个数。

    用于 needs_clarification（读数可疑 → 先让现场确认，别急着下结论）。
    """
    bad = 0
    air = float(raw["air_temp_c"])
    proc = float(raw["process_temp_c"])
    rpm = float(raw["rpm"])
    torque = float(raw["torque_nm"])
    wear = float(raw["tool_wear_min"])
    if not (-20.0 <= air <= 80.0):
        bad += 1
    if not (air <= proc <= 150.0):
        bad += 1
    if not (0.0 < rpm <= 4000.0):
        bad += 1
    if not (0.0 <= torque <= 200.0):
        bad += 1
    if not (0.0 <= wear <= 300.0):
        bad += 1
    return bad


# ------------------------------------------------------------------ 数据加载
def _failure_type(item: dict[str, Any]) -> str:
    """兼容 AI4I 的两种发布格式。

    - 旧版：有 `Failure Type` 单列（No Failure / TWF / ...）
    - 新版：只有 5 个 0/1 指示列（TWF/HDF/PWF/OSF/RNF），需还原类型
    """
    direct = item.get("Failure Type")
    if direct is not None and str(direct).strip():
        return str(direct).strip()
    for name in TYPES[1:]:
        if str(item.get(name, "0")).strip() in {"1", "1.0", "True", "true"}:
            return name
    return "No Failure"


def load_rows(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for item in csv.DictReader(fh):
            rows.append({
                "air_temp_c": float(item["Air temperature [K]"]) - 273.15,
                "process_temp_c": float(item["Process temperature [K]"]) - 273.15,
                "rpm": float(item["Rotational speed [rpm]"]),
                "torque_nm": float(item["Torque [Nm]"]),
                "tool_wear_min": float(item["Tool wear [min]"]),
                "failure": int(item["Machine failure"]),
                "type": _failure_type(item),
            })
    return rows


def split_indices(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """60/20/20 划分（与训练完全同一套种子与置换），供评测复现。"""
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(n)
    n_tr, n_ca = int(n * 0.6), int(n * 0.2)
    return idx[:n_tr], idx[n_tr:n_tr + n_ca], idx[n_tr + n_ca:]


def holdout_rows(path: str) -> list[dict[str, Any]]:
    """只返回**留出测试集**的行，避免评测时把训练样本算进去。"""
    rows = load_rows(path)
    _, _, te = split_indices(len(rows))
    return [rows[i] for i in te]


# ------------------------------------------------------------------ 网络
class MLP(nn.Module):
    def __init__(self, d_in: int, n_out: int, hidden: tuple[int, int] = (64, 32)) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden[0]), nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
            nn.Linear(hidden[1], n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # noqa: D102
        return self.net(x)


def _train_head(X: np.ndarray, y: np.ndarray, n_out: int, device: str,
                epochs: int = 800) -> MLP:
    torch.manual_seed(SEED)
    model = MLP(X.shape[1], n_out).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=300, gamma=0.5)
    xb = torch.tensor(X, dtype=torch.float32, device=device)
    yb = torch.tensor(y, dtype=torch.long, device=device)
    lossf = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(model(xb), yb)
        loss.backward()
        opt.step()
        sched.step()
    model.eval()
    return model


@torch.no_grad()
def _logits(model: nn.Module, X: np.ndarray, device: str) -> np.ndarray:
    xt = torch.tensor(np.ascontiguousarray(X), dtype=torch.float32, device=device)
    return model(xt).cpu().numpy()


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """在标定集上最小化 NLL，网格搜索温度 T（逐头独立）。"""
    best_t, best_nll = 1.0, float("inf")
    n = len(y)
    for t in np.exp(np.linspace(math.log(0.05), math.log(20.0), 240)):
        p = _softmax(logits / t)
        nll = -np.log(np.clip(p[np.arange(n), y], 1e-12, 1.0)).mean()
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    return best_t


# ------------------------------------------------------------------ 聚合
@dataclass
class SentinelHeads:
    """三个决策头的集合，附标准化参数与逐头温度。"""

    m_noul: MLP
    m_type: MLP
    m_sev: MLP
    mu: np.ndarray
    sd: np.ndarray
    temps: dict[str, float]
    device: str

    # -------------------------------------------------------------- 推理
    def predict(self, raw: dict[str, Any]) -> dict[str, Any]:
        feats = np.asarray([engineer(raw)], dtype=np.float32)
        xs = ((feats - self.mu) / self.sd).astype(np.float32)

        zn = _logits(self.m_noul, xs, self.device)
        zt = _logits(self.m_type, xs, self.device)
        zs = _logits(self.m_sev, xs, self.device)

        p_fault = float(_softmax(zn / self.temps["noul"])[0, 1])
        tp = _softmax(zt / self.temps["type"])[0]
        sp = _softmax(zs / self.temps["sev"])[0]
        return {
            "p_fault": p_fault,
            "type_probs": {TYPES[i]: float(tp[i]) for i in range(len(TYPES))},
            "sev_probs": {SEVERITY_LABELS[i]: float(sp[i]) for i in range(len(SEVERITY_LABELS))},
        }

    # -------------------------------------------------------------- 持久化
    def save(self, path: str) -> None:
        torch.save({
            "noul": self.m_noul.state_dict(),
            "type": self.m_type.state_dict(),
            "sev": self.m_sev.state_dict(),
            "mu": self.mu, "sd": self.sd, "temps": self.temps,
        }, path)

    @classmethod
    def load(cls, path: str, device: str) -> "SentinelHeads":
        blob = torch.load(path, map_location=device, weights_only=False)
        m_noul = MLP(len(FEATURE_NAMES), 2).to(device)
        m_type = MLP(len(FEATURE_NAMES), len(TYPES)).to(device)
        m_sev = MLP(len(FEATURE_NAMES), len(SEVERITY_LABELS)).to(device)
        m_noul.load_state_dict(blob["noul"])
        m_type.load_state_dict(blob["type"])
        m_sev.load_state_dict(blob["sev"])
        for m in (m_noul, m_type, m_sev):
            m.eval()
        return cls(m_noul, m_type, m_sev, blob["mu"], blob["sd"], blob["temps"], device)

    # -------------------------------------------------------------- 训练
    @classmethod
    def train(cls, data_csv: str, device: str) -> "SentinelHeads":
        rows = load_rows(data_csv)
        X = np.asarray([engineer(r) for r in rows], dtype=np.float32)
        y_noul = np.asarray([r["failure"] for r in rows], dtype=np.int64)
        y_type = np.asarray([TYPES.index(r["type"]) for r in rows], dtype=np.int64)
        y_sev = np.asarray([severity(r) for r in rows], dtype=np.int64)

        tr, ca, _ = split_indices(len(rows))        # 划分口径与评测共用同一实现

        mu = X[tr].mean(axis=0)
        sd = X[tr].std(axis=0) + 1e-9
        Xs = ((X - mu) / sd).astype(np.float32)

        m_noul = _train_head(Xs[tr], y_noul[tr], 2, device)
        m_type = _train_head(Xs[tr], y_type[tr], len(TYPES), device)
        m_sev = _train_head(Xs[tr], y_sev[tr], len(SEVERITY_LABELS), device)

        temps = {
            "noul": fit_temperature(_logits(m_noul, Xs[ca], device), y_noul[ca]),
            "type": fit_temperature(_logits(m_type, Xs[ca], device), y_type[ca]),
            "sev": fit_temperature(_logits(m_sev, Xs[ca], device), y_sev[ca]),
        }
        return cls(m_noul, m_type, m_sev, mu, sd, temps, device)

    @classmethod
    def load_or_train(cls, data_csv: str, device: str, cache: str,
                      retrain: bool = False) -> tuple["SentinelHeads", bool]:
        if not retrain and os.path.exists(cache):
            return cls.load(cache, device), True
        heads = cls.train(data_csv, device)
        heads.save(cache)
        return heads, False


def export_meta(heads: SentinelHeads, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({
            "features": FEATURE_NAMES,
            "types": TYPES,
            "severity_labels": SEVERITY_LABELS,
            "temperatures": {k: round(v, 6) for k, v in heads.temps.items()},
            "device": heads.device,
            "torch": torch.__version__,
            "seed": SEED,
        }, fh, ensure_ascii=False, indent=2)


# ------------------------------------------------------------------ 自测
def _self_test(data_csv: str, device: str) -> None:
    rows = load_rows(data_csv)
    X = np.asarray([engineer(r) for r in rows], dtype=np.float32)
    y_noul = np.asarray([r["failure"] for r in rows], dtype=np.int64)
    y_type = np.asarray([TYPES.index(r["type"]) for r in rows], dtype=np.int64)
    y_sev = np.asarray([severity(r) for r in rows], dtype=np.int64)

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(rows))
    n_te = int(len(rows) * 0.2)
    te = idx[-n_te:]

    heads, cached = SentinelHeads.load_or_train(  # noqa: F841
        data_csv, device, os.path.join(os.path.dirname(data_csv), "heads.pt"))
    Xs = ((X - heads.mu) / heads.sd).astype(np.float32)

    pn = _softmax(_logits(heads.m_noul, Xs[te], device) / heads.temps["noul"])[:, 1]
    pt = _softmax(_logits(heads.m_type, Xs[te], device) / heads.temps["type"]).argmax(1)
    ps = _softmax(_logits(heads.m_sev, Xs[te], device) / heads.temps["sev"]).argmax(1)

    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 f1_score, matthews_corrcoef, precision_score, recall_score)
    pred_n = (pn >= 0.5).astype(int)
    print(f"  缓存命中: {cached}   device: {device}   温度: "
          + ", ".join(f"{k}={v:.3f}" for k, v in heads.temps.items()))
    print(f"  Noul  Acc {accuracy_score(y_noul[te], pred_n):.4f}  "
          f"BAcc {balanced_accuracy_score(y_noul[te], pred_n):.4f}  "
          f"P {precision_score(y_noul[te], pred_n, zero_division=0):.4f}  "
          f"R {recall_score(y_noul[te], pred_n, zero_division=0):.4f}  "
          f"F1 {f1_score(y_noul[te], pred_n, zero_division=0):.4f}  "
          f"MCC {matthews_corrcoef(y_noul[te], pred_n):.4f}")
    print(f"  Type  Acc {accuracy_score(y_type[te], pt):.4f}  "
          f"MacroF1 {f1_score(y_type[te], pt, average='macro', zero_division=0):.4f}")
    print(f"  Sev   Acc {accuracy_score(y_sev[te], ps):.4f}  "
          f"MacroF1 {f1_score(y_sev[te], ps, average='macro', zero_division=0):.4f}")

    raw = {"air_temp_c": 24.1, "process_temp_c": 37.6, "rpm": 1380.0,
           "torque_nm": 62.3, "tool_wear_min": 243.0}
    print(f"\n  样例（明显故障）: {json.dumps(heads.predict(raw), ensure_ascii=False)[:220]}")


if __name__ == "__main__":
    import sys
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _self_test(sys.argv[1] if len(sys.argv) > 1 else "data/ai4i2020.csv", dev)
