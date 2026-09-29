#!/usr/bin/env python3
"""各版本端到端性能横向对比（AI4I 2020 数据集）。

对比思路
--------
jev-decision-service 各版本的默认「模型」是规则引擎，规则词表跨版本基本没变，
因此单看模型本身没有可比性。真正能体现迭代差异的是**整条决策链路**：
  决策头 → 校准 → OOD/稀有类保护 → 策略路由 → 最终动作

所以这里固定同一个哨兵决策头（同样种子、同样数据训练），
把它接到各版本的 StructuredHeadProvider 上，对同一批 AI4I 测试样本跑一遍，
比较最终产出的**路由决策质量**与**延迟**。

用法：python3 version_bench.py <版本包目录> <输出json>
"""
from __future__ import annotations

import csv
import dataclasses
import inspect
import json
import math
import os
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

BASE = sys.argv[1]
OUT = sys.argv[2]
sys.path.insert(0, os.path.join(BASE, "src"))

from jev_service.calibration import DecisionHeadProfile          # noqa: E402
from jev_service.models import DecisionRequest                    # noqa: E402
from jev_service.engine import DecisionEngine                    # noqa: E402
from jev_service.policy import PolicyConfig, choose_action        # noqa: E402
from jev_service.provider import StructuredHeadProvider           # noqa: E402
from jev_service.questions import QuestionProfile                 # noqa: E402
from jev_service.state_builder import StateBuilder                # noqa: E402

DATA = "/home/USER/jev-service/data/ai4i2020.csv"
SEED = 20260921
EPOCHS = 800
FAULT_TYPES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
SEV_LABELS = ["观察", "计划", "尽快", "立即"]
KEYS = ["air_temp_c", "process_temp_c", "rpm", "torque_nm", "tool_wear_min"]

INDUSTRIAL_PROFILE = QuestionProfile(
    name="industrial-maintenance", version="v1",
    questions={
        "risk_level": {"type": "score", "levels": SEV_LABELS,
                       "question": "维护优先级？"},
        "needs_evidence": {"type": "noul", "question": "是否需要查阅手册/历史？"},
        "needs_clarification": {"type": "noul", "question": "是否需要现场确认？"},
        "allows_side_effect": {"type": "noul", "question": "是否允许自动派工单？"},
        "next_action": {"type": "choice",
                        "options": ["answer_from_context", "retrieve_evidence",
                                    "ask_clarification", "draft_suggestion",
                                    "human_review"],
                        "question": "下一步动作？"},
    },
)


# ------------------------------------------------------------------ 数据
def load_rows(path):
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def feats_of(row):
    air = float(row["Air temperature [K]"])
    proc = float(row["Process temperature [K]"])
    return {
        "air_temp_c": air - 273.15,
        "process_temp_c": proc - 273.15,
        "rpm": float(row["Rotational speed [rpm]"]),
        "torque_nm": float(row["Torque [Nm]"]),
        "tool_wear_min": float(row["Tool wear [min]"]),
    }


def encode(f):
    air = f["air_temp_c"] + 273.15
    proc = f["process_temp_c"] + 273.15
    rpm, torque, wear = f["rpm"], f["torque_nm"], f["tool_wear_min"]
    return [air, proc, rpm, torque, wear, proc - air,
            rpm * torque * 2 * math.pi / 60.0 / 1000.0,
            wear * torque / 100.0,
            (proc - air) / max(rpm / 1000.0, 1e-6)]


def severity(row):
    air = float(row["Air temperature [K]"])
    proc = float(row["Process temperature [K]"])
    torque = float(row["Torque [Nm]"])
    wear = float(row["Tool wear [min]"])
    diff = proc - air
    if wear >= 200 or (diff >= 12 and torque >= 50):
        return 3
    if wear >= 120 or diff >= 11:
        return 2
    if wear >= 50 or diff >= 9:
        return 1
    return 0


# ------------------------------------------------------------------ 决策头
class Head(nn.Module):
    def __init__(self, out_dim, in_dim=9, hidden=(64, 32)):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def fit_temp(z, y, binary=False, iters=600, lr=0.05):
    T = 1.0
    for _ in range(iters):
        if binary:
            p = 1 / (1 + np.exp(-np.clip(z / T, -60, 60)))
            g = np.mean((y - p) * z / (T ** 2))
        else:
            P = softmax(z / T)
            oh = np.zeros_like(P)
            oh[np.arange(len(y)), y] = 1.0
            g = np.sum((oh - P) * z / (T ** 2)) / len(y)
        T = float(np.clip(T - lr * g, 0.05, 10.0))
    return T


def train_heads(dev):
    rows = load_rows(DATA)
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(rows))
    tr, ca, te = idx[:6000], idx[6000:8000], idx[8000:10000]

    F = [feats_of(r) for r in rows]
    X = np.array([encode(f) for f in F], dtype=np.float64)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    Xs = np.ascontiguousarray((X - mu) / sd, dtype=np.float32)

    y_n = np.array([int(r["Machine failure"]) for r in rows])
    tl = ["No Failure"] + FAULT_TYPES
    y_t = np.array([tl.index(next((k for k in FAULT_TYPES if int(r[k]) == 1),
                                  "No Failure")) for r in rows])
    y_s = np.array([severity(r) for r in rows])

    heads = {}
    for key, y, dim in [("noul", y_n, 2), ("type", y_t, 6), ("sev", y_s, 4)]:
        torch.manual_seed(SEED)
        m = Head(dim).to(dev)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=300, gamma=0.5)
        lossf = nn.CrossEntropyLoss()
        Xt = torch.tensor(Xs[tr], dtype=torch.float32, device=dev)
        yt = torch.tensor(y[tr], dtype=torch.long, device=dev)
        m.train()
        for _ in range(EPOCHS):
            opt.zero_grad()
            lossf(m(Xt), yt).backward()
            opt.step()
            sch.step()
        m.eval()
        with torch.no_grad():
            zc = m(torch.tensor(Xs[ca], device=dev)).cpu().numpy()
        T = fit_temp(zc[:, 1] - zc[:, 0], y[ca], binary=True) if dim == 2 \
            else fit_temp(zc, y[ca])
        heads[key] = (m, T)
    return heads, mu, sd, y_n, y_s, te, F


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    heads, mu, sd, y_n, y_s, te, F = train_heads(dev)
    prior = float(y_n[te].mean())

    # ---- 预计算所有测试样本的决策头输出（批量，避免逐条 GPU 往返）
    cache = {}
    with torch.no_grad():
        Xall = np.ascontiguousarray(((np.array([encode(f) for f in F]) - mu) / sd),
                                    dtype=np.float32)
        Xt = torch.tensor(Xall[te], device=dev)
        z_n = heads["noul"][0](Xt).cpu().numpy()
        z_t = heads["type"][0](Xt).cpu().numpy()
        z_s = heads["sev"][0](Xt).cpu().numpy()
    p_fault = 1 / (1 + np.exp(-np.clip((z_n[:, 1] - z_n[:, 0]) / heads["noul"][1],
                                       -60, 60)))
    Pt = softmax(z_t / heads["type"][1])
    Ps = softmax(z_s / heads["sev"][1])
    for j, i in enumerate(te):
        key = tuple(round(F[i][k], 3) for k in KEYS)
        cache[key] = (float(p_fault[j]), Pt[j].copy(), Ps[j].copy())

    def evaluator(features: dict) -> dict:
        key = tuple(round(float(features[k]), 3) for k in KEYS)
        p, pt, ps = cache[key]
        sev = int(ps.argmax())
        # 工业策略：置信度够高且非「立即」才自动；否则转人工。
        # 刻意不设「追问」分支——产线上没有人可以追问，含糊就必须转人工。
        conf = max(p, 1.0 - p)
        allow = bool(conf >= 0.9 and sev < 3)
        if sev >= 3:
            nxt = "human_review"
        elif allow:
            nxt = "draft_suggestion" if p >= 0.5 else "answer_from_context"
        else:
            nxt = "human_review"
        opts = INDUSTRIAL_PROFILE.questions["next_action"]["options"]
        return {
            "risk_level": {"type": "score", "score": sev,
                           "probabilities": {SEV_LABELS[i]: round(float(ps[i]), 4)
                                             for i in range(4)}},
            "needs_evidence": {"type": "noul", "noul": 0.05},
            "needs_clarification": {"type": "noul", "noul": 0.05},
            "allows_side_effect": {"type": "noul",
                                   "noul": round(0.9 if allow else 0.05, 4)},
            "next_action": {"type": "choice", "choice": nxt,
                            "probabilities": {o: round(0.7 if o == nxt else 0.3 / 4, 4)
                                              for o in opts}},
        }

    # ---- 按版本支持的字段构造 profile
    supported = {f.name for f in dataclasses.fields(DecisionHeadProfile)}
    kw = {"name": "sentinel-mlp", "version": "d2", "calibrated": True,
          "temperature": 1.0, "train_positive_rate": prior,
          "calibration_distribution": "ai4i-natural",
          "label_provenance": "derived_rule"}
    if "class_support" in supported:
        kw.update({"class_support": {"TWF": 6, "HDF": 29, "PWF": 18, "OSF": 14,
                                     "RNF": 6}, "min_class_support": 1,
                   "rare_class_threshold": 10})
    if "class_support_head" in supported:
        kw.update({"class_support_head": "risk_level"})
    profile = DecisionHeadProfile(**kw)

    provider = StructuredHeadProvider(evaluator=evaluator, profile=profile,
                                      questions=INDUSTRIAL_PROFILE, name="sentinel")
    policy_config = PolicyConfig(urgent_score=3.0)
    engine = DecisionEngine(provider=provider, policy=policy_config)

    takes_warnings = "warnings" in inspect.signature(choose_action).parameters

    # ---- 跑全测试集
    stats = Counter()
    sev3_total = sev3_miss = 0
    ok_total = ok_blocked = 0
    fault_total = fault_reviewed = 0
    auto_total = auto_correct = 0
    lat = []
    for i in te:
        f = F[i]
        req = DecisionRequest(conversation_id=f"b{i}", current_message="设备状态上报",
                              conversation_state={"structured_features": f})
        t0 = time.perf_counter()
        try:
            resp = engine.decide(req)
        except Exception as exc:                                  # noqa: BLE001
            stats["provider_error"] += 1
            continue
        lat.append((time.perf_counter() - t0) * 1000)

        act = resp.recommended_action.kind
        stats[act] += 1
        real_fault = int(y_n[i]) == 1
        sev = int(y_s[i])
        if sev >= 3:
            sev3_total += 1
            if act != "human_review":
                sev3_miss += 1
        if not real_fault:
            ok_total += 1
            if act in ("human_review", "ask_clarification"):
                ok_blocked += 1
        else:
            fault_total += 1
            if act in ("human_review",):
                fault_reviewed += 1
        AUTO = ("answer_from_context", "draft_suggestion",
                "auto_close", "auto_dispatch")
        if act in AUTO:
            auto_total += 1
            # answer_from_context = 判定正常并直接回答 → 真值正常才算对
            # draft_suggestion / auto_dispatch = 起草工单 → 真值故障才算对
            if act in ("answer_from_context", "auto_close"):
                auto_correct += int(not real_fault)
            else:
                auto_correct += int(real_fault)

    n = len(lat)
    result = {
        "version": os.path.basename(BASE.rstrip("/")),
        "n": n,
        "provider_errors": stats["provider_error"],
        "actions": {k: v for k, v in stats.items() if k != "provider_error"},
        "automation_rate": round(auto_total / max(n, 1), 4),
        "auto_precision": round(auto_correct / max(auto_total, 1), 4),
        "sev3_total": sev3_total,
        "sev3_miss": sev3_miss,
        "sev3_miss_rate": round(sev3_miss / max(sev3_total, 1), 4),
        "normal_total": ok_total,
        "normal_blocked": ok_blocked,
        "false_block_rate": round(ok_blocked / max(ok_total, 1), 4),
        "fault_total": fault_total,
        "fault_reviewed": fault_reviewed,
        "fault_review_rate": round(fault_reviewed / max(fault_total, 1), 4),
        "latency_p50_ms": round(float(np.percentile(lat, 50)), 3) if lat else None,
        "latency_p95_ms": round(float(np.percentile(lat, 95)), 3) if lat else None,
        "profile_fields": sorted(supported),
        "policy_takes_warnings": takes_warnings,
        "supports_class_support_head": "class_support_head" in supported,
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
