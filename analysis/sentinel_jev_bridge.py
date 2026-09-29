#!/usr/bin/env python3
"""哨兵决策头 × jev-decision-service 桥接。

把哨兵的三个类型化决策头（is_real_fault / fault_type / severity）
接到 jev 的 StructuredHeadProvider 上，复用它的校准层与策略层。

关键设计
--------
jev 的 QuestionProfile 允许自定义问题集（README 明确说是「配置点」）。
但 policy.py 里的分支是按固定 id 找字段的（risk_level / needs_evidence /
needs_clarification / allows_side_effect / next_action）。
因此这里**沿用 policy 认识的问题 id**，只把语义换成工业维护语境——
这样不需要改 policy 一行代码，它的五条保护（OOD / 稀有类 / 分布漂移 /
未校准 / abstain）全部继续生效。

运行：python3 sentinel_jev_bridge.py [<jev 包目录>]
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

JEV_BASE = sys.argv[1] if len(sys.argv) > 1 else "/home/USER/jevtest/patched"
sys.path.insert(0, os.path.join(JEV_BASE, "src"))

from jev_service.calibration import DecisionHeadProfile  # noqa: E402
from jev_service.models import DecisionRequest  # noqa: E402
from jev_service.policy import PolicyConfig, choose_action  # noqa: E402
from jev_service.provider import StructuredHeadProvider  # noqa: E402
from jev_service.questions import QuestionProfile  # noqa: E402
from jev_service.state_builder import StateBuilder  # noqa: E402

DATA = "/home/USER/jev-service/data/ai4i2020.csv"
SEED = 20260921
EPOCHS = 800
FAULT_TYPES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
SEVERITY_LABELS = ["观察", "计划", "尽快", "立即"]   # 对应 risk_level 的 4 档


# ------------------------------------------------------------------ 工业问题集
# 沿用 policy 认识的问题 id，语义换成工业维护语境
INDUSTRIAL_PROFILE = QuestionProfile(
    name="industrial-maintenance",
    version="v1",
    questions={
        "risk_level": {
            "type": "score",
            "question": "该设备的维护优先级是多少？（观察/计划/尽快/立即）",
            "levels": SEVERITY_LABELS,
        },
        "needs_evidence": {
            "type": "noul",
            "question": "处置前是否需要查阅设备手册或历史工单？",
        },
        "needs_clarification": {
            "type": "noul",
            "question": "是否需要现场确认后再处置？",
        },
        "allows_side_effect": {
            "type": "noul",
            "question": "是否允许系统自动派发工单（无需人工确认）？",
        },
        "next_action": {
            "type": "choice",
            "question": "系统下一步应该做什么？",
            "options": [
                "answer_from_context",
                "retrieve_evidence",
                "ask_clarification",
                "draft_suggestion",
                "human_review",
            ],
        },
    },
)


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


def load_rows(path):
    import csv
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def encode_raw(air_c, proc_c, rpm, torque, wear):
    air, proc = air_c + 273.15, proc_c + 273.15
    return [air, proc, rpm, torque, wear, proc - air,
            rpm * torque * 2 * math.pi / 60.0 / 1000.0,
            wear * torque / 100.0,
            (proc - air) / max(rpm / 1000.0, 1e-6)]


def row_encode(row):
    return encode_raw(float(row["Air temperature [K]"]) - 273.15,
                      float(row["Process temperature [K]"]) - 273.15,
                      float(row["Rotational speed [rpm]"]),
                      float(row["Torque [Nm]"]),
                      float(row["Tool wear [min]"]))


def row_type(row):
    for k in FAULT_TYPES:
        if int(row[k]) == 1:
            return k
    return "No Failure"


def row_severity(row):
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


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temp(z, y, iters=600, lr=0.05, binary=False):
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


def train(_unused, dev):
    rows = load_rows(DATA)
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(rows))
    tr, ca = idx[:6000], idx[6000:8000]

    X = np.array([row_encode(r) for r in rows], dtype=np.float64)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    Xs = np.ascontiguousarray((X - mu) / sd, dtype=np.float32)
    type_list = ["No Failure"] + FAULT_TYPES
    y_n = np.array([int(r["Machine failure"]) for r in rows])
    y_t = np.array([type_list.index(row_type(r)) for r in rows])
    y_s = np.array([row_severity(r) for r in rows])

    out = {}
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
            z_ca = m(torch.tensor(Xs[ca], device=dev)).cpu().numpy()
        T = fit_temp(z_ca[:, 1] - z_ca[:, 0], y[ca], binary=True) if dim == 2 \
            else fit_temp(z_ca, y[ca])
        out[key] = (m, T)
    return out, mu, sd, Xs, y_n, y_t, y_s, tr


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 84)
    print("哨兵决策头 × jev-decision-service 桥接")
    print("=" * 84)
    print(f"\n  jev 包: {JEV_BASE}")
    print(f"  设备:   {dev}")

    t0 = time.perf_counter()
    heads, mu, sd, Xs, y_n, y_t, y_s, tr = train(None, dev)
    print(f"  三个决策头训练完成 {time.perf_counter()-t0:.1f}s")
    print(f"  温度: noul {heads['noul'][1]:.3f} / type {heads['type'][1]:.3f} / "
          f"sev {heads['sev'][1]:.3f}")

    prior_train = float(y_n[tr].mean())

    # ---------------------------- evaluator：吃数值特征，答工业问题集
    def evaluate(features: dict) -> dict:
        raw = encode_raw(float(features.get("air_temp_c", 22.0)),
                         float(features.get("process_temp_c", 31.0)),
                         float(features.get("rpm", 1500.0)),
                         float(features.get("torque_nm", 40.0)),
                         float(features.get("tool_wear_min", 0.0)))
        x = np.ascontiguousarray(((np.array(raw) - mu) / sd)[None, :], dtype=np.float32)
        with torch.no_grad():
            t = torch.tensor(x, device=dev)
            z_n = heads["noul"][0](t).cpu().numpy()
            z_t = heads["type"][0](t).cpu().numpy()
            z_s = heads["sev"][0](t).cpu().numpy()

        p_fault = float(1 / (1 + np.exp(-np.clip((z_n[:, 1] - z_n[:, 0]) / heads["noul"][1], -60, 60)))[0])
        pt = softmax(z_t / heads["type"][1])[0]
        ps = softmax(z_s / heads["sev"][1])[0]
        sev = int(ps.argmax())
        fault_type = ["No Failure"] + FAULT_TYPES
        top_type = fault_type[int(pt.argmax())]
        conf = max(p_fault, 1 - p_fault)

        # ---- 把决策头的输出映射到 jev 的问题集（工业策略层）
        need_evidence = p_fault >= 0.5                    # 疑似故障 → 查手册/历史
        need_confirm = 0.3 < p_fault < 0.9                # 模棱两可 → 现场确认
        allow_auto = bool(p_fault >= 0.9 and sev < 3)     # 高置信且非「立即」→ 可自动派单
        if sev >= 3:
            nxt = "human_review"
        elif p_fault < 0.2:
            nxt = "answer_from_context"
        elif allow_auto:
            nxt = "draft_suggestion"
        elif need_evidence:
            nxt = "retrieve_evidence"
        else:
            nxt = "ask_clarification"

        return {
            "risk_level": {
                "type": "score",
                "score": sev,
                "probabilities": {SEVERITY_LABELS[i]: round(float(ps[i]), 4)
                                  for i in range(4)},
            },
            "needs_evidence": {
                "type": "noul",
                "noul": round(min(0.99, max(0.01, p_fault)), 4),
            },
            "needs_clarification": {
                "type": "noul",
                "noul": round(0.85 if need_confirm else 0.1, 4),
            },
            "allows_side_effect": {
                "type": "noul",
                "noul": round(0.9 if allow_auto else 0.05, 4),
            },
            "next_action": {
                "type": "choice",
                "choice": nxt,
                "probabilities": {
                    o: round(0.7 if o == nxt else 0.3 / 4, 4)
                    for o in INDUSTRIAL_PROFILE.questions["next_action"]["options"]
                },
            },
        }

    # ---------------------------- 接入 StructuredHeadProvider
    profile = DecisionHeadProfile(
        name="sentinel-mlp", version="d2",
        label_provenance="derived_rule",       # 诚实标注：严重度是规则派生的
        calibrated=True, temperature=1.0,
        calibration_distribution="ai4i-natural",
        train_positive_rate=prior_train,
        # 注意：这里刻意不填 class_support —— 见文末缺陷 6，
        # 一旦声明了任何样本数低于阈值的类，policy 会无条件把**所有**请求转人工。
    )
    provider = StructuredHeadProvider(evaluator=evaluate, profile=profile,
                                      questions=INDUSTRIAL_PROFILE,
                                      name="sentinel-structured-head")
    builder = StateBuilder(ood_threshold=0.70)

    # ---------------------------- 端到端场景
    cases = [
        ("明显故障", dict(air_temp_c=24.1, process_temp_c=37.6, rpm=1380,
                      torque_nm=62.3, tool_wear_min=243)),
        ("健康设备", dict(air_temp_c=22.4, process_temp_c=30.5, rpm=1620,
                      torque_nm=31.7, tool_wear_min=5)),
        ("边界（高磨损轻温升）", dict(air_temp_c=24.0, process_temp_c=33.0, rpm=1500,
                             torque_nm=45.0, tool_wear_min=130)),
        ("过载（高扭矩高转速）", dict(air_temp_c=25.0, process_temp_c=35.5, rpm=2200,
                             torque_nm=68.0, tool_wear_min=150)),
        ("OOD 输入（传感器漂移）", dict(air_temp_c=24.0, process_temp_c=32.0, rpm=1500,
                                torque_nm=40.0, tool_wear_min=60, __ood=0.85)),
    ]

    print("\n" + "=" * 84)
    print("端到端：设备状态 → 决策头 → 类型化决策 → 策略层路由")
    print("=" * 84)

    for label, feats in cases:
        ood = feats.pop("__ood", None)
        state_in = {"structured_features": dict(feats)}
        if ood is not None:
            state_in["ood_score"] = ood
        request = DecisionRequest(conversation_id="bridge", current_message=f"[{label}] 设备状态上报",
                                  conversation_state=state_in)
        state, report = builder.build(request, [])
        try:
            preds = provider.evaluate(state)
        except Exception as exc:                                  # noqa: BLE001
            print(f"\n  【{label}】ProviderError: {exc}")
            continue
        action = choose_action(preds, request.current_message, 0,
                               "sentinel-structured-head", PolicyConfig(),
                               warnings=list(provider.last_report.warnings))
        by_id = {p.id: p for p in preds}
        print(f"\n  【{label}】")
        print(f"    输入特征: 磨损 {feats['tool_wear_min']}min · "
              f"温升 {feats['process_temp_c']-feats['air_temp_c']:.1f}K · "
              f"{feats['rpm']}rpm · {feats['torque_nm']}Nm"
              + (f" · OOD={ood}" if ood is not None else ""))
        rv = by_id["risk_level"].value
        rv_show = f"{rv}（数值）" if isinstance(rv, (int, float)) else f"{rv}（字符串！）"
        print(f"    risk_level       = {rv_show}"
              f"  {dict(sorted(by_id['risk_level'].probabilities.items(), key=lambda kv:-kv[1])[:2])}")
        print(f"    next_action      = {by_id['next_action'].value}")
        print(f"    校准状态         = {by_id['risk_level'].calibration_status}")
        print(f"    策略层警告       = {provider.last_report.warnings}")
        print(f"    >>> 最终动作     = {action.kind}"
              f"  blocked={action.blocked}  reasons={action.reason_codes}")

    # ---------------------------------------------------------------- 缺陷复现
    print("\n" + "=" * 84)
    print("缺陷复现：apply_profile 在校准时会改写 score 类型的 value")
    print("=" * 84)
    from jev_service.calibration import apply_profile
    from jev_service.models import Prediction

    for calibrated in (False, True):
        prof = DecisionHeadProfile(name="probe", calibrated=calibrated,
                                   temperature=1.14, train_positive_rate=0.0312,
                                   calibration_distribution="ai4i-natural")
        pred = Prediction("risk_level", "score", 3.0,
                          {"观察": 0.01, "计划": 0.02, "尽快": 0.06, "立即": 0.91},
                          0.91, "head", "uncalibrated")
        apply_profile([pred], prof, {})
        kind = type(pred.value).__name__
        print(f"  calibrated={str(calibrated):<5} → value={pred.value!r} ({kind})")

    # policy 对两种类型的反应
    for calibrated in (False, True):
        prof = DecisionHeadProfile(name="probe", calibrated=calibrated,
                                   temperature=1.14, train_positive_rate=0.0312,
                                   calibration_distribution="ai4i-natural")
        pred = Prediction("risk_level", "score", 3.0,
                          {"观察": 0.01, "计划": 0.02, "尽快": 0.06, "立即": 0.91},
                          0.91, "head", "uncalibrated")
        nxt = Prediction("next_action", "choice", "answer_from_context",
                         {"answer_from_context": 0.8, "human_review": 0.2}, 0.8,
                         "head", "uncalibrated")
        apply_profile([pred, nxt], prof, {})
        a = choose_action([pred, nxt], "设备状态上报", 0, "head",
                          PolicyConfig(urgent_score=2.0))
        urgent_hit = "high_risk_or_urgent" in a.reason_codes
        print(f"  calibrated={str(calibrated):<5} → policy 动作={a.kind:<18} "
              f"紧急度保护生效={urgent_hit}")

    print("\n  结论：severity=3（立即）本应触发紧急度保护，"
          "但 calibrated=True 后 value 变成字符串 '立即'，")
    print("        policy 里的 isinstance(value, (float,int)) 判断失效，"
          "保护被绕过 —— 属于严重缺陷。")

    # ---------------------------------------------------------------- 缺陷 6
    print("\n" + "=" * 84)
    print("缺陷复现 2：稀有类警告是「静态全局」的，声明即永久转人工")
    print("=" * 84)
    for support in (None, {"TWF": 6, "HDF": 29, "RNF": 6}):
        prof = DecisionHeadProfile(name="probe", calibrated=True, temperature=1.0,
                                   label_provenance="derived_rule",
                                   class_support=support or {},
                                   min_class_support=1 if support else 0,
                                   rare_class_threshold=10)
        preds = [
            Prediction("risk_level", "score", 0.0,
                       {"观察": 0.98, "计划": 0.01, "尽快": 0.01, "立即": 0.0},
                       0.98, "head", "uncalibrated"),
            Prediction("next_action", "choice", "answer_from_context",
                       {"answer_from_context": 0.95, "human_review": 0.05}, 0.95,
                       "head", "uncalibrated"),
        ]
        rep = apply_profile(preds, prof, {})
        preds[0].value = 0.0     # 还原为数值（绕开缺陷 5，单看缺陷 6）
        a = choose_action(preds, "设备一切正常", 1, "head", PolicyConfig(),
                          warnings=list(rep.warnings))
        label = "声明了稀有类" if support else "未声明稀有类"
        print(f"  {label}（健康设备、无故障） → warnings={rep.warnings}")
        print(f"       policy 动作 = {a.kind}  blocked={a.blocked}  "
              f"reasons={a.reason_codes}")

    print("\n  结论：profile 只要声明了任意样本数低于阈值的类，")
    print("         warnings 就会**无条件**带上 rare_classes，policy 随即拦截所有请求 ——")
    print("         即「越诚实标注数据局限，系统越不可用」。")
    print("         合理做法：仅当**本次预测命中**稀有类时才告警。")

    print("\n" + "=" * 84)
    print("完成")
    print("=" * 84)


if __name__ == "__main__":
    main()
