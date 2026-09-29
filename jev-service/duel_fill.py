#!/usr/bin/env python3
"""从 duel_partial.json 续跑并汇总。

SSH 会话超时会导致主脚本中断，但每个配置跑完都会增量落盘。
本脚本复现评估集（同种子）、补跑缺失配置、重建决策头，产出完整 duel_result.json。
"""
from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, "/home/USER/jev-service")
import duel_bench as D  # noqa: E402

PARTIAL = "/home/USER/jev-service/duel_partial.json"
RESULT = "/home/USER/jev-service/duel_result.json"

rows = D.load_rows(D.DATA)
RNG = np.random.default_rng(D.SEED)
idx = RNG.permutation(len(rows))
tr, ca, te = idx[:6000], idx[6000:8000], idx[8000:10000]

y_noul = np.array([int(r["Machine failure"]) for r in rows])
y_type = np.array([D.TYPES.index(D.failure_type(r)) for r in rows])
y_sev = np.array([D.severity(r) for r in rows])

te_fail = np.array([i for i in te if y_noul[i] == 1])
te_ok = np.array([i for i in te if y_noul[i] == 0])
sel_ok = RNG.choice(te_ok, len(te_fail), replace=False)
ev = np.concatenate([te_fail, sel_ok])
RNG.shuffle(ev)
print(f"[评估集] n={len(ev)}（故障 {len(te_fail)} / 正常 {len(sel_ok)}）")

with open(PARTIAL, encoding="utf-8") as fh:
    P = json.load(fh)
results = P["results"]

descs = [D.describe(rows[i]) for i in ev]

# ---- 补跑缺失配置
todo = []
for mid, tag in D.MODELS:
    for lv in ["L0", "L1", "L2"]:
        if f"{tag}-{lv}" not in results:
            todo.append((mid, tag, lv))
for mid, tag, lv in todo:
    print(f"  补跑 {tag} @ {lv} ...", flush=True)
    results[f"{tag}-{lv}"] = D.run_model(mid, tag, descs, lv)
    with open(PARTIAL, "w", encoding="utf-8") as fh:
        json.dump({"n_eval": len(ev), "n_fail": int(len(te_fail)),
                   "results": results}, fh, ensure_ascii=False, default=str)

# ---- 重建决策头
X = np.array([D.encode(rows[i]) for i in range(len(rows))])
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

scaler = StandardScaler().fit(X[tr])
Xs = scaler.transform(X)
h_n = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800,
                    random_state=D.SEED).fit(Xs[tr], y_noul[tr])
h_t = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800,
                    random_state=D.SEED).fit(Xs[tr], y_type[tr])
h_s = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800,
                    random_state=D.SEED).fit(Xs[tr], y_sev[tr])

lp_n = h_n.predict_log_proba(Xs[ca])
T_n = D.temperature_scale_binary(lp_n[:, 1] - lp_n[:, 0], y_noul[ca])
T_t = D.temperature_scale(h_t.predict_log_proba(Xs[ca]), y_type[ca])
T_s = D.temperature_scale(h_s.predict_log_proba(Xs[ca]), y_sev[ca])
print(f"  温度：Noul {T_n:.3f} / Choice {T_t:.3f} / Score {T_s:.3f}")

import time  # noqa: E402

t0 = time.perf_counter()
for _ in range(3):
    _p = h_n.predict_proba(Xs[ev])
head_ms = (time.perf_counter() - t0) / 3 / len(ev) * 1000.0
lp_ev = h_n.predict_log_proba(Xs[ev])
p_head = 1.0 / (1.0 + np.exp(-((lp_ev[:, 1] - lp_ev[:, 0]) / T_n)))
pred_f_head = (p_head >= 0.5).astype(int)
P_type = D.softmax(h_t.predict_log_proba(Xs[ev]) / T_t)
pred_t_head = P_type.argmax(1)
P_sev = D.softmax(h_s.predict_log_proba(Xs[ev]) / T_s)
pred_s_head = P_sev.argmax(1)

gt_fault, gt_type, gt_sev = y_noul[ev], y_type[ev], y_sev[ev]

from sklearn.metrics import (accuracy_score, balanced_accuracy_score,  # noqa: E402
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score)


def score_block(pred_f, pred_t, pred_s, p_f=None, lat_ms=None, n_valid=None):
    """与 duel_bench.main 内同名函数保持一致（该函数是嵌套定义，无法从模块级复用）。"""
    n = len(gt_fault)
    if n_valid is not None and n_valid < n:
        m = slice(0, n_valid)
        gf, gt, gs = gt_fault[m], gt_type[m], gt_sev[m]
        pf, pt, ps = pred_f[m], pred_t[m], pred_s[m]
    else:
        gf, gt, gs = gt_fault, gt_type, gt_sev
        pf, pt, ps = pred_f, pred_t, pred_s
    pf, pt, ps = np.asarray(pf), np.asarray(pt), np.asarray(ps)
    block = {
        "noul_acc": float(accuracy_score(gf, pf)),
        "noul_bacc": float(balanced_accuracy_score(gf, pf)),
        "noul_prec": float(precision_score(gf, pf, zero_division=0)),
        "noul_rec": float(recall_score(gf, pf, zero_division=0)),
        "noul_f1": float(f1_score(gf, pf, zero_division=0)),
        "noul_mcc": float(matthews_corrcoef(gf, pf)) if len(set(pf)) > 1 else 0.0,
        "type_acc": float(accuracy_score(gt, pt)),
        "type_macro_f1": float(f1_score(gt, pt, average="macro", zero_division=0)),
        "sev_acc": float(accuracy_score(gs, ps)),
        "sev_macro_f1": float(f1_score(gs, ps, average="macro", zero_division=0)),
        "sev_mae": float(np.mean(np.abs(ps - gs))),
        "sev_within1": float(np.mean(np.abs(ps - gs) <= 1)),
    }
    if p_f is not None:
        p_f = np.clip(np.asarray(p_f, dtype=float), 0.0, 1.0)
        block["ece"] = D.ece_binary(p_f, gf.astype(float))
        block["brier"] = D.brier_binary(p_f, gf.astype(float))
    else:
        block["ece"] = None
        block["brier"] = None
    block["latency_ms"] = lat_ms
    return block


summary = {}
summary["哨兵决策头 (MLP+校准)"] = score_block(pred_f_head, pred_t_head, pred_s_head, p_f=p_head, lat_ms=head_ms)
summary["哨兵决策头 (MLP+校准)"]["format_ok"] = 1.0

rng2 = np.random.default_rng(7)
summary["随机基线"] = score_block(
    rng2.integers(0, 2, len(gt_fault)),
    rng2.integers(0, len(D.TYPES), len(gt_fault)),
    rng2.integers(0, 4, len(gt_fault)), p_f=np.full(len(gt_fault), 0.5))
summary["随机基线"]["format_ok"] = 1.0
summary["全判正常（多数类）"] = score_block(
    np.zeros(len(gt_fault), dtype=int),
    np.full(len(gt_fault), D.TYPES.index("No Failure")),
    np.full(len(gt_fault), 1), p_f=np.zeros(len(gt_fault)))
summary["全判正常（多数类）"]["format_ok"] = 1.0

for key, outs in results.items():
    valid = [o for o in outs if o is not None]
    faults, probs, types, sevs, lats = [], [], [], [], []
    for o in valid:
        f, p, t, s = o["fault"], o["prob"], o["type"], o["sev"]
        faults.append(int(bool(f)) if f is not None else 0)
        probs.append(float(p) if p is not None else 0.5)
        types.append(D.TYPES.index(t) if t in D.TYPES else D.TYPES.index("No Failure"))
        sevs.append(int(s) if s is not None else 0)
        lats.append(o["latency_ms"])
    n_ok = sum(1 for o in valid if o["fault"] is not None)
    n_valid = len(valid)
    summary[key] = score_block(
        np.array(faults), np.array(types), np.array(sevs),
        p_f=np.array(probs), lat_ms=float(np.median(lats)), n_valid=n_valid)
    summary[key]["format_ok"] = n_ok / max(n_valid, 1)
    summary[key]["n"] = n_valid

order = ["哨兵决策头 (MLP+校准)"] + \
    [k for k in results if k.endswith("-L2") and "CoT" not in k] + \
    [k for k in results if k.endswith("-L1")] + \
    [k for k in results if k.endswith("-L0")] + \
    [k for k in results if "CoT" in k] + \
    ["全判正常（多数类）", "随机基线"]

with open(RESULT, "w", encoding="utf-8") as fh:
    json.dump({"n_eval": int(len(ev)), "n_fail": int(len(te_fail)),
               "summary": summary, "order": order}, fh,
              ensure_ascii=False, indent=2, default=str)

# ---- 打印
def line(name, s):
    ece = f"{s['ece']:.4f}" if s.get("ece") is not None else "—"
    br = f"{s['brier']:.4f}" if s.get("brier") is not None else "—"
    lat = f"{s['latency_ms']:.1f}" if s.get("latency_ms") else "—"
    return (f"{name:<22} Acc {s['noul_acc']:.3f}  BAcc {s['noul_bacc']:.3f}  "
            f"F1 {s['noul_f1']:.3f}  MCC {s['noul_mcc']:.3f}  "
            f"TypeF1 {s['type_macro_f1']:.3f}  SevAcc {s['sev_acc']:.3f}  "
            f"ECE {ece}  Brier {br}  合规 {(s.get('format_ok') or 0)*100:5.1f}%  "
            f"{lat} ms")

print("\n" + "=" * 100)
for k in order:
    if k in summary:
        print(line(k, summary[k]))
print("=" * 100)
print("已写出", RESULT)
