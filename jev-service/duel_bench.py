#!/usr/bin/env python3
"""哨兵 EdgeSentinel —— 通用大模型 vs 垂类决策头 对决实验

核心问题：在工业设备状态判定这类「结构化数值 → 类型化决策」任务上，
通用大模型（27B / 8B）到底比我们训练的小 MLP 决策头强还是弱？

设计要点（保证公平）：
  · 三方吃**完全相同的样本**、**相同的划分种子**（20260921）
  · 大模型侧设三档信息量，观察「提示工程能否救回来」：
      L0 裸数值 / L1 加单位与正常范围 / L2 再加故障机理 + few-shot 示例
  · 一次调用同时输出三个决策（对标 Jev 并行采样）：故障? 类型? 严重度?
  · 统一指标：Accuracy / Balanced Acc / F1 / MCC / ECE / Brier / 格式合规率 / 延迟

依赖：numpy + scikit-learn + ollama（本地）
"""
from __future__ import annotations

import csv
import json
import math
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, precision_score, recall_score)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

DATA = "/home/USER/jev-service/data/ai4i2020.csv"
OLLAMA = "http://127.0.0.1:11434/api/chat"
SEED = 20260921
RNG = np.random.default_rng(SEED)

TYPES = ["No Failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
SEV_LABELS = ["观察", "计划", "尽快", "立即"]

MODELS = [
    ("modelscope.cn/unsloth/Qwen3.8-27B-GGUF:latest", "Qwen3-27B"),
    ("modelscope.cn/unsloth/Qwen3-8B-GGUF:latest", "Qwen3-8B"),
]
WORKERS = 4


# ------------------------------------------------------------------ 数据
def load_rows(path):
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def to_float(row, key):
    return float(row[key])


def encode(row):
    air = to_float(row, "Air temperature [K]")
    proc = to_float(row, "Process temperature [K]")
    rpm = to_float(row, "Rotational speed [rpm]")
    torque = to_float(row, "Torque [Nm]")
    wear = to_float(row, "Tool wear [min]")
    return np.array([
        air, proc, rpm, torque, wear,
        proc - air,
        rpm * torque * 2 * math.pi / 60.0 / 1000.0,
        wear * torque / 100.0,
        (proc - air) / max(rpm / 1000.0, 1e-6),
    ], dtype=np.float64)


def failure_type(row):
    for key in ("TWF", "HDF", "PWF", "OSF", "RNF"):
        if int(row[key]) == 1:
            return key
    return "No Failure"


def severity(row):
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


def describe(row):
    """把一行传感器读数写成自然语言（K 转 ℃，便于人类/模型读）。"""
    air = to_float(row, "Air temperature [K]") - 273.15
    proc = to_float(row, "Process temperature [K]") - 273.15
    rpm = to_float(row, "Rotational speed [rpm]")
    torque = to_float(row, "Torque [Nm]")
    wear = to_float(row, "Tool wear [min]")
    return (f"气温 {air:.1f}℃、过程温度 {proc:.1f}℃、"
            f"转速 {rpm:.0f}rpm、扭矩 {torque:.1f}Nm、刀具磨损 {wear:.0f}min")


def softmax(z, axis=-1):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def temperature_scale(logits, y, iters=200, lr=0.05):
    """多元温度缩放：最小化 NLL，dNLL/dT = (1/T^2)·Σ(y-p)·z。"""
    T = 1.0
    for _ in range(iters):
        p = softmax(logits / T)
        n = len(y)
        onehot = np.zeros_like(p)
        onehot[np.arange(n), y] = 1.0
        grad = np.sum((onehot - p) * logits / (T ** 2)) / n
        T = T - lr * grad
        T = float(np.clip(T, 0.05, 10.0))
    return T


def temperature_scale_binary(z, y, iters=400, lr=0.05):
    """二元温度缩放：z 必须是 logit（log p1 - log p0），不是 log p1。"""
    T = 1.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-z / T))
        grad = np.mean((y - p) * z / (T ** 2))
        T = T - lr * grad
        T = float(np.clip(T, 0.05, 10.0))
    return T


def ece_binary(p, y, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum():
            total += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return float(total)


def brier_binary(p, y):
    return float(np.mean((p - y) ** 2))


def ece_multiclass(P, y, bins=10):
    conf = P.max(1)
    pred = P.argmax(1)
    acc = (pred == y).astype(float)
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum():
            total += m.sum() / len(P) * abs(conf[m].mean() - acc[m].mean())
    return float(total)


def brier_multiclass(P, y):
    onehot = np.zeros_like(P)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((P - onehot) ** 2, axis=1)))


# ------------------------------------------------------------------ 提示词
OUT_SPEC = ('只输出一行 JSON，不要解释、不要 Markdown 代码块：'
            '{"fault": true 或 false, "fault_prob": 0到1之间的小数, '
            '"type": "TWF"/"HDF"/"PWF"/"OSF"/"RNF"/"No Failure" 之一, '
            '"severity": 0/1/2/3 之一}')

L1_KNOWLEDGE = (
    "参考范围：过程温度与气温之差（温升）正常 < 9K，> 12K 提示散热异常；"
    "刀具磨损 < 50min 为轻微，50~120 中等，120~200 较重，> 200min 高危；"
    "扭矩 × 转速 反映机械功率，高扭矩高转速易过载。"
)

L2_KNOWLEDGE = L1_KNOWLEDGE + (
    "\n故障机理：TWF=刀具磨损失效（磨损大、扭矩高）；"
    "HDF=散热失效（温升大、转速低）；"
    "PWF=功率失效（功率偏离额定）；"
    "OSF=过载失效（扭矩×转速超限）；"
    "RNF=随机失效（无明显征兆）。"
    "\n严重度：0=观察，1=计划，2=尽快，3=立即。"
    "\n示例：磨损 5min、温升 8.1K、扭矩 31.7Nm → 无故障，严重度 0。"
    "磨损 243min、温升 13.5K、扭矩 62.3Nm → 故障(TWF)，严重度 3。"
)


def build_prompt(desc: str, level: str) -> str:
    if level == "L0":
        return f"设备状态：{desc}。\n请判断该设备当前状态。\n{OUT_SPEC}"
    if level == "L1":
        return (f"你是工业设备预测性维护专家。\n{L1_KNOWLEDGE}\n"
                f"设备状态：{desc}。\n请判断该设备当前状态。\n{OUT_SPEC}")
    return (f"你是工业设备预测性维护专家。\n{L2_KNOWLEDGE}\n"
            f"设备状态：{desc}。\n请判断该设备当前状态。\n{OUT_SPEC}")


# ------------------------------------------------------------------ ollama
def call_ollama(model: str, prompt: str, think: bool = False,
                num_predict: int = 96, timeout: int = 180):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": think,
        "options": {"temperature": 0, "num_predict": num_predict},
    }
    req = urllib.request.Request(
        OLLAMA, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        return None, (time.perf_counter() - t0) * 1000.0, f"ERR {exc}"
    dt = (time.perf_counter() - t0) * 1000.0
    return str(body.get("message", {}).get("content", "")), dt, None


JSON_RE = re.compile(r"\{[^{}]*\}")


def parse_answer(text):
    """返回 (fault: bool|None, prob: float|None, type: str|None, sev: int|None)。"""
    if not text:
        return None, None, None, None
    m = JSON_RE.search(text)
    obj = None
    if m:
        try:
            obj = json.loads(m.group(0).replace("true", "true")
                             .replace("True", "true").replace("False", "false"))
        except Exception:  # noqa: BLE001
            obj = None
    if not isinstance(obj, dict):
        # 退化解析：直接找关键词
        low = text.lower()
        fault = True if '"fault": true' in low or "故障：是" in text else (
            False if '"fault": false' in low else None)
        typ = next((t for t in TYPES if t.lower() in low and t != "No Failure"), None)
        pm = re.search(r"(?:fault_prob|prob|probability)[^0-9]{0,6}([01]?\.?\d+)", low)
        prob = float(pm.group(1)) if pm else None
        return fault, prob, typ, None
    fault = obj.get("fault")
    if isinstance(fault, str):
        fault = fault.strip().lower() in ("true", "yes", "1")
    prob = obj.get("fault_prob", obj.get("prob"))
    try:
        prob = float(prob) if prob is not None else None
    except (TypeError, ValueError):
        prob = None
    if prob is not None and prob > 1.0:
        prob = prob / 100.0
    typ = obj.get("type")
    if isinstance(typ, str):
        typ = typ.strip()
        low = typ.lower()
        for t in TYPES:
            if t.lower() == low:
                typ = t
                break
        else:
            typ = None
    sev = obj.get("severity")
    try:
        sev = int(sev) if sev is not None else None
    except (TypeError, ValueError):
        sev = None
    if sev is not None and not (0 <= sev <= 3):
        sev = None
    return fault, prob, typ, sev


def run_model(model: str, tag: str, descs: list[str], level: str, think=False,
              num_predict: int | None = None, timeout: int = 180):
    prompts = [build_prompt(d, level) for d in descs]
    out = [None] * len(prompts)
    npred = num_predict or (512 if think else 96)

    def work(i):
        text, dt, err = call_ollama(model, prompts[i], think=think,
                                    num_predict=npred, timeout=timeout)
        return i, text, dt, err

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i, text, dt, err in ex.map(work, range(len(prompts))):
            f, p, t, s = parse_answer(text)
            out[i] = {"fault": f, "prob": p, "type": t, "sev": s,
                      "latency_ms": dt, "err": err}
    ok = sum(1 for o in out if o["fault"] is not None)
    print(f"    {tag:<10} {level}{'(CoT)' if think else '':<6} "
          f"可解析 {ok}/{len(out)}  平均 {np.mean([o['latency_ms'] for o in out]):.0f} ms")
    return out


# ------------------------------------------------------------------ 主流程
def main():
    print("=" * 78)
    print("通用大模型 vs 垂类决策头 · 对决实验")
    print("=" * 78)

    rows = load_rows(DATA)
    idx = RNG.permutation(len(rows))
    tr, ca, te = idx[:6000], idx[6000:8000], idx[8000:10000]
    print(f"\n[数据] 总 {len(rows)} 条；训练 {len(tr)} / 校准 {len(ca)} / 测试 {len(te)}")

    X = np.array([encode(rows[i]) for i in range(len(rows))])
    y_noul = np.array([int(rows[i]["Machine failure"]) for i in range(len(rows))])
    y_type = np.array([TYPES.index(failure_type(rows[i])) for i in range(len(rows))])
    y_sev = np.array([severity(rows[i]) for i in range(len(rows))])

    # ---- 决策头（与 sentinel_proto 完全一致）
    print("\n[训练] 垂类决策头 MLP(64,32) + 温度缩放校准")
    scaler = StandardScaler().fit(X[tr])
    Xs = scaler.transform(X)
    head_noul = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800,
                              random_state=SEED).fit(Xs[tr], y_noul[tr])
    head_type = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800,
                              random_state=SEED).fit(Xs[tr], y_type[tr])
    head_sev = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800,
                             random_state=SEED).fit(Xs[tr], y_sev[tr])

    # 校准集上求温度
    lp_n = head_noul.predict_log_proba(Xs[ca])
    z_ca_n = lp_n[:, 1] - lp_n[:, 0]          # logit，而非 log p1
    T_n = temperature_scale_binary(z_ca_n, y_noul[ca]) if len(set(y_noul[ca])) > 1 else 1.0
    lp_t = head_type.predict_log_proba(Xs[ca])
    T_t = temperature_scale(lp_t, y_type[ca])
    lp_s = head_sev.predict_log_proba(Xs[ca])
    T_s = temperature_scale(lp_s, y_sev[ca])
    print(f"    温度系数：Noul {T_n:.3f} / Choice {T_t:.3f} / Score {T_s:.3f}")

    # ---- 评估集：均衡抽样（故障样本全上，正常样本等量随机）
    te_fail = np.array([i for i in te if y_noul[i] == 1])
    te_ok = np.array([i for i in te if y_noul[i] == 0])
    sel_ok = RNG.choice(te_ok, len(te_fail), replace=False)
    ev = np.concatenate([te_fail, sel_ok])
    RNG.shuffle(ev)
    print(f"\n[评估集] 均衡抽样 n={len(ev)}（故障 {len(te_fail)} / 正常 {len(sel_ok)}）")

    descs = [describe(rows[i]) for i in ev]
    gt_fault = y_noul[ev]
    gt_type = y_type[ev]
    gt_sev = y_sev[ev]

    # ---- 决策头推理（含延迟）
    t0 = time.perf_counter()
    for _ in range(3):
        pn_raw = head_noul.predict_proba(Xs[ev])
    head_ms = (time.perf_counter() - t0) / 3 / len(ev) * 1000.0
    lp_ev = head_noul.predict_log_proba(Xs[ev])
    z_n = lp_ev[:, 1] - lp_ev[:, 0]
    p_head = 1.0 / (1.0 + np.exp(-(z_n / T_n)))
    pred_f_head = (p_head >= 0.5).astype(int)
    P_type = softmax(head_type.predict_log_proba(Xs[ev]) / T_t)
    pred_t_head = P_type.argmax(1)
    P_sev = softmax(head_sev.predict_log_proba(Xs[ev]) / T_s)
    pred_s_head = P_sev.argmax(1)

    # ---- 大模型
    available = []
    for mid, tag in MODELS:
        try:
            req = urllib.request.Request("http://127.0.0.1:11434/api/tags")
            with urllib.request.urlopen(req, timeout=10) as r:
                names = [m["name"] for m in json.loads(r.read().decode())["models"]]
        except Exception:  # noqa: BLE001
            names = []
        if any(mid.split(":")[0] in n for n in names):
            available.append((mid, tag))
        else:
            print(f"  [跳过] {tag} 未安装")
    print(f"\n[模型] 可用：{[t for _, t in available]}")

    results = {}
    levels = ["L0", "L1", "L2"]

    def dump_partial():
        """增量落盘：即使中途中断，已完成的部分也不会丢。"""
        try:
            with open("duel_partial.json", "w", encoding="utf-8") as fh:
                json.dump({"n_eval": int(len(ev)), "n_fail": int(len(te_fail)),
                           "results": results}, fh, ensure_ascii=False, default=str)
        except Exception:  # noqa: BLE001
            pass

    for mi, (mid, tag) in enumerate(available):
        for lv in levels:
            print(f"  运行 {tag} @ {lv} ...", flush=True)
            results[f"{tag}-{lv}"] = run_model(mid, tag, descs, lv)
            dump_partial()
        # CoT 只对第一个模型（27B）做，且限量——思考链单条可达数十秒
        if mi == 0:
            print(f"  运行 {tag} @ L2+CoT（20 条，限时 90s/条） ...", flush=True)
            cot = run_model(mid, tag, descs[:20], "L2", think=True,
                            num_predict=320, timeout=90)
            results[f"{tag}-L2-CoT"] = cot + [None] * (len(descs) - len(cot))
            dump_partial()

    # ---------------------------------------------------------------- 汇总
    def score_block(name, pred_f, pred_t, pred_s, p_f=None, P_t=None, P_s=None,
                    lat_ms=None, n_valid=None):
        n = len(gt_fault)
        if n_valid is not None and n_valid < n:
            m = slice(0, n_valid)
            gf, gt, gs = gt_fault[m], gt_type[m], gt_sev[m]
            pf, pt, ps = pred_f[m], pred_t[m], pred_s[m]
        else:
            gf, gt, gs = gt_fault, gt_type, gt_sev
            pf, pt, ps = pred_f, pred_t, pred_s
        pf = np.asarray(pf)
        pt = np.asarray(pt)
        ps = np.asarray(ps)
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
            "sev_mae": float(np.mean(np.abs(np.asarray(ps) - gs))),
            "sev_within1": float(np.mean(np.abs(np.asarray(ps) - gs) <= 1)),
        }
        if p_f is not None:
            p_f = np.clip(np.asarray(p_f, dtype=float), 0.0, 1.0)
            block["ece"] = ece_binary(p_f, gf.astype(float))
            block["brier"] = brier_binary(p_f, gf.astype(float))
        elif P_t is not None:
            P_t = np.asarray(P_t, dtype=float)
            block["ece"] = ece_multiclass(P_t, gt)
            block["brier"] = brier_multiclass(P_t, gt)
        else:
            block["ece"] = None
            block["brier"] = None
        block["latency_ms"] = lat_ms
        return block

    print("\n" + "=" * 78)
    print("结果汇总")
    print("=" * 78)

    summary = {}
    summary["哨兵决策头 (MLP+校准)"] = score_block(
        "head", pred_f_head, pred_t_head, pred_s_head,
        p_f=p_head, lat_ms=head_ms)

    # 基线
    rng2 = np.random.default_rng(7)
    rand_f = rng2.integers(0, 2, len(gt_fault))
    summary["随机基线"] = score_block("rand", rand_f,
                                      rng2.integers(0, len(TYPES), len(gt_fault)),
                                      rng2.integers(0, 4, len(gt_fault)),
                                      p_f=np.full(len(gt_fault), 0.5))
    maj = np.zeros(len(gt_fault), dtype=int)
    summary["全判正常（多数类）"] = score_block(
        "maj", maj, np.full(len(gt_fault), TYPES.index("No Failure")),
        np.full(len(gt_fault), 1),
        p_f=np.zeros(len(gt_fault)))

    for key, outs in results.items():
        valid = [o for o in outs if o is not None]
        faults, probs, types, sevs, lats = [], [], [], [], []
        for o in valid:
            f, p, t, s = o["fault"], o["prob"], o["type"], o["sev"]
            faults.append(int(bool(f)) if f is not None else 0)
            probs.append(float(p) if p is not None else 0.5)
            types.append(TYPES.index(t) if t in TYPES else TYPES.index("No Failure"))
            sevs.append(int(s) if s is not None else 0)
            lats.append(o["latency_ms"])
        n_ok = sum(1 for o in valid if o["fault"] is not None)
        valid_rate = n_ok / max(len(valid), 1)
        summary[key] = score_block(
            key, np.array(faults), np.array(types), np.array(sevs),
            p_f=np.array(probs), lat_ms=float(np.median(lats)),
            n_valid=len(valid))
        summary[key]["format_ok"] = valid_rate
        summary[key]["n"] = len(valid)

    summary["哨兵决策头 (MLP+校准)"]["format_ok"] = 1.0
    summary["随机基线"]["format_ok"] = 1.0
    summary["全判正常（多数类）"]["format_ok"] = 1.0

    # ---- 打印表格
    def row(name, s):
        fok = s.get("format_ok", 1.0)
        ece = f"{s['ece']:.4f}" if s["ece"] is not None else "—"
        br = f"{s['brier']:.4f}" if s["brier"] is not None else "—"
        lat = f"{s['latency_ms']:.3f}" if s["latency_ms"] else "—"
        return (f"| {name:<24} | {s['noul_acc']:.3f} | {s['noul_bacc']:.3f} | "
                f"{s['noul_f1']:.3f} | {s['noul_mcc']:.3f} | {s['type_macro_f1']:.3f} | "
                f"{s['sev_acc']:.3f} | {ece} | {br} | {fok*100:5.1f}% | {lat} |")

    print("\n### 表 1：Noul（是否故障）+ 综合")
    print("| 选手 | Acc | B.Acc | F1 | MCC | Type宏F1 | Sev Acc | ECE | Brier | 合规 | 延迟ms |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    order = ["哨兵决策头 (MLP+校准)"] + \
        [k for k in results.keys() if k.endswith("-L2") and "CoT" not in k] + \
        [k for k in results.keys() if k.endswith("-L1")] + \
        [k for k in results.keys() if k.endswith("-L0")] + \
        [k for k in results.keys() if "CoT" in k] + \
        ["全判正常（多数类）", "随机基线"]
    for k in order:
        if k in summary:
            print(row(k, summary[k]))

    print("\n### 表 2：三档信息量对大模型的影响（Noul 平衡准确率）")
    for tag in [t for _, t in available]:
        vals = []
        for lv in levels + ["L2-CoT"]:
            k = f"{tag}-{lv}"
            if k in summary:
                vals.append(f"{lv}: {summary[k]['noul_bacc']:.3f}")
        print(f"  {tag}:  " + "   ".join(vals))

    print("\n### 表 3：精确率 / 召回率 细分（Noul）")
    print(f"| {'选手':<24} | {'Precision':>9} | {'Recall':>7} |")
    print("|---|---|---|")
    for k in order:
        if k in summary:
            s = summary[k]
            print(f"| {k:<24} | {s['noul_prec']:9.3f} | {s['noul_rec']:7.3f} |")

    out = {
        "n_eval": int(len(ev)),
        "n_fail": int(len(te_fail)),
        "summary": summary,
        "order": order,
    }
    with open("duel_result.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2, default=str)
    print("\n已写出 duel_result.json")


if __name__ == "__main__":
    main()
