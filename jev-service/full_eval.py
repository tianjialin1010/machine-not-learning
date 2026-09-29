#!/usr/bin/env python3
"""哨兵决策层 · 完整评估报告（技术交接用）

输出：分类报告 / 混淆矩阵 / AUC / 校准指标 / 阈值权衡 / 延迟分位数，
同时给出生成式基线对照。结果以 Markdown 表格打印，可直接贴进汇报材料。
"""
from __future__ import annotations

import csv
import time

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from sentinel_proto import (
    DATA, SEVERITY_LABELS, TYPES, encode, failure_type, severity,
    ece_binary, brier_binary,
)

BINS = 10
N_BOOT = 20


def load():
    with open(DATA, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def mce(p: np.ndarray, y: np.ndarray, bins: int = BINS) -> float:
    idx = np.clip((p * bins).astype(int), 0, bins - 1)
    worst = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum():
            worst = max(worst, abs(float(y[m].mean()) - float(p[m].mean())))
    return worst


def boot_balanced(p: np.ndarray, y: np.ndarray, rng, n: int = N_BOOT):
    """在 50/50 均衡子集上重采样评估。"""
    fail = np.where(y == 1)[0]
    ok = np.where(y == 0)[0]
    acc, bri, ec = [], [], []
    for _ in range(n):
        idx = np.concatenate([fail, rng.choice(ok, len(fail), replace=False)])
        pb, yb = p[idx], y[idx]
        acc.append(float(((pb >= 0.5).astype(int) == yb).mean()))
        bri.append(brier_binary(pb, yb))
        ec.append(ece_binary(pb, yb)[0])
    return np.mean(acc), np.std(acc), np.mean(bri), np.mean(ec)


def main() -> None:
    print("# 哨兵 EdgeSentinel · 决策层评估报告\n")
    print("> 硬件：NVIDIA DGX Spark（GB10）　数据：AI4I 2020（10000 条）　")
    print("> 划分：训练 60% / 校准 20% / 测试 20%（随机种子 20260921）\n")

    rows = load()
    X = np.vstack([encode(r) for r in rows])
    y_noul = np.array([int(r["Machine failure"]) for r in rows])
    y_choice = np.array([TYPES.index(failure_type(r)) for r in rows])
    y_score = np.array([severity(r) for r in rows])

    n = len(rows)
    rng = np.random.default_rng(20260921)
    perm = rng.permutation(n)
    i1, i2 = int(n * 0.6), int(n * 0.8)
    tr, ca, te = perm[:i1], perm[i1:i2], perm[i2:]

    sc = StandardScaler().fit(X[tr])
    Xtr, Xca, Xte = sc.transform(X[tr]), sc.transform(X[ca]), sc.transform(X[te])

    print("## 0. 样本分布\n")
    print("| 项目 | 数值 |")
    print("|---|---|")
    print(f"| 总样本 | {n} |")
    print(f"| 训练集 / 校准集 / 测试集 | {len(tr)} / {len(ca)} / {len(te)} |")
    print(f"| 故障样本占比 | {100 * y_noul.mean():.2f}%（{int(y_noul.sum())} 条） |")
    for i, t in enumerate(TYPES):
        print(f"| 类型 {t} | {int((y_choice == i).sum())} 条 |")
    for s in range(4):
        print(f"| 严重度 {s}（{SEVERITY_LABELS[s]}） | {int((y_score == s).sum())} 条 |")
    print()

    print("## 1. 训练配置\n")
    print("| 项目 | 配置 |")
    print("|---|---|")
    print("| 编码器 | 9 维特征：气温/过程温度/转速/扭矩/磨损 + 温升/机械功率/磨损-负载耦合/单位转速温升 |")
    print("| 决策头 | MLP，隐藏层 (64, 32)，ReLU，max_iter=800，random_state=20260921 |")
    print("| 校准方法 | 温度缩放（校准集上最小化 NLL，二元用 logit、多元用 log-proba 作 logits） |")
    print()

    head_n = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800,
                           random_state=20260921).fit(Xtr, y_noul[tr])
    head_c = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800,
                           random_state=20260921).fit(Xtr, y_choice[tr])
    head_s = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800,
                           random_state=20260921).fit(Xtr, y_score[tr])

    # ---- 校准 ----
    def fit_T_binary(z, y):
        best, bn = 1.0, float("inf")
        for t in np.linspace(0.1, 5.0, 200):
            pp = np.clip(1.0 / (1.0 + np.exp(-z / t)), 1e-9, 1 - 1e-9)
            v = -np.mean(y * np.log(pp) + (1 - y) * np.log(1 - pp))
            if v < bn:
                best, bn = float(t), v
        return best

    def fit_T_multi(Z, y):
        best, bn = 1.0, float("inf")
        for t in np.linspace(0.1, 5.0, 200):
            z = Z / t
            z -= z.max(1, keepdims=True)
            e = np.exp(z)
            p = e / e.sum(1, keepdims=True)
            v = -np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1.0)))
            if v < bn:
                best, bn = float(t), v
        return best

    p_raw = head_n.predict_proba(Xte)[:, 1]
    z_te = np.log(np.clip(p_raw, 1e-9, 1 - 1e-9) / np.clip(1 - p_raw, 1e-9, 1.0))
    p_ca = head_n.predict_proba(Xca)[:, 1]
    z_ca = np.log(np.clip(p_ca, 1e-9, 1 - 1e-9) / np.clip(1 - p_ca, 1e-9, 1.0))
    T = fit_T_binary(z_ca, y_noul[ca])
    p_cal = 1.0 / (1.0 + np.exp(-z_te / T))
    Tc = fit_T_multi(head_c.predict_log_proba(Xca), y_choice[ca])
    Ts = fit_T_multi(head_s.predict_log_proba(Xca), y_score[ca])

    def sm(Z, t):
        z = Z / t
        z -= z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    pc = sm(head_c.predict_log_proba(Xte), Tc)
    ps = sm(head_s.predict_log_proba(Xte), Ts)

    # ---- 2. Noul ----
    print("## 2. Noul：是否真实故障（二元判定）\n")
    pred = (p_cal >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_noul[te], pred).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0

    print("| 指标 | 数值 |")
    print("|---|---|")
    print(f"| Precision（精确率） | {prec:.4f} |")
    print(f"| Recall（召回率 / 灵敏度） | {rec:.4f} |")
    print(f"| F1 | {f1:.4f} |")
    print(f"| Specificity（特异度） | {spec:.4f} |")
    print(f"| Accuracy | {(tp + tn) / len(te):.4f} |")
    print(f"| Balanced Accuracy | {balanced_accuracy_score(y_noul[te], pred):.4f} |")
    print(f"| MCC（马修斯相关系数） | {matthews_corrcoef(y_noul[te], pred):.4f} |")
    print(f"| ROC-AUC | {roc_auc_score(y_noul[te], p_cal):.4f} |")
    print(f"| PR-AUC（平均精度） | {average_precision_score(y_noul[te], p_cal):.4f} |")
    print()
    print("**混淆矩阵（测试集）**\n")
    print("|  | 预测正常 | 预测故障 |")
    print("|---|---|---|")
    print(f"| 实际正常 | {tn} | {fp} |")
    print(f"| 实际故障 | {fn} | {tp} |")
    print()

    # ---- 3. 校准 ----
    print("## 3. 校准质量\n")
    print("| 指标 | 校准前 | 校准后 |")
    print("|---|---|---|")
    print(f"| 温度系数 T | 1.000 | {T:.3f} |")
    print(f"| ECE（10 分箱） | {ece_binary(p_raw, y_noul[te])[0]:.4f} | {ece_binary(p_cal, y_noul[te])[0]:.4f} |")
    print(f"| MCE（最大校准误差） | {mce(p_raw, y_noul[te]):.4f} | {mce(p_cal, y_noul[te]):.4f} |")
    print(f"| Brier Score | {brier_binary(p_raw, y_noul[te]):.4f} | {brier_binary(p_cal, y_noul[te]):.4f} |")
    print()
    print("**校准后可靠性曲线**\n")
    print("| 预测概率区间中点 | 实际发生率 | 样本数 |")
    print("|---|---|---|")
    _, curve = ece_binary(p_cal, y_noul[te])
    for c, a, k in curve:
        print(f"| {c:.3f} | {a:.3f} | {k} |")
    print()

    ba_mean, ba_std, ba_bri, ba_ece = boot_balanced(p_cal, y_noul[te], np.random.default_rng(11))
    print("**分布敏感性（同一模型，不同部署分布）**\n")
    print("| 评估分布 | Accuracy | ECE | Brier |")
    print("|---|---|---|---|")
    print(f"| 自然分布（3.4% 故障） | {(tp + tn) / len(te):.3f} | {ece_binary(p_cal, y_noul[te])[0]:.4f} | {brier_binary(p_cal, y_noul[te]):.4f} |")
    print(f"| 均衡分布（50% 故障，{N_BOOT} 次抽样） | {ba_mean:.3f} ± {ba_std:.3f} | {ba_ece:.4f} | {ba_bri:.4f} |")
    print("\n> 结论：校准与部署分布强相关，换分布需重新校准或做先验校正。\n")

    # ---- 4. Choice ----
    print("## 4. Choice：故障类型（6 类）\n")
    yp = pc.argmax(1)
    rep = classification_report(y_choice[te], yp,
                                target_names=TYPES, zero_division=0, digits=4,
                                output_dict=True)
    print("| 类别 | Precision | Recall | F1 | Support |")
    print("|---|---|---|---|---|")
    for t in TYPES:
        r = rep[t]
        print(f"| {t} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1-score']:.4f} | {int(r['support'])} |")
    print(f"| **宏平均** | {rep['macro avg']['precision']:.4f} | "
          f"{rep['macro avg']['recall']:.4f} | {rep['macro avg']['f1-score']:.4f} | {int(rep['macro avg']['support'])} |")
    print(f"| **加权平均** | {rep['weighted avg']['precision']:.4f} | "
          f"{rep['weighted avg']['recall']:.4f} | {rep['weighted avg']['f1-score']:.4f} | {int(rep['weighted avg']['support'])} |")
    print()
    print("**混淆矩阵**\n")
    cm = confusion_matrix(y_choice[te], yp)
    print("| 实际 \\ 预测 | " + " | ".join(TYPES) + " |")
    print("|---|" + "---|" * len(TYPES))
    for i, t in enumerate(TYPES):
        print(f"| {t} | " + " | ".join(str(v) for v in cm[i]) + " |")
    print("\n> 注意：少数类（TWF 46 条、RNF 18 条）样本极少，宏平均召回偏低属数据限制。\n")

    # ---- 5. Score ----
    print("## 5. Score：维护优先级（0~3，序数）\n")
    ys = ps.argmax(1)
    reps = classification_report(y_score[te], ys,
                                 target_names=[f"{i}·{SEVERITY_LABELS[i]}" for i in range(4)],
                                 zero_division=0, digits=4, output_dict=True)
    print("| 等级 | Precision | Recall | F1 | Support |")
    print("|---|---|---|---|---|")
    for i in range(4):
        k = f"{i}·{SEVERITY_LABELS[i]}"
        r = reps[k]
        print(f"| {k} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1-score']:.4f} | {int(r['support'])} |")
    print(f"| **宏平均** | {reps['macro avg']['precision']:.4f} | {reps['macro avg']['recall']:.4f} | "
          f"{reps['macro avg']['f1-score']:.4f} | {int(reps['macro avg']['support'])} |")
    print()
    qwk = cohen_kappa_score(y_score[te], ys, weights="quadratic")
    mae = float(np.mean(np.abs(ys - y_score[te])))
    within1 = float(np.mean(np.abs(ys - y_score[te]) <= 1))
    print("| 序数指标 | 数值 |")
    print("|---|---|")
    print(f"| QWK（二次加权 Kappa） | {qwk:.4f} |")
    print(f"| MAE（平均绝对误差） | {mae:.4f} |")
    print(f"| 误差 ≤1 级的比例 | {within1:.4f} |")
    print(f"| Accuracy | {float((ys == y_score[te]).mean()):.4f} |")
    print("\n> 注：0~3 为由磨损/温升/负载派生的维护优先级标签，非人工标注。\n")

    # ---- 6. 阈值权衡 ----
    print("## 6. 阈值 — 自动化率 — 准确率权衡（核心）\n")
    conf = np.maximum(p_cal, 1 - p_cal)
    print("| 阈值 | 自动化率 | 自动部分 Precision | 自动部分 Recall | 人工复核率 | 自动部分错误率 |")
    print("|---|---|---|---|---|---|")
    for thr in (0.60, 0.70, 0.80, 0.90, 0.95, 0.98):
        m = conf >= thr
        if m.sum() == 0:
            continue
        sub_p, sub_y = p_cal[m], y_noul[te][m]
        sp_ = (sub_p >= 0.5).astype(int)
        tp2 = int(((sp_ == 1) & (sub_y == 1)).sum())
        fp2 = int(((sp_ == 1) & (sub_y == 0)).sum())
        fn2 = int(((sp_ == 0) & (sub_y == 1)).sum())
        pr2 = tp2 / (tp2 + fp2) if tp2 + fp2 else 0.0
        rc2 = tp2 / (tp2 + fn2) if tp2 + fn2 else 0.0
        acc2 = float((sp_ == sub_y).mean())
        print(f"| {thr:.2f} | {m.mean():.1%} | {pr2:.4f} | {rc2:.4f} | "
              f"{1 - m.mean():.1%} | {1 - acc2:.4f} |")
    print()

    # ---- 7. 延迟 ----
    print("## 7. 延迟与吞吐\n")
    sample = Xte[:1000]
    lat = []
    for i in range(len(sample)):
        t0 = time.perf_counter()
        head_n.predict_proba(sample[i:i + 1])
        head_c.predict_proba(sample[i:i + 1])
        head_s.predict_proba(sample[i:i + 1])
        lat.append((time.perf_counter() - t0) * 1000)
    lat = np.array(lat)
    print("| 指标 | 数值 |")
    print("|---|---|")
    print(f"| p50 延迟 | {np.percentile(lat, 50):.3f} ms |")
    print(f"| p95 延迟 | {np.percentile(lat, 95):.3f} ms |")
    print(f"| p99 延迟 | {np.percentile(lat, 99):.3f} ms |")
    print(f"| 吞吐 | {1000 / np.mean(lat):.0f} 条/秒（三问同出） |")
    print(f"| 生成式基线（27B，关思考，单问） | 700 ms/条 |")
    print(f"| 生成式基线（27B，开思考，6 问 JSON） | 83700 ms/条 |")
    print(f"| 加速比（vs 开思考） | {83700 / np.mean(lat):.0f}x |")
    print()

    # ---- 8. 生成式对照 ----
    print("## 8. 与生成式自报置信度对照（均衡集，n=40）\n")
    print("| 指标 | 生成式 27B 自报 | 哨兵决策头 | 倍差 |")
    print("|---|---|---|---|")
    print(f"| 平均置信度 | 0.912 | {float(np.maximum(p_cal, 1 - p_cal).mean()):.3f} | — |")
    print(f"| 实际准确率 | 0.475 | {ba_mean:.3f} | {ba_mean / 0.475:.2f}x |")
    print(f"| ECE | 0.4375 | {ba_ece:.4f} | {0.4375 / ba_ece:.1f}x 更准 |")
    print(f"| Brier | 0.4325 | {ba_bri:.4f} | {0.4325 / ba_bri:.1f}x 更准 |")
    print(f"| 延迟 | 700 ms | {np.mean(lat):.3f} ms | {700 / np.mean(lat):.0f}x |")
    print()
    print("> 生成式模型自报 91.2% 置信度、实际仅 47.5% 正确，为典型的 RLHF 类")
    print("> 系统性过度自信；校准决策层正是为解决该问题而设计。\n")
    print("---\n")
    print("评估脚本：`full_eval.py`　运行：`cd ~/jev-service && python3 full_eval.py`")


if __name__ == "__main__":
    main()
