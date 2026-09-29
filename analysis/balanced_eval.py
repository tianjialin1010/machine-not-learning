#!/usr/bin/env python3
"""公平对比：在「均衡样本集」上评估哨兵决策头

gen_baseline 用的是 20 故障 / 20 正常的均衡集；
sentinel_proto 报的 97.8% 是自然分布（3.4% 故障）下的数字，两者不可直接比。
本脚本在同一类均衡集上重算决策头指标。
"""
from __future__ import annotations

import csv

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from sentinel_proto import DATA, encode, ece_binary, brier_binary

REPEAT = 20  # 重复抽样次数，报告均值±标准差


def main() -> None:
    with open(DATA, encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    X = np.vstack([encode(r) for r in rows])
    y = np.array([int(r["Machine failure"]) for r in rows])

    n = len(rows)
    rng = np.random.default_rng(20260921)
    perm = rng.permutation(n)
    i1, i2 = int(n * 0.6), int(n * 0.8)
    tr, ca, te = perm[:i1], perm[i1:i2], perm[i2:]

    scaler = StandardScaler().fit(X[tr])
    Xtr, Xca, Xte = scaler.transform(X[tr]), scaler.transform(X[ca]), scaler.transform(X[te])
    head = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800,
                         random_state=20260921).fit(Xtr, y[tr])

    # 温度缩放（同 sentinel_proto）
    p_ca = head.predict_proba(Xca)[:, 1]
    z_ca = np.log(np.clip(p_ca, 1e-9, 1 - 1e-9) / np.clip(1 - p_ca, 1e-9, 1.0))
    best_t, best_nll = 1.0, float("inf")
    for t in np.linspace(0.1, 5.0, 200):
        pp = 1.0 / (1.0 + np.exp(-z_ca / t))
        pp = np.clip(pp, 1e-9, 1 - 1e-9)
        nll = -np.mean(y[ca] * np.log(pp) + (1 - y[ca]) * np.log(1 - pp))
        if nll < best_nll:
            best_t, best_nll = float(t), float(nll)

    p_te_raw = head.predict_proba(Xte)[:, 1]
    z_te = np.log(np.clip(p_te_raw, 1e-9, 1 - 1e-9) / np.clip(1 - p_te_raw, 1e-9, 1.0))
    p_te = 1.0 / (1.0 + np.exp(-z_te / best_t))

    y_te = y[te]
    fail_idx = np.where(y_te == 1)[0]
    ok_idx = np.where(y_te == 0)[0]
    print("=" * 62)
    print("公平对比：均衡样本集上的哨兵决策头")
    print("=" * 62)
    print(f"\n测试集中故障样本 {len(fail_idx)} 条，正常样本 {len(ok_idx)} 条")
    print(f"温度系数 T = {best_t:.3f}\n")

    accs, eces, briers = [], [], []
    sub_rng = np.random.default_rng(11)
    for _ in range(REPEAT):
        pick_ok = sub_rng.choice(ok_idx, len(fail_idx), replace=False)
        idx = np.concatenate([fail_idx, pick_ok])
        pb, yb = p_te[idx], y_te[idx]
        accs.append(float(((pb >= 0.5).astype(int) == yb).mean()))
        eces.append(ece_binary(pb, yb)[0])
        briers.append(brier_binary(pb, yb))

    def stat(v: list) -> str:
        return f"{np.mean(v):.3f} ± {np.std(v):.3f}"

    print(f"  {REPEAT} 次均衡抽样（每次 {2 * len(fail_idx)} 条，50/50）")
    print(f"  准确率 : {stat(accs)}")
    print(f"  ECE    : {stat(eces)}")
    print(f"  Brier  : {stat(briers)}")

    print("\n" + "=" * 62)
    print("与生成式自报置信度对比（同为均衡集）")
    print("=" * 62)
    print("                        生成式 27B        哨兵决策头")
    idx0 = np.concatenate([fail_idx, np.random.default_rng(11).choice(ok_idx, len(fail_idx), replace=False)])
    mean_conf = float(np.maximum(p_te[idx0], 1 - p_te[idx0]).mean())
    print(f"  实际准确率             0.475            {np.mean(accs):.3f}")
    print(f"  ECE                    0.4375           {np.mean(eces):.4f}")
    print(f"  Brier                  0.4325           {np.mean(briers):.4f}")
    print(f"  延迟                   0.7 s/条         0.098 ms/条")


if __name__ == "__main__":
    main()
