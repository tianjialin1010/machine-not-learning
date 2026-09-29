#!/usr/bin/env python3
"""对照实验：生成式模型「自报置信度」的校准度 vs 我们的校准决策头

Jev / RLCD 的核心论点：RLHF 模型口头报的置信度系统性过度自信。
本脚本在同一批样本上测量本地 27B 自报概率的 ECE / Brier，
与哨兵决策头（温度缩放校准后）对比。
"""
from __future__ import annotations

import csv
import json
import time
import urllib.request

import numpy as np

DATA = "/home/USER/jev-service/data/ai4i2020.csv"
OLLAMA = "http://127.0.0.1:11434/api/chat"
MODEL = "modelscope.cn/unsloth/Qwen3.8-27B-GGUF:latest"
N_FAIL, N_OK = 20, 20


def load_rows() -> list[dict[str, str]]:
    with open(DATA, encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def describe(row: dict[str, str]) -> str:
    return (
        f"设备型号 {row['Type']}："
        f"气温 {float(row['Air temperature [K]']) - 273.15:.1f}℃，"
        f"过程温度 {float(row['Process temperature [K]']) - 273.15:.1f}℃，"
        f"转速 {row['Rotational speed [rpm]']} rpm，"
        f"扭矩 {row['Torque [Nm]']} Nm，"
        f"刀具磨损 {row['Tool wear [min]']} min。"
    )


def ask(prob_prompt: str) -> float | None:
    """调用 ollama 原生接口。

    注意：该 27B 为带思考链的模型，必须显式关闭思考（think=False），
    否则生成的 token 会被推理过程耗尽，返回空内容。
    """
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prob_prompt}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_predict": 24},
    }
    req = urllib.request.Request(
        OLLAMA,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        print("  请求失败:", exc)
        return None
    text = str(body.get("message", {}).get("content", "")).strip()
    if not text:
        return None
    for token in text.replace("%", "").replace("。", " ").split():
        try:
            value = float(token)
        except ValueError:
            continue
        if value > 1.0:
            value = value / 100.0
        if 0.0 <= value <= 1.0:
            return value
    return None


def ece(p: np.ndarray, y: np.ndarray, bins: int = 5) -> tuple[float, list]:
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


def main() -> None:
    rows = load_rows()
    failures = [r for r in rows if int(r["Machine failure"]) == 1]
    healthy = [r for r in rows if int(r["Machine failure"]) == 0]
    rng = np.random.default_rng(7)
    sample = ([failures[i] for i in rng.choice(len(failures), N_FAIL, replace=False)]
              + [healthy[i] for i in rng.choice(len(healthy), N_OK, replace=False)])
    y = np.array([1] * N_FAIL + [0] * N_OK)

    print("=" * 62)
    print("对照实验：生成式自报置信度 vs 校准决策头")
    print("=" * 62)
    print(f"\n样本：{len(sample)} 条（故障 {N_FAIL} / 正常 {N_OK}）")

    probs: list[float] = []
    t0 = time.perf_counter()
    for i, row in enumerate(sample, 1):
        prompt = (
            "你是工业设备诊断专家。以下是某台设备的实时传感器读数。\n"
            f"{describe(row)}\n"
            "请判断该设备当前处于故障状态的概率。"
            "只回复一个 0 到 1 之间的小数，不要任何解释。"
        )
        value = ask(prompt)
        probs.append(0.5 if value is None else value)
        print(f"  [{i:2d}/{len(sample)}] 真值={y[i-1]}  模型自报={probs[-1]:.3f}")
    elapsed = time.perf_counter() - t0

    p = np.array(probs)
    ece_val, curve = ece(p, y)
    brier = float(np.mean((p - y) ** 2))
    acc = float(((p >= 0.5).astype(int) == y).mean())
    mean_conf = float(np.maximum(p, 1 - p).mean())

    print("\n" + "=" * 62)
    print("生成式自报置信度（本地 27B，n=%d）" % len(sample))
    print("=" * 62)
    print(f"  平均置信度 : {mean_conf:.3f}")
    print(f"  实际准确率 : {acc:.3f}")
    print(f"  ECE        : {ece_val:.4f}")
    print(f"  Brier      : {brier:.4f}")
    print(f"  总耗时     : {elapsed:.0f} s  （{elapsed / len(sample):.1f} s/条）")
    print("\n  可靠性曲线（自报概率 vs 实际发生率）：")
    print("    自报   实际   样本数")
    for conf, a, n in curve:
        print(f"    {conf:<6.3f} {a:<6.3f} {n}")

    print("\n  参考：同一模型开启思考链、输出 6 问结构化 JSON 时（经 jev 服务）为 83.7 s/条")
    print("\n对照：哨兵决策头（MLP + 温度缩放，测试集 n=2000）")
    print("  ECE 0.0066 | Brier 0.0156 | 0.098 ms/条")
    print("\n结论要点：若生成式的平均置信度显著高于实际准确率，")
    print("即为 RLHF 类模型的系统性过度自信，正是校准决策层要解决的问题。")


if __name__ == "__main__":
    main()
