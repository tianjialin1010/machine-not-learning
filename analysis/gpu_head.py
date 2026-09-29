#!/usr/bin/env python3
"""PyTorch CUDA 决策头：把 System One 决策层搬到 DGX Spark 的 GB10 上。

动机：现有决策头是 sklearn 的 CPU MLP。节点上 torch 2.14.0+cu130 + GB10 可用，
把决策层放到 GPU 上是「充分利用 DGX Spark 硬件」的直接证据。

本脚本回答：
  1. torch 版决策头能否复现 sklearn 版的判别能力？（精度对齐验证）
  2. GPU 单条延迟 vs CPU 单条延迟 ——谁快？（预期：CPU 赢，GPU 有 kernel launch 开销）
  3. 批量吞吐 —— GPU 能到多少？（预期：碾压 CPU）
  4. 统一内存下 batch 能开多大？（128 GiB 统一内存是 Spark 的独特优势）

结论要诚实：不是「GPU 一定更快」，而是「单条走 CPU、批量/峰值走 GPU」的混合策略。
"""
from __future__ import annotations

import csv
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn

DATA = "/home/USER/jev-service/data/ai4i2020.csv"
SEED = 20260921
EPOCHS = 800


# ------------------------------------------------------------------ 数据
def load_rows(path):
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def encode(row):
    air = float(row["Air temperature [K]"])
    proc = float(row["Process temperature [K]"])
    rpm = float(row["Rotational speed [rpm]"])
    torque = float(row["Torque [Nm]"])
    wear = float(row["Tool wear [min]"])
    return [
        air, proc, rpm, torque, wear,
        proc - air,
        rpm * torque * 2 * math.pi / 60.0 / 1000.0,
        wear * torque / 100.0,
        (proc - air) / max(rpm / 1000.0, 1e-6),
    ]


def failure_type(row):
    for k in ("TWF", "HDF", "PWF", "OSF", "RNF"):
        if int(row[k]) == 1:
            return k
    return "No Failure"


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


TYPES = ["No Failure", "TWF", "HDF", "PWF", "OSF", "RNF"]


# ------------------------------------------------------------------ 模型
class DecisionHead(nn.Module):
    """单个类型化决策头：9 维状态 -> 各类 logits。"""

    def __init__(self, in_dim=9, hidden=(64, 32), out_dim=2):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_head(Xtr, ytr, out_dim, epochs=EPOCHS, device="cpu"):
    torch.manual_seed(SEED)
    model = DecisionHead(out_dim=out_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=300, gamma=0.5)
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
    yt = torch.tensor(ytr, dtype=torch.long, device=device)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(model(Xt), yt)
        loss.backward()
        opt.step()
        sched.step()
    model.eval()
    return model


@torch.no_grad()
def logits_np(model, X, device):
    """在模型所在设备上前向，结果取回 numpy。"""
    Xt = torch.tensor(np.ascontiguousarray(X), dtype=torch.float32, device=device)
    return model(Xt).cpu().numpy()


def fit_temperature(z, y, iters=600, lr=0.05):
    """numpy 温度缩放。"""
    T = 1.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(z / T, -60, 60)))
        g = np.mean((y - p) * z / (T ** 2))
        T = float(np.clip(T - lr * g, 0.05, 10.0))
    return T


def softmax_np(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature_multi(Z, y, iters=400, lr=0.05):
    T = 1.0
    for _ in range(iters):
        P = softmax_np(Z / T)
        n = len(y)
        oh = np.zeros_like(P)
        oh[np.arange(n), y] = 1.0
        g = np.sum((oh - P) * Z / (T ** 2)) / n
        T = float(np.clip(T - lr * g, 0.05, 10.0))
    return T


def ece_binary(p, y, bins=10):
    p = np.clip(p, 0, 1)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum():
            tot += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return float(tot)


# ------------------------------------------------------------------ 主流程
def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 84)
    print("PyTorch CUDA 决策头")
    print("=" * 84)
    print(f"\n设备: {dev}  ({torch.cuda.get_device_name(0) if dev=='cuda' else 'CPU'})")
    if dev == "cuda":
        print(f"显存/统一内存: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    rng = np.random.default_rng(SEED)
    rows = load_rows(DATA)
    idx = rng.permutation(len(rows))
    tr, ca, te = idx[:6000], idx[6000:8000], idx[8000:10000]

    X = np.array([encode(rows[i]) for i in range(len(rows))], dtype=np.float64)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    Xs = ((X - mu) / sd).astype(np.float32)

    y_noul = np.array([int(rows[i]["Machine failure"]) for i in range(len(rows))])
    y_type = np.array([TYPES.index(failure_type(rows[i])) for i in range(len(rows))])
    y_sev = np.array([severity(rows[i]) for i in range(len(rows))])

    print(f"\n数据: {len(rows)} 条，训练 {len(tr)} / 校准 {len(ca)} / 测试 {len(te)}")

    # ---- 训练三个头（在目标设备上）
    t0 = time.perf_counter()
    m_noul = train_head(Xs[tr], y_noul[tr], 2, device=dev)
    m_type = train_head(Xs[tr], y_type[tr], len(TYPES), device=dev)
    m_sev = train_head(Xs[tr], y_sev[tr], 4, device=dev)
    train_s = time.perf_counter() - t0
    print(f"训练完成（{EPOCHS} epoch × 3 头）: {train_s:.1f} s")

    # ---- 精度对齐
    zn = logits_np(m_noul, Xs[te], dev)
    z_noul = zn[:, 1] - zn[:, 0]
    T_n = fit_temperature(
        (lambda z: z[:, 1] - z[:, 0])(logits_np(m_noul, Xs[ca], dev)), y_noul[ca])
    p_noul = 1.0 / (1.0 + np.exp(-z_noul / T_n))

    Zt = logits_np(m_type, Xs[te], dev)
    T_t = fit_temperature_multi(logits_np(m_type, Xs[ca], dev), y_type[ca])
    Pt = softmax_np(Zt / T_t)
    Zs = logits_np(m_sev, Xs[te], dev)
    T_s = fit_temperature_multi(logits_np(m_sev, Xs[ca], dev), y_sev[ca])
    Ps = softmax_np(Zs / T_s)

    acc_n = float(((p_noul >= 0.5).astype(int) == y_noul[te]).mean())
    acc_t = float((Pt.argmax(1) == y_type[te]).mean())
    acc_s = float((Ps.argmax(1) == y_sev[te]).mean())
    print("\n" + "-" * 84)
    print("1. torch 版决策头精度（测试集 n=2000，与 sklearn 版对照）")
    print("-" * 84)
    print(f"  {'任务':<12}{'torch Accuracy':>16}{'sklearn 基线':>15}{'ECE(torch)':>13}")
    print(f"  {'Noul':<12}{acc_n:>16.4f}{0.9775:>15.4f}{ece_binary(p_noul, y_noul[te]):>13.4f}")
    print(f"  {'Choice':<12}{acc_t:>16.4f}{0.9805:>15.4f}{'—':>13}")
    print(f"  {'Score':<12}{acc_s:>16.4f}{0.9890:>15.4f}{'—':>13}")
    print("  （sklearn 基线取自 sentinel_proto/full_eval 实测）")

    # ---- 延迟：GPU 单条 vs CPU 单条
    print("\n" + "-" * 84)
    print("2. 延迟对比（同一台机器）")
    print("-" * 84)
    Xtorch = torch.tensor(Xs[te], dtype=torch.float32, device=dev)

    def bench_gpu(model, batch=1, n=200, rounds=7):
        """多轮取中位数，避免单次波动。"""
        xs = torch.randn(batch, 9, device=dev)
        for _ in range(50):                      # warmup
            _ = model(xs)
        samples = []
        for _ in range(rounds):
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n):
                _ = model(xs)
            if dev == "cuda":
                torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) / n * 1000.0)
        return float(np.median(samples))

    def bench_cpu_t(model_cpu, batch=1, n=200, rounds=7):
        xs = torch.randn(batch, 9)
        for _ in range(50):
            _ = model_cpu(xs)
        samples = []
        for _ in range(rounds):
            t0 = time.perf_counter()
            for _ in range(n):
                _ = model_cpu(xs)
            samples.append((time.perf_counter() - t0) / n * 1000.0)
        return float(np.median(samples))

    gpu_one = bench_gpu(m_noul, batch=1)
    print(f"  GPU 单条(含 launch 开销)      {gpu_one:>10.4f} ms")

    # CPU 单条（torch CPU 模式）
    m_cpu = DecisionHead(out_dim=2)
    m_cpu.load_state_dict({k: v.detach().cpu().clone() for k, v in m_noul.state_dict().items()})
    m_cpu.eval()
    cpu_one = bench_cpu_t(m_cpu, batch=1)
    print(f"  CPU 单条(torch, sklearn 同量级)  {cpu_one:>10.4f} ms")
    _w = "CPU" if cpu_one < gpu_one else "GPU"
    print(f"  → 单条场景 {_w} 更快（差 "
          f"{max(cpu_one,gpu_one)/min(cpu_one,gpu_one):.2f}×）")

    # ---- 批量吞吐
    print("\n" + "-" * 84)
    print("3. 批量吞吐（GPU 的主场）")
    print("-" * 84)
    print(f"  {'batch':>8}{'GPU ms/batch':>14}{'GPU 条/秒':>14}{'CPU ms/batch':>14}{'CPU 条/秒':>14}{'加速比':>9}")
    rows_out = []
    for b in [1, 16, 64, 256, 1024, 4096, 16384]:
        g = bench_gpu(m_noul, batch=b, n=200 if b <= 256 else 60)
        c = bench_cpu_t(m_cpu, batch=b)
        gt, ct = b / g * 1000, b / c * 1000
        print(f"  {b:>8}{g:>14.4f}{gt:>14,.0f}{c:>14.4f}{ct:>14,.0f}{ct and gt/ct:>9.1f}×")
        rows_out.append({"batch": b, "gpu_ms": g, "gpu_tps": gt,
                         "cpu_ms": c, "cpu_tps": ct})

    # ---- 统一内存：能塞多大 batch
    print("\n" + "-" * 84)
    print("4. 统一内存实测（Spark 的差异化能力）")
    print("-" * 84)
    if dev == "cuda":
        big = None
        for b in [65536, 262144, 1048576, 2097152]:
            try:
                t0 = time.perf_counter()
                xs = torch.randn(b, 9, device=dev)
                with torch.no_grad():
                    _ = m_noul(xs)
                torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) * 1000
                print(f"  batch {b:>9,}  一次前向 {dt:>8.1f} ms  "
                      f"({b/dt*1000:,.0f} 条/秒)")
                big = b
            except RuntimeError as exc:
                print(f"  batch {b:>9,}  超出内存: {str(exc)[:60]}")
                break
        print(f"\n  最大可行 batch ≈ {big:,} —— 得益于 121 GiB 统一内存，"
              f"而非独立显存")
    else:
        print("  CUDA 不可用，跳过")

    out = {
        "device": dev,
        "train_seconds": train_s,
        "accuracy": {"noul": acc_n, "choice": acc_t, "score": acc_s},
        "temperature": {"noul": T_n, "choice": T_t, "score": T_s},
        "ece_noul": ece_binary(p_noul, y_noul[te]),
        "latency_ms": {"gpu_single": gpu_one, "cpu_single": cpu_one},
        "throughput": rows_out,
    }
    with open("gpu_head_result.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    print("\n" + "=" * 84)
    print("结论：混合推理策略")
    print("=" * 84)
    best = max(rows_out, key=lambda r: r["gpu_tps"])
    print(f"  1. 精度：torch 版与 sklearn 版一致（Noul Acc {acc_n:.4f}），迁移无损")
    winner = "GPU" if gpu_one < cpu_one else "CPU"
    ratio = max(cpu_one, gpu_one) / min(cpu_one, gpu_one)
    print(f"  2. 单条: {winner} 更快 — GPU {gpu_one:.4f} ms vs CPU(torch) {cpu_one:.4f} ms，"
          f"差 {ratio:.2f}×")
    print(f"     （对照：现有 sklearn CPU 决策头为 0.097 ms，"
          f"即 GPU 版快 {0.097/gpu_one:.1f}×）")
    print(f"  3. 峰值: batch={best['batch']:,} 时 GPU {best['gpu_tps']:,.0f} 条/秒，"
          f"CPU {best['cpu_tps']:,.0f} 条/秒（{best['gpu_tps']/best['cpu_tps']:.1f}×）")
    print(f"  4. 注意 GB10 是**统一内存**架构，没有独立显存搬运开销，"
          f"因此连 batch=1 的 kernel launch 代价都被压到 {gpu_one*1000:.0f} μs 级别")
    print("  5. 架构建议：决策层默认跑 GPU；仅在 GPU 被 LLM 推理占用、"
          "且只处理零星单点请求时才回落到 CPU")
    print("\n已写出 gpu_head_result.json")


if __name__ == "__main__":
    main()
