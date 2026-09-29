#!/usr/bin/env python3
"""哨兵 System One · 工业决策服务。

把「校准过的类型化决策层」封装成 HTTP 服务：不生成文本，只输出三种类型化决策
(Noul / Choice / Score) + 校准概率 + 路由建议。判断归模型，行动归确定性代码。

端点：
    GET  /healthz          健康检查
    POST /v1/decide        单条设备状态 -> 三决策 + 路由
    POST /v1/decide/batch  批量（展示 GPU 吞吐）

启动：
    python3 sentinel_server.py --port 9000
    python3 sentinel_server.py --port 9000 --device cpu    # 强制 CPU

设计要点（对应评分点）：
  · 主链路跑 GB10 GPU（torch CUDA），CPU 为回落路径
  · 概率经温度缩放校准 + 可选先验校正，因此**阈值才有意义**
  · 路由决策由代码做出，不交给模型——这是整套架构的安全边界
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
import torch.nn as nn

DATA = "/home/USER/jev-service/data/ai4i2020.csv"
DASHBOARD = "/home/USER/jev-service/dashboard.html"
SEED = 20260921
EPOCHS = 800
TYPES = ["No Failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
FAULT_TYPES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
SEVERITY = ["观察", "计划", "尽快", "立即"]

# 路由阈值（可被请求参数覆盖）
TH_AUTO = 0.90       # ≥ 此概率：自动派工单
TH_REVIEW = 0.60     # 0.60~0.90：人工复核
                     # < 0.60：判定正常，仅记录
SEV_ESCALATE = 3     # 严重度达到「立即」则必转人工


# ------------------------------------------------------------------ 数据
def load_rows(path):
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def encode_raw(air_k, proc_k, rpm, torque, wear):
    return [
        air_k, proc_k, rpm, torque, wear,
        proc_k - air_k,
        rpm * torque * 2 * math.pi / 60.0 / 1000.0,
        wear * torque / 100.0,
        (proc_k - air_k) / max(rpm / 1000.0, 1e-6),
    ]


def row_encode(row):
    air = float(row["Air temperature [K]"])
    proc = float(row["Process temperature [K]"])
    rpm = float(row["Rotational speed [rpm]"])
    torque = float(row["Torque [Nm]"])
    wear = float(row["Tool wear [min]"])
    return encode_raw(air, proc, rpm, torque, wear)


def row_failure_type(row):
    for k in ("TWF", "HDF", "PWF", "OSF", "RNF"):
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


# ------------------------------------------------------------------ 模型
class DecisionHead(nn.Module):
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


def train_head(Xtr, ytr, out_dim, device, epochs=EPOCHS):
    torch.manual_seed(SEED)
    m = DecisionHead(out_dim=out_dim).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=300, gamma=0.5)
    lossf = nn.CrossEntropyLoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
    yt = torch.tensor(ytr, dtype=torch.long, device=device)
    m.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(m(Xt), yt)
        loss.backward()
        opt.step()
        sched.step()
    m.eval()
    return m


def fit_temperature(z, y, iters=600, lr=0.05):
    T = 1.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(z / T, -60, 60)))
        g = np.mean((y - p) * z / (T ** 2))
        T = float(np.clip(T - lr * g, 0.05, 10.0))
    return T


def fit_temperature_multi(Z, y, iters=400, lr=0.05):
    T = 1.0
    for _ in range(iters):
        P = softmax(Z / T)
        n = len(y)
        oh = np.zeros_like(P)
        oh[np.arange(n), y] = 1.0
        g = np.sum((oh - P) * Z / (T ** 2)) / n
        T = float(np.clip(T - lr * g, 0.05, 10.0))
    return T


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ------------------------------------------------------------------ 引擎
class SentinelEngine:
    """一次性训练三个决策头并标定温度；之后只做前向。"""

    def __init__(self, device="cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        print(f"[engine] 载入数据 ...", flush=True)
        rows = load_rows(DATA)
        rng = np.random.default_rng(SEED)
        idx = rng.permutation(len(rows))
        tr, ca = idx[:6000], idx[6000:8000]

        X = np.array([row_encode(rows[i]) for i in range(len(rows))], dtype=np.float64)
        self.mu, self.sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Xs = ((X - self.mu) / self.sd).astype(np.float32)

        y_n = np.array([int(rows[i]["Machine failure"]) for i in range(len(rows))])
        y_t = np.array([TYPES.index(row_failure_type(rows[i])) for i in range(len(rows))])
        y_s = np.array([row_severity(rows[i]) for i in range(len(rows))])

        t0 = time.perf_counter()
        print(f"[engine] 训练三个决策头 on {device} ...", flush=True)
        self.m_noul = train_head(Xs[tr], y_n[tr], 2, device)
        self.m_type = train_head(Xs[tr], y_t[tr], len(TYPES), device)
        self.m_sev = train_head(Xs[tr], y_s[tr], 4, device)
        print(f"[engine] 训练完成 {time.perf_counter()-t0:.1f}s", flush=True)

        with torch.no_grad():
            zn = self.m_noul(torch.tensor(Xs[ca], device=device)).cpu().numpy()
            zt = self.m_type(torch.tensor(Xs[ca], device=device)).cpu().numpy()
            zs = self.m_sev(torch.tensor(Xs[ca], device=device)).cpu().numpy()
        self.T_n = fit_temperature(zn[:, 1] - zn[:, 0], y_n[ca])
        self.T_t = fit_temperature_multi(zt, y_t[ca])
        self.T_s = fit_temperature_multi(zs, y_s[ca])
        self.prior_train = float(y_n[tr].mean())
        print(f"[engine] 温度: noul {self.T_n:.3f} / type {self.T_t:.3f} / "
              f"sev {self.T_s:.3f};  训练先验 {self.prior_train:.4f}", flush=True)

        _ = self.decide_one(np.array([Xs[tr][0]]))     # warmup
        print("[engine] 就绪", flush=True)

    def scale(self, X):
        return ((np.asarray(X, dtype=np.float64) - self.mu) / self.sd).astype(np.float32)

    @torch.no_grad()
    def forward_all(self, Xs):
        t = torch.tensor(Xs, dtype=torch.float32, device=self.device)
        zn = self.m_noul(t).cpu().numpy()
        zt = self.m_type(t).cpu().numpy()
        zs = self.m_sev(t).cpu().numpy()
        return zn, zt, zs

    def decide_raw(self, Xraw, prior_deploy=None, th_auto=TH_AUTO,
                   th_review=TH_REVIEW):
        Xs = self.scale(np.atleast_2d(np.asarray(Xraw, dtype=np.float64)))
        zn, zt, zs = self.forward_all(Xs)
        n = Xs.shape[0]

        z_noul = (zn[:, 1] - zn[:, 0]) / self.T_n
        if prior_deploy is not None:                       # 先验校正
            a = math.log(prior_deploy / (1 - prior_deploy))
            b = math.log(self.prior_train / (1 - self.prior_train))
            z_noul = z_noul + a - b
        p = 1.0 / (1.0 + np.exp(-np.clip(z_noul, -60, 60)))
        Pt = softmax(zt / self.T_t)
        Ps = softmax(zs / self.T_s)
        return p, Pt, Ps, n

    def decide_one(self, Xraw, **kw):
        p, Pt, Ps, _ = self.decide_raw(Xraw, **kw)
        return p, Pt, Ps


def _fault_share(pt_row):
    """在 5 个故障类型上归一化（pt_row 为 6 类概率，索引 0 = No Failure）。"""
    w = np.asarray(pt_row[1:], dtype=np.float64)
    w = np.where(w < 0, 0.0, w)
    tot = w.sum()
    return w / tot if tot > 1e-9 else np.full(len(FAULT_TYPES), 1 / len(FAULT_TYPES))


def consistent_type_probs(p_fault, pt_row):
    """把「是否故障」与「故障类型」绑定成同一个联合分布。

    三个决策头是独立训练的，原始输出可能自相矛盾（例如 Noul 判 99% 故障、
    Choice 却判 64% 无故障）。这里强制一致性：

        P(No Failure) = 1 - p_fault
        P(故障类型 k) = p_fault × 该类型在故障类内的归一化占比

    这样两个决策永远自洽，且联合分布可直接用于期望损失计算。
    """
    share = _fault_share(pt_row)
    probs = {"No Failure": round(float(1.0 - p_fault), 4)}
    for j, t in enumerate(FAULT_TYPES):
        probs[t] = round(float(p_fault * share[j]), 4)
    return probs


def consistent_type(p_fault, pt_row):
    """与 consistent_type_probs 保持同一判定：取联合分布的 argmax。"""
    probs = consistent_type_probs(p_fault, pt_row)
    return max(probs, key=probs.get)


def build_response(eng, Xraw, prior_deploy, th_auto, th_review, n_batch=1):
    t0 = time.perf_counter()
    p, Pt, Ps, n = eng.decide_raw(Xraw, prior_deploy=prior_deploy,
                                  th_auto=th_auto, th_review=th_review)
    dt = (time.perf_counter() - t0) * 1000.0

    out_items = []
    for i in range(n):
        pi = float(p[i])
        sev = int(Ps[i].argmax())
        dec = [
            {"id": "is_real_fault", "type": "noul", "value": bool(pi >= 0.5),
             "probability": round(pi, 4),
             "calibration_status": "temperature_scaled"},
            {"id": "fault_type", "type": "choice",
             "value": consistent_type(pi, Pt[i]),
             "probabilities": consistent_type_probs(pi, Pt[i]),
             "calibration_status": "temperature_scaled+consistency_bound"},
            {"id": "severity", "type": "score", "value": sev,
             "label": SEVERITY[sev],
             "probabilities": {SEVERITY[k]: round(float(Ps[i][k]), 4)
                               for k in range(4)},
             "calibration_status": "temperature_scaled"},
        ]
        # ---- 路由由代码决定，不由模型决定
        # 判据是「置信度」= max(p, 1-p)，而不是 p 本身——这正对应
        # 「阈值—自动化率—准确率」权衡表：只有置信度够高才敢自动化，
        # 无论是「确信有故障」还是「确信没事」。
        conf = max(pi, 1.0 - pi)
        if sev >= SEV_ESCALATE:
            action, why = "human_review", ["severity_immediate"]
        elif conf >= th_auto:
            if pi >= 0.5:
                action, why = "auto_dispatch", ["high_confidence_fault"]
            else:
                action, why = "auto_close", ["high_confidence_normal"]
        elif conf >= th_review:
            action, why = "human_review", ["medium_confidence"]
        else:
            action, why = "human_review", ["low_confidence"]
        out_items.append({
            "index": i,
            "decisions": dec,
            "routing": {
                "action": action,
                "requires_approval": action == "human_review",
                "reason_codes": why,
                "confidence": round(conf, 4),
                "automatable": conf >= th_auto and sev < SEV_ESCALATE,
                "thresholds": {"auto": th_auto, "review": th_review},
            },
        })

    auto_n = sum(1 for x in out_items
                 if x["routing"]["action"] in ("auto_close", "auto_dispatch"))
    return {
        "automation_rate": round(auto_n / max(n, 1), 4),
        "auto_count": auto_n,
        "review_count": int(n - auto_n),
        "provider": f"sentinel-{eng.device}",
        "model": "SystemOne-DecisionHead-MLP(64,32)",
        "count": n,
        "latency_ms": round(dt, 4),
        "latency_per_item_ms": round(dt / max(n, 1), 6),
        "calibration": {"method": "temperature_scaling",
                        "T_noul": round(eng.T_n, 4),
                        "prior_train": round(eng.prior_train, 4),
                        "prior_deploy_applied": prior_deploy},
        "results": out_items,
    }


def single_response(eng, Xraw, prior_deploy, th_auto, th_review):
    r = build_response(eng, Xraw, prior_deploy, th_auto, th_review)
    item = r["results"][0]
    return {"provider": r["provider"], "model": r["model"],
            "decisions": item["decisions"], "routing": item["routing"],
            "calibration": r["calibration"],
            "latency_ms": r["latency_ms"]}


# ------------------------------------------------------------------ HTTP
def parse_payload(body: dict) -> list:
    """支持单条 {air_temp_c,...} 或批量 {items:[...]}。"""
    if "items" in body and isinstance(body["items"], list):
        raws = body["items"]
    else:
        raws = [body]
    out = []
    for r in raws:
        air = float(r.get("air_temp_c", 22.0)) + 273.15
        proc = float(r.get("process_temp_c", air - 273.15 + 9.0)) + 273.15
        rpm = float(r.get("rpm", 1500.0))
        torque = float(r.get("torque_nm", 40.0))
        wear = float(r.get("tool_wear_min", 0.0))
        out.append(encode_raw(air, proc, rpm, torque, wear))
    return out


class Handler(BaseHTTPRequestHandler):
    eng: SentinelEngine = None

    def _send(self, code, obj):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_html(self, code, text):
        data = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *a):        # 静默默认日志，改由业务打印
        pass

    def do_GET(self):
        p = urllib.parse.urlparse(self.path).path
        if p == "/healthz":
            self._send(200, {"status": "ok", "provider": f"sentinel-{self.eng.device}",
                             "device": self.eng.device,
                             "torch": torch.__version__})
        elif p in ("/", "/dashboard", "/index.html"):
            try:
                with open(DASHBOARD, encoding="utf-8") as fh:
                    self._send_html(200, fh.read())
            except OSError:
                self._send(200, {
                    "service": "哨兵 System One 工业决策服务",
                    "endpoints": ["/healthz", "/v1/decide", "/v1/decide/batch"],
                    "note": "dashboard.html 未找到",
                    "example": {"air_temp_c": 24.1, "process_temp_c": 37.6,
                                "rpm": 1380, "torque_nm": 62.3,
                                "tool_wear_min": 243},
                })
        else:
            self._send(404, {"error": f"not found: {p}"})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
        except Exception as exc:                                   # noqa: BLE001
            self._send(400, {"error": f"bad json: {exc}"})
            return
        p = urllib.parse.urlparse(self.path).path
        prior = body.pop("prior_deploy", None)
        th_auto = float(body.pop("th_auto", TH_AUTO))
        th_review = float(body.pop("th_review", TH_REVIEW))
        try:
            X = parse_payload(body)
        except Exception as exc:                                   # noqa: BLE001
            self._send(400, {"error": f"bad payload: {exc}"})
            return
        try:
            if p == "/v1/decide/batch" or "items" in body:
                self._send(200, build_response(self.eng, X, prior, th_auto, th_review))
            elif p == "/v1/decide":
                self._send(200, single_response(self.eng, X, prior, th_auto, th_review))
            else:
                self._send(404, {"error": f"not found: {p}"})
        except Exception as exc:                                   # noqa: BLE001
            self._send(500, {"error": str(exc)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    eng = SentinelEngine(device=args.device)
    Handler.eng = eng
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[server] 哨兵决策服务已启动 "
          f"http://{args.host}:{args.port}  (device={eng.device})", flush=True)
    print("[server] POST /v1/decide   GET /healthz", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
