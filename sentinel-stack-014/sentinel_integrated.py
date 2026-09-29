#!/usr/bin/env python3
"""哨兵 × jev-decision-service —— 正式接入版服务。

设计原则
--------
1. **不复制框架代码**：决策层复用 jev 的 calibration / policy / engine / server，
   我们只提供一个 `evaluator`（吃结构化特征、吐答题卡）和一套工业问题集。
2. **职责分离**：哨兵负责「结构化数值 → 概率分布」；jev 负责
   校准策略、分布漂移检测、稀有类保护、OOD 拦截、确定性路由。
3. **响应契约不变**：完全沿用 jev 的 /v1/decide 字段（recommended_action / route /
   warnings / calibration_profile ...），看板按原契约消费。

三层路由
--------
    结构化传感器 → [System One] 哨兵决策头（torch, GPU, 亚毫秒）
                        ↓ 置信度 < floor
                   [System Two] 生成式（本地 27B）
                        ↓ 失败 / 无结构化特征
                   [兜底]       规则 provider（框架自带）

用法
----
    python3 sentinel_integrated.py --port 9000 [--escalate] [--retrain]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from http.server import ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "vendor"))

from jev_service.calibration import DecisionHeadProfile               # noqa: E402
from jev_service.engine import DecisionEngine                        # noqa: E402
from jev_service.models import DecisionRequest, Prediction           # noqa: E402
from jev_service.policy import PolicyConfig                          # noqa: E402
from jev_service.provider import (DecisionProvider, OpenSourceProvider,  # noqa: E402
                                  ProviderError, RuleBasedProvider,
                                  StructuredHeadProvider, TieredProvider)
from jev_service.questions import QuestionProfile                    # noqa: E402
from jev_service.server import Handler as JevHandler                  # noqa: E402
from jev_service.service import Service                              # noqa: E402
from jev_service.state_builder import StateBuilder                   # noqa: E402

import sentinel_heads as SH                                          # noqa: E402

DASHBOARD = os.path.join(ROOT, "dashboard.html")


# ==================================================================== 问题集
def industrial_profile() -> QuestionProfile:
    """工业维护版问题集。

    **刻意沿用 policy 认识的问题 id**（risk_level / needs_evidence /
    needs_clarification / allows_side_effect / next_action），
    因此策略层一行都不用改。新增 fault_type 承载故障类型，
    policy 对它无感（只做通用处理），但它是校准与稀有类保护的落点。
    """
    return QuestionProfile(
        name="industrial-maintenance",
        version="v1",
        questions={
            "fault": {
                "type": "noul",
                "question": "该设备当前是否处于故障状态？",
            },
            "risk_level": {
                "type": "score",
                "question": "该设备当前的维护优先级是第几档？依据刀具磨损量、温升、负载综合判断。",
                "levels": SH.SEVERITY_LABELS,
            },
            "fault_type": {
                "type": "choice",
                "question": ("最可能的故障类型？No Failure=无故障，TWF=工具磨损失效，"
                             "HDF=散热失效，PWF=功率失效，OSF=过载失效，RNF=随机失效。"),
                "options": SH.TYPES,
            },
            "needs_evidence": {
                "type": "noul",
                "question": "是否需要调取维修手册或历史工单才能给出可靠处置？",
            },
            "needs_clarification": {
                "type": "noul",
                "question": "传感器读数是否可疑（超物理量程或互相矛盾），需要现场先确认？",
            },
            "allows_side_effect": {
                "type": "noul",
                "question": "当前判断是否已足够可靠，可以自动派工单（无需人工过目）？",
            },
            "next_action": {
                "type": "choice",
                "question": "系统下一步应当做什么？",
                "options": ["answer_from_context", "retrieve_evidence",
                            "ask_clarification", "draft_suggestion", "human_review"],
            },
        },
    )


# ==================================================================== evaluator
NEXT_ACTIONS = ["answer_from_context", "retrieve_evidence",
                "ask_clarification", "draft_suggestion", "human_review"]
# 无信息时的先验落点：偏安全分支（不确定就转人工 / 先查依据）
NEXT_PRIOR = {"answer_from_context": 0.20, "retrieve_evidence": 0.10,
              "ask_clarification": 0.20, "draft_suggestion": 0.10,
              "human_review": 0.40}


def damped_distribution(branch: str, confidence: float) -> dict[str, float]:
    """置信度阻尼分布：alpha·onehot(branch) + (1-alpha)·prior，alpha = 置信度。

    这样「概率分布的形状」直接由已标定的置信度决定：
    高置信 → 尖锐（可自动处置）；低置信 → 摊平到安全分支（转人工/追问）。
    比人为拍一个 0.9 更诚实，也让 policy 的阈值有真实含义。
    """
    alpha = min(max(confidence, 0.0), 1.0)
    probs = {k: (1.0 - alpha) * NEXT_PRIOR[k] + (alpha if k == branch else 0.0)
             for k in NEXT_ACTIONS}
    total = sum(probs.values()) or 1.0
    return {k: v / total for k, v in probs.items()}


def _norm_entropy(probs: dict[str, float]) -> float:
    vals = [p for p in probs.values() if p > 1e-12]
    if len(vals) <= 1:
        return 0.0
    h = -sum(p * math.log(p) for p in vals)
    return h / math.log(len(probs))


def make_evaluator(heads: SH.SentinelHeads, cfg: "SentinelConfig"):
    """返回 evaluator(features) -> jev 答题卡。

    决策头只负责三个原语；下面三个 noul 与 next_action 是**策略桥**
    （把标定后的置信度翻译成框架能消费的信号），它们的定义都留在返回值里，
    便于审计与复现。
    """
    def evaluator(features: dict[str, Any]) -> dict[str, Any]:
        missing = [k for k in SH.RAW_KEYS if k not in features]
        if missing:
            raise ValueError(f"缺少传感器字段: {','.join(missing)}")
        raw = {k: float(features[k]) for k in SH.RAW_KEYS}

        out = heads.predict(raw)
        p = out["p_fault"]
        tp = out["type_probs"]
        sp = out["sev_probs"]
        sev_exp = sum(i * sp[label] for i, label in enumerate(SH.SEVERITY_LABELS))
        conf = max(p, 1.0 - p)

        # --- fault_type：与 Noul 做一致性绑定，保证两个决策永不自相矛盾
        raw_share = {t: max(tp[t], 0.0) for t in SH.FAULT_TYPES}
        tot = sum(raw_share.values())
        share = ({t: v / tot for t, v in raw_share.items()} if tot > 1e-9
                 else {t: 1.0 / len(SH.FAULT_TYPES) for t in SH.FAULT_TYPES})
        type_probs = {"No Failure": 1.0 - p}
        type_probs.update({t: p * share[t] for t in SH.FAULT_TYPES})
        s = sum(type_probs.values()) or 1.0
        type_probs = {k: v / s for k, v in type_probs.items()}
        type_choice = max(type_probs, key=type_probs.get)

        # --- needs_evidence：**确认有故障、但类型判不准**时才需要翻手册/历史工单
        # 早期版本漏乘 p_fault，导致健康样本（类型条件分布的熵天然很高）也被
        # 判成「需查手册」，自动化率被压到 13.6%。加入 p_fault 权重后语义才正确。
        p_evidence = min(0.95, max(0.05,
                                   0.05 + 0.90 * p * _norm_entropy(share)))

        # --- needs_clarification：传感器可信度检查（读数可疑 → 现场先确认）
        bad = SH.sensor_plausibility(raw)
        p_clarify = 0.05 if bad == 0 else min(0.95, 0.35 + 0.25 * bad)

        # --- allows_side_effect：高置信才认为可以自动派单
        p_side_effect = 0.90 if conf >= cfg.auto_confidence else 0.20

        # --- next_action：只按「置信度」分三档。
        # 紧急度不在这里判——那是 policy.urgent_score 的职责，避免两处各写一套阈值
        # （会造成「调 policy 阈值却不见效果」的假象，早期版本踩过）。
        if conf >= cfg.auto_confidence:
            branch = "draft_suggestion" if p >= 0.5 else "answer_from_context"
        elif conf >= cfg.review_confidence:
            branch = "retrieve_evidence"
        else:
            branch = "human_review"
        na = damped_distribution(branch, conf)

        return {
            # Noul 原语：是不是故障。框架的 apply_profile 会在此处应用先验修正
            "fault": {"type": "noul", "noul": round(p, 6)},
            "risk_level": {
                "type": "score",
                "score": round(sev_exp, 6),
                "probabilities": {k: round(v, 6) for k, v in sp.items()},
                "confidence": round(max(sp.values()), 6),
            },
            "fault_type": {
                "type": "choice",
                "choice": type_choice,
                "probabilities": {k: round(v, 6) for k, v in type_probs.items()},
                "confidence": round(max(type_probs.values()), 6),
            },
            "needs_evidence": {"type": "noul", "noul": round(p_evidence, 6)},
            "needs_clarification": {"type": "noul", "noul": round(p_clarify, 6)},
            "allows_side_effect": {"type": "noul", "noul": round(p_side_effect, 6)},
            "next_action": {
                "type": "choice",
                "choice": branch,
                "probabilities": {k: round(v, 6) for k, v in na.items()},
                "confidence": round(conf, 6),
            },
        }

    return evaluator


# ==================================================================== provider 栈
@dataclass
class SentinelProvider:
    """按输入模态分派的三层 provider。

    - 有结构化特征 → TieredProvider（哨兵决策头；低置信升级生成式）
    - 无结构化特征 → 直接走生成式（文本类工单/报修描述）
    - 任一层失败 → 规则 provider 兜底，并标记 degraded
    """

    structured: DecisionProvider | None = None
    generative: DecisionProvider | None = None
    fallback: DecisionProvider = field(default_factory=RuleBasedProvider)
    name: str = "sentinel"
    degraded: bool = False
    last_report: Any = None
    last_route: str = "fast"
    last_reason: str = ""
    last_normalization_warnings: list[str] = field(default_factory=list)

    def _has_features(self, state: dict[str, Any]) -> bool:
        feats = (state.get("conversation_state") or {}).get("structured_features")
        return isinstance(feats, dict) and bool(feats)

    def evaluate(self, state: dict[str, Any]) -> list[Prediction]:
        self.degraded = False
        self.last_normalization_warnings = []
        primary = self.structured if self._has_features(state) else self.generative
        if primary is not None:
            try:
                preds = primary.evaluate(state)
                self.last_report = getattr(primary, "last_report", None)
                self.last_route = str(getattr(primary, "last_route", "fast"))
                self.last_reason = str(getattr(primary, "last_reason", ""))
                self.last_normalization_warnings = list(
                    getattr(primary, "last_normalization_warnings", []))
                # TieredProvider 的降级遥测会被它自己的收尾语句覆盖成
                # 「fast / fast_head_accepted」，导致上层无法区分
                # 「正常走快头」与「升级生成式失败后退回」。这里在包装层纠正。
                if getattr(primary, "degraded", False):
                    self.degraded = True
                    self.last_route = "fast_degraded"
                    self.last_reason = "generative_tier_failed_fell_back_to_fast_head"
                return preds
            except ProviderError as exc:
                self.degraded = True
                self.last_route = "fallback"
                self.last_reason = f"primary_failed:{exc}"
        preds = self.fallback.evaluate(state)
        self.last_report = getattr(self.fallback, "last_report", None)
        if primary is None:
            self.degraded = True
            self.last_route = "fallback"
            self.last_reason = "no_generative_tier_for_unstructured_input"
        return preds


@dataclass(frozen=True)
class SentinelConfig:
    auto_confidence: float = 0.90      # ≥ 此置信度才允许自动处置
    review_confidence: float = 0.80    # [review, auto) 之间 → 先查手册/历史工单
    urgent_score: float = 2.5          # 严重度期望值 ≥ 此值 → 强制人工（4 档制）
    escalate_floor: float = 0.80       # 决策头置信度低于此值 → 升级生成式
    escalate: bool = False
    # 升级层用 8B 而非 27B：27B 是带思考链的模型，经 ollama 的
    # OpenAI 兼容端点调用时思考过程会耗尽输出预算，返回空内容
    # （实测 80.9s 后 JSON 解析失败，注入 think=false 亦无效）。
    # 8B 同路径 22.2s 可用，故选它做升级层。
    generative_model: str = "modelscope.cn/unsloth/Qwen3-8B-GGUF:latest"
    generative_endpoint: str = "http://127.0.0.1:11434/v1/chat/completions"
    generative_timeout: float = 180.0


def build_engine(heads: SH.SentinelHeads, cfg: SentinelConfig,
                 class_support: dict[str, int]) -> DecisionEngine:
    profile = industrial_profile()

    head_profile = DecisionHeadProfile(
        name="sentinel-mlp",
        version="1.0",
        label_provenance="derived_rule",          # 严重度是派生标签，如实标注
        calibrated=True,
        calibration_distribution="ai4i-2020-natural",
        # 逐头温度已在 evaluator 内应用（profile 只支持单一 temperature），
        # 这里留 1.0 让框架的温度除法成为恒等；**先验修正仍由框架完成**。
        temperature=1.0,
        train_positive_rate=0.0312,
        min_class_support=1,
        class_support=class_support,
        rare_class_threshold=50,
        class_support_head="fault_type",          # 罕见故障类型才是风险落点
    )

    fast = StructuredHeadProvider(
        evaluator=make_evaluator(heads, cfg),
        profile=head_profile,
        questions=profile,
        name="sentinel-head",
    )

    generative: DecisionProvider | None = None
    if cfg.escalate:
        generative = OpenSourceProvider(
            endpoint=cfg.generative_endpoint,
            model=cfg.generative_model,
            timeout=cfg.generative_timeout,
            questions=profile,
            name="local-generative",
        )
        structured: DecisionProvider | None = TieredProvider(
            fast=fast, generative=generative, fast_confidence_floor=cfg.escalate_floor)
    else:
        structured = fast

    provider = SentinelProvider(
        structured=structured,
        # 无结构化特征的输入（工单文本、报修描述）交给生成式；
        # 未开启升级时留空，直接落规则兜底，避免拖慢报警通道。
        generative=generative,
        fallback=RuleBasedProvider(),
    )

    policy = PolicyConfig(
        auto_confidence=0.80,
        auto_probability=0.70,
        ambiguity_margin=0.20,
        urgent_score=cfg.urgent_score,
        require_calibration_for_auto=True,
        task_probability_thresholds={"next_action": 0.70,
                                     "needs_evidence": 0.70,
                                     "needs_clarification": 0.70},
        ood_review_threshold=0.70,
    )
    return DecisionEngine(
        provider=provider,
        policy=policy,
        state_builder=StateBuilder(ood_threshold=policy.ood_review_threshold),
    )


# ==================================================================== HTTP
class Handler(JevHandler):
    """复用 jev 的路由，只加两件它没有的东西：看板页 + 批量端点。"""

    engine: DecisionEngine | None = None
    service: Service | None = None
    meta: dict[str, Any] = {}

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("access-control-allow-origin", "*")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, status: int, text: str) -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "text/html; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ---------------------------------------------------------- GET
    def do_GET(self) -> None:                            # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/dashboard", "/index.html"):
            try:
                with open(DASHBOARD, encoding="utf-8") as fh:
                    self._html(200, fh.read())
            except OSError:
                self._json(200, {"service": "sentinel × jev", "note": "dashboard.html 缺失",
                                 "endpoints": ["/healthz", "/v1/meta", "/v1/decide",
                                               "/v1/decide/batch", "/v1/replay"]})
            return
        if path == "/healthz" and self.engine is not None:
            self._json(200, {
                "ok": True,
                "provider": self.engine.provider_name,
                "degraded": self.engine.degraded,
                **self.meta,
            })
            return
        if path == "/v1/meta" and self.engine is not None:
            self._json(200, self.meta)
            return
        if path == "/v1/eval/dataset" and self.engine is not None:
            from urllib.parse import parse_qs
            qs = parse_qs(urlparse(self.path).query)
            self._eval_dataset(int(qs.get("n", ["500"])[0]),
                               int(qs.get("seed", ["7"])[0]),
                               float(qs.get("urgent", ["2.5"])[0]))
            return
        super().do_GET()

    # ------------------------------------------------- 内置评测（真实留出集）
    def _eval_dataset(self, n: int, seed: int, urgent: float = 2.5) -> None:
        """在 AI4I **留出测试集**上跑端到端，统计自动化率与安全指标。

        注意用的是留出集（训练时未见过的 20%），不是全量，避免数据泄漏。
        urgent 可覆盖 PolicyConfig.urgent_score，用于扫描「自动化率 ↔ 安全」权衡。
        """
        import dataclasses
        import numpy as np
        original_policy = self.engine.policy
        if urgent != original_policy.urgent_score:
            self.engine.policy = dataclasses.replace(original_policy, urgent_score=urgent)
        rows = SH.holdout_rows(self.meta["data_path"])
        rng = np.random.default_rng(seed)
        k = min(max(1, n), len(rows))
        sel = rng.choice(len(rows), k, replace=False)
        requests, truths = [], []
        for i in sel:
            row = rows[int(i)]
            requests.append(DecisionRequest(
                conversation_id=f"eval-{int(i)}-{seed}",
                current_message="设备状态上报",
                conversation_state={"structured_features": {
                    key: float(row[key]) for key in SH.RAW_KEYS}},
            ))
            truths.append(int(row["failure"]))

        t0 = time.perf_counter()
        results, lat = [], []
        for request in requests:
            t1 = time.perf_counter()
            results.append(self.engine.decide(request).to_dict())
            lat.append((time.perf_counter() - t1) * 1000.0)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        actions: dict[str, int] = {}
        for result in results:
            kind = (result.get("recommended_action") or {}).get("kind", "?")
            actions[kind] = actions.get(kind, 0) + 1

        AUTO_FAULT = ("draft_suggestion",)
        AUTO_NORMAL = ("answer_from_context",)
        REVIEW = ("human_review", "retrieve_evidence", "ask_clarification",
                  "abstain", "stop")
        buckets = {"auto_fault": 0, "auto_normal": 0, "review": 0}
        auto_total = auto_correct = 0
        fault_total = fault_auto = fault_review = fault_miss = 0
        normal_total = normal_auto_ok = normal_review = 0
        reason_hist: dict[str, int] = {}
        cross: dict[str, int] = {}

        for truth, result in zip(truths, results):
            action = result.get("recommended_action") or {}
            kind = action.get("kind")
            codes = action.get("reason_codes") or ["(none)"]
            for code in codes:
                reason_hist[code] = reason_hist.get(code, 0) + 1
            key = f"{kind} ← {codes[0]}"
            cross[key] = cross.get(key, 0) + 1
            if kind in AUTO_FAULT:
                buckets["auto_fault"] += 1
            elif kind in AUTO_NORMAL:
                buckets["auto_normal"] += 1
            else:
                buckets["review"] += 1
            if kind in AUTO_FAULT or kind in AUTO_NORMAL:
                auto_total += 1
                expect_fault = kind in AUTO_FAULT
                auto_correct += int(expect_fault == bool(truth))
            if truth == 1:
                fault_total += 1
                if kind in AUTO_FAULT:
                    fault_auto += 1
                elif kind in AUTO_NORMAL:
                    fault_miss += 1
                else:
                    fault_review += 1
            else:
                normal_total += 1
                if kind in AUTO_NORMAL:
                    normal_auto_ok += 1
                elif kind in REVIEW:
                    normal_review += 1

        total = len(truths)
        auto_n = buckets["auto_fault"] + buckets["auto_normal"]
        payload = {
            "dataset": "AI4I 2020 · 留出测试集（训练未见）",
            "n": total,
            "seed": seed,
            "urgent_score": urgent,
            "reason_hist": dict(sorted(reason_hist.items(), key=lambda kv: -kv[1])),
            "action_reason_cross": dict(sorted(cross.items(), key=lambda kv: -kv[1])),
            "automation_rate": round(auto_n / max(total, 1), 4),
            "auto_count": auto_n,
            "review_count": total - auto_n,
            "auto_precision": (round(auto_correct / auto_total, 4)
                               if auto_total else None),
            "actions": actions,
            "buckets": buckets,
            "fault_total": fault_total,
            "fault_recall_reviewed_or_auto": (round((fault_auto + fault_review) / fault_total, 4)
                                              if fault_total else None),
            "fault_auto_dispatch_rate": (round(fault_auto / fault_total, 4)
                                         if fault_total else None),
            "fault_miss_rate": (round(fault_miss / fault_total, 4)
                                if fault_total else None),
            "fault_miss_count": fault_miss,
            "normal_total": normal_total,
            "normal_false_review_rate": (round(normal_review / normal_total, 4)
                                         if normal_total else None),
            "latency_ms": round(elapsed_ms, 3),
            "latency_per_item_ms": round(elapsed_ms / max(total, 1), 6),
            "throughput_per_s": round(total / max(elapsed_ms / 1000.0, 1e-9), 1),
            "latency_p50_engine_ms": (round(sorted(lat)[len(lat) // 2], 4)
                                      if lat else None),
            "latency_p95_engine_ms": (round(sorted(lat)[int(len(lat) * 0.95)], 4)
                                      if lat else None),
        }
        self.engine.policy = original_policy        # 还原策略，避免污染后续请求
        self._json(200, payload)

    # ---------------------------------------------------------- POST
    def do_POST(self) -> None:                           # noqa: N802
        path = urlparse(self.path).path
        if path == "/v1/decide/batch":
            self._batch()
            return
        super().do_POST()

    def _batch(self) -> None:
        if self.engine is None:
            self._json(500, {"error": "engine not ready"})
            return
        try:
            length = int(self.headers.get("content-length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            items = payload.get("items") or payload.get("requests") or []
            if not isinstance(items, list) or not items:
                raise ValueError("items must be a non-empty array")
            requests = [DecisionRequest.from_dict(item) for item in items]
        except (ValueError, json.JSONDecodeError) as exc:
            self._json(422, {"error": str(exc)})
            return

        t0 = time.perf_counter()
        replay = self.engine.replay(requests)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        actions = replay["actions"]
        auto_kinds = ("answer_from_context", "draft_suggestion")
        auto_n = sum(actions.get(k, 0) for k in auto_kinds)
        results = replay["results"]

        # 可选：items 里带 expected.fault 时顺带算自动部分准确率
        auto_correct = auto_total = 0
        for item, result in zip(items, results):
            expected = item.get("expected") if isinstance(item, dict) else None
            kind = (result.get("recommended_action") or {}).get("kind")
            if kind not in auto_kinds:
                continue
            auto_total += 1
            if not isinstance(expected, dict) or "fault" not in expected:
                continue
            truth = bool(expected["fault"])
            pred_fault = kind == "draft_suggestion"
            auto_correct += int(truth == pred_fault)

        self._json(200, {
            **replay,
            "automation_rate": round(auto_n / max(len(items), 1), 4),
            "auto_count": auto_n,
            "review_count": len(items) - auto_n,
            "auto_evaluated": auto_total,
            "auto_correct": auto_correct,
            "auto_precision": (round(auto_correct / auto_total, 4)
                               if auto_total else None),
            "latency_ms": round(elapsed_ms, 3),
            "latency_per_item_ms": round(elapsed_ms / max(len(items), 1), 6),
            "throughput_per_s": round(len(items) / max(elapsed_ms / 1000.0, 1e-9), 1),
        })


# ==================================================================== main
def main() -> None:
    import torch

    ap = argparse.ArgumentParser(description="哨兵 × jev 集成决策服务")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "ai4i2020.csv"))
    ap.add_argument("--cache", default=os.path.join(ROOT, "heads.pt"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--escalate", action="store_true", help="开启低置信升级生成式")
    ap.add_argument("--escalate-floor", type=float, default=0.80)
    ap.add_argument("--no-generative", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    heads, cached = SH.SentinelHeads.load_or_train(
        args.data, args.device, args.cache, retrain=args.retrain)
    SH.export_meta(heads, os.path.join(ROOT, "heads_meta.json"))
    train_s = time.perf_counter() - t0

    cfg = SentinelConfig(
        escalate=bool(args.escalate) and not args.no_generative,
        escalate_floor=args.escalate_floor,
    )
    support = head_profile_support(args.data)
    engine = build_engine(heads, cfg, class_support=support)

    Handler.service = Service(engine=engine)
    Handler.engine = engine
    Handler.meta = {
        "framework": "jev-decision-service 0.1.4",
        "decision_head": f"sentinel-mlp@{args.device}",
        "torch": torch.__version__,
        "device": args.device,
        "data_path": args.data,
        "temperatures": {k: round(v, 4) for k, v in heads.temps.items()},
        "class_support": support,
        "train_seconds": round(train_s, 2),
        "cache_hit": cached,
        "escalate": cfg.escalate,
        "escalate_floor": cfg.escalate_floor,
        "policy": {"auto_confidence": 0.80, "urgent_score": cfg.urgent_score,
                   "next_action_threshold": 0.70},
    }

    print(f"[启动] 框架 jev 0.1.4 | 决策头 sentinel-mlp @ {args.device} "
          f"| 缓存命中={cached} | 用时 {train_s:.2f}s")
    print(f"[启动] 温度 {({k: round(v, 3) for k, v in heads.temps.items()})} "
          f"| 升级生成式={cfg.escalate}")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[就绪] http://{args.host}:{args.port}  (/ 看板  /v1/decide  /v1/decide/batch)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def head_profile_support(data_csv: str) -> dict[str, int]:
    """从训练集统计各故障类型样本数，供稀有类保护使用。"""
    counts = {t: 0 for t in SH.TYPES}
    for row in SH.load_rows(data_csv):
        counts[row["type"]] = counts.get(row["type"], 0) + 1
    return counts


if __name__ == "__main__":
    main()
