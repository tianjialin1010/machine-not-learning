#!/usr/bin/env python3
"""哨兵 × jev-decision-service **0.1.5** —— 集成服务（升级版）。

相对 0.1.4 版的改动（都是 0.1.5 新能力，不是绕过）
--------------------------------------------------
1. **问题集用 `roles` 映射**，不再把领域名硬塞进 policy 认识的 id
   （0.1.5 的 `QuestionProfile.roles` + `PolicyConfig.question_roles`）。
2. **逐头温度交给框架**：`DecisionHeadProfile.temperatures`，evaluator 只吐原始概率。
3. **先验修正限定目标头**：`prior_correction_head="fault"` + `prior_shift_assumption="label_shift"`，
   不再误伤其它 noul 预测（0.1.4 的老问题）。
4. **Score 用规范数字键**（"0".."3"），符合 0.1.5 `apply_profile` 的硬要求。
5. **特征契约**：feature_count / feature_names / feature_bounds / feature_units，
   输入不匹配时 `accepts()` 返回 False，TieredProvider 自动改走文本模型而不是报错。
6. **删掉自写的 SentinelProvider 包装层**：0.1.5 的 TieredProvider 已经正确处理
   降级遥测（`fast_head_uncertain` / `generative_failed:<code>`）、`accepts()` 路由、
   `provider_disagreement`、`last_safety_predictions`（升级后原始风险不丢）。
7. 升级层支持 **StepFun 官方 API**（设 STEPFUN_API_KEY 即启用）或本地 8B。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import replace
from http.server import ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "vendor"))

from jev_service.calibration import DecisionHeadProfile                  # noqa: E402
from jev_service.engine import DecisionEngine                            # noqa: E402
from jev_service.models import DecisionRequest                           # noqa: E402
from jev_service.policy import PolicyConfig                              # noqa: E402
from jev_service.provider import (AbstainProvider, OpenSourceProvider,   # noqa: E402
                                  ProviderError, ResilientProvider,
                                  StepFunProvider, StructuredHeadProvider,
                                  TieredProvider)
from jev_service.questions import QuestionProfile                        # noqa: E402
from jev_service.server import Handler as JevHandler                      # noqa: E402
from jev_service.service import Service                                  # noqa: E402
from jev_service.state_builder import StateBuilder                       # noqa: E402

import sentinel_heads as SH                                              # noqa: E402

from frontend_static import handle_frontend_get

DASHBOARD = os.path.join(ROOT, "dashboard.html")
FRAMEWORK_VERSION = "0.1.5"

# 对外输入契约：5 路原始传感器。派生特征由决策头内部完成。
FEATURE_UNITS = {"air_temp_c": "C", "process_temp_c": "C", "rpm": "rpm",
                 "torque_nm": "Nm", "tool_wear_min": "min"}
FEATURE_BOUNDS = {"air_temp_c": (-20.0, 80.0), "process_temp_c": (-20.0, 200.0),
                  "rpm": (0.0, 4000.0), "torque_nm": (0.0, 200.0),
                  "tool_wear_min": (0.0, 300.0)}

# 无信息时的先验落点：偏安全分支（不确定就转人工 / 先查依据）
NEXT_PRIOR = {"answer_from_context": 0.20, "retrieve_evidence": 0.10,
              "ask_clarification": 0.20, "draft_suggestion": 0.10,
              "human_review": 0.40}
NEXT_ACTIONS = list(NEXT_PRIOR)
SCORE_KEYS = [str(i) for i in range(len(SH.SEVERITY_LABELS))]


# ==================================================================== 问题集
def industrial_profile() -> QuestionProfile:
    """工业维护版问题集（0.1.5：用 roles 映射，不依赖固定 id 名）。"""
    return QuestionProfile(
        name="industrial-maintenance",
        version="v1.5",
        roles={"risk": "intervention_risk",
               "action": "next_action",
               "evidence": "needs_evidence",
               "clarification": "needs_clarification",
               "side_effect": "allows_side_effect"},
        questions={
            "fault": {
                "type": "noul",
                "question": "该设备当前是否处于故障状态？",
            },
            "fault_type": {
                "type": "choice",
                "question": ("最可能的故障类型？No Failure=无故障，TWF=工具磨损失效，"
                             "HDF=散热失效，PWF=功率失效，OSF=过载失效，RNF=随机失效。"),
                "options": SH.TYPES,
            },
            "intervention_risk": {
                "type": "score",
                "question": ("该设备当前的维护优先级？0=观察，1=计划，2=尽快，3=立即；"
                             "依据刀具磨损量、温升与负载综合判断。"),
                "levels": SH.SEVERITY_LABELS,
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
                "options": NEXT_ACTIONS,
            },
        },
    )


# ==================================================================== evaluator
def damped_distribution(branch: str, confidence: float) -> dict[str, float]:
    """置信度阻尼分布：alpha·onehot(branch) + (1-alpha)·prior，alpha = 已标定置信度。

    高置信 → 尖锐（可自动处置）；低置信 → 摊平到安全分支（转人工/查依据）。
    这样 policy 的阈值比较才有真实含义，而不是人为拍一个 0.9。
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
    return -sum(p * math.log(p) for p in vals) / math.log(len(probs))


def make_evaluator(heads: SH.SentinelHeads, cfg: "SentinelConfig"):
    """返回 evaluator(features) -> jev 答题卡。

    **只吐原始（未标定）概率**：温度缩放与先验修正由框架的 apply_profile 负责，
    依据 DecisionHeadProfile 里声明的 temperatures / prior_correction_head。
    """
    def evaluator(features: dict[str, Any]) -> dict[str, Any]:
        raw = {k: float(features[k]) for k in SH.RAW_KEYS}
        out = heads.predict(raw)
        p = out["p_fault"]
        tp = out["type_probs"]
        sp = out["sev_probs"]
        sev_exp = sum(i * sp[label] for i, label in enumerate(SH.SEVERITY_LABELS))
        conf = max(p, 1.0 - p)

        # fault_type：与 Noul 做一致性绑定，保证两个决策永不自相矛盾
        raw_share = {t: max(tp[t], 0.0) for t in SH.FAULT_TYPES}
        tot = sum(raw_share.values())
        share = ({t: v / tot for t, v in raw_share.items()} if tot > 1e-9
                 else {t: 1.0 / len(SH.FAULT_TYPES) for t in SH.FAULT_TYPES})
        type_probs = {"No Failure": 1.0 - p}
        type_probs.update({t: p * share[t] for t in SH.FAULT_TYPES})
        s = sum(type_probs.values()) or 1.0
        type_probs = {k: v / s for k, v in type_probs.items()}
        type_choice = max(type_probs, key=type_probs.get)

        # needs_evidence：确认有故障、但类型判不准时才需要翻手册/历史工单
        p_evidence = min(0.95, max(0.05, 0.05 + 0.90 * p * _norm_entropy(share)))
        # needs_clarification：传感器可信度检查
        bad = SH.sensor_plausibility(raw)
        p_clarify = 0.05 if bad == 0 else min(0.95, 0.35 + 0.25 * bad)
        # allows_side_effect：高置信才认为可以自动派单
        p_side_effect = 0.90 if conf >= cfg.auto_confidence else 0.20

        # next_action：只按置信度分三档；紧急度由 policy.urgent_score 判（单一权威）
        if conf >= cfg.auto_confidence:
            branch = "draft_suggestion" if p >= 0.5 else "answer_from_context"
        elif conf >= cfg.review_confidence:
            branch = "retrieve_evidence"
        else:
            branch = "human_review"
        na = damped_distribution(branch, conf)

        return {
            "fault": {"type": "noul", "noul": round(p, 6)},
            "fault_type": {
                "type": "choice", "choice": type_choice,
                "probabilities": {k: round(v, 6) for k, v in type_probs.items()},
                "confidence": round(max(type_probs.values()), 6),
            },
            "intervention_risk": {
                "type": "score", "score": round(sev_exp, 6),
                # 0.1.5 要求标定后的 Score 使用规范数字键
                "probabilities": {SCORE_KEYS[i]: round(float(sp[label]), 6)
                                  for i, label in enumerate(SH.SEVERITY_LABELS)},
                "confidence": round(float(max(sp.values())), 6),
            },
            "needs_evidence": {"type": "noul", "noul": round(p_evidence, 6)},
            "needs_clarification": {"type": "noul", "noul": round(p_clarify, 6)},
            "allows_side_effect": {"type": "noul", "noul": round(p_side_effect, 6)},
            "next_action": {
                "type": "choice", "choice": branch,
                "probabilities": {k: round(v, 6) for k, v in na.items()},
                "confidence": round(conf, 6),
            },
        }

    return evaluator


# ==================================================================== 配置
class SentinelConfig:
    """可调阈值集合（服务端唯一权威，evaluator 与 policy 共用同一组数值）。"""

    def __init__(self, auto_confidence: float = 0.90, review_confidence: float = 0.80,
                 urgent_score: float = 2.5, escalate: bool = False,
                 escalate_floor: float = 0.80, generative: str = "local",
                 generative_model: str = "modelscope.cn/unsloth/Qwen3-8B-GGUF:latest",
                 generative_endpoint: str = "http://127.0.0.1:11434/v1/chat/completions",
                 generative_timeout: float = 180.0,
                 generative_max_tokens: int = 2048) -> None:
        self.auto_confidence = auto_confidence
        self.review_confidence = review_confidence
        self.urgent_score = urgent_score
        self.escalate = escalate
        self.escalate_floor = escalate_floor
        self.generative = generative
        self.generative_model = generative_model
        self.generative_endpoint = generative_endpoint
        self.generative_timeout = generative_timeout
        self.generative_max_tokens = generative_max_tokens


def build_generative(cfg: SentinelConfig, profile: QuestionProfile):
    """升级层：StepFun 官方 API（有 key 时）或本地 OpenAI 兼容端点。"""
    if cfg.generative == "stepfun":
        key = os.getenv("STEPFUN_API_KEY", "").strip()
        if not key:
            raise ProviderError("STEPFUN_API_KEY 未配置，无法启用 StepFun 升级层",
                                code="configuration_error")
        base = os.getenv("STEPFUN_BASE_URL", "https://api.stepfun.com/v1").rstrip("/")
        return StepFunProvider(
            endpoint=base + "/chat/completions",
            model=os.getenv("STEPFUN_MODEL", "step-3.5-flash"),
            api_key=key,
            timeout=float(os.getenv("STEPFUN_TIMEOUT", "30")),
            questions=profile,
        )
    return OpenSourceProvider(
        endpoint=cfg.generative_endpoint, model=cfg.generative_model,
        timeout=cfg.generative_timeout, questions=profile, name="local-generative",
        # 7 问嵌套 JSON 在默认 1024 token 下会被截断（finish_reason=length
        # → incomplete_output），实测需 2048 才够
        max_tokens=cfg.generative_max_tokens,
    )


def build_engine(heads: SH.SentinelHeads, cfg: SentinelConfig,
                 class_support: dict[str, int], profile: QuestionProfile) -> DecisionEngine:
    head_profile = DecisionHeadProfile(
        name="sentinel-mlp", version="1.5",
        label_provenance="derived_rule",             # 严重度是派生标签，如实标注
        calibrated=True,
        calibration_distribution="ai4i-2020-natural",
        # 逐头温度：框架在对应用预测上应用（evaluator 只吐原始概率）
        temperatures={"fault": heads.temps["noul"],
                      "fault_type": heads.temps["type"],
                      "intervention_risk": heads.temps["sev"]},
        # 先验修正只作用于 fault 头，且要求显式声明 label_shift 假设
        prior_correction_head="fault",
        prior_shift_assumption="label_shift",
        calibration_positive_rate=0.0312,
        min_class_support=1,
        class_support=class_support,
        rare_class_threshold=50,
        class_support_head="fault_type",
        # 特征契约：维度 / 名称 / 量程 / 单位，四道校验
        feature_count=len(SH.RAW_KEYS),
        feature_names=tuple(SH.RAW_KEYS),
        feature_bounds={k: tuple(v) for k, v in FEATURE_BOUNDS.items()},
        feature_units=dict(FEATURE_UNITS),
    )

    fast = StructuredHeadProvider(
        evaluator=make_evaluator(heads, cfg),
        profile=head_profile,
        questions=profile,
        name="sentinel-head",
    )
    generative = build_generative(cfg, profile) if cfg.escalate else None
    tiered = TieredProvider(fast=fast, generative=generative,
                            fast_confidence_floor=cfg.escalate_floor)
    # 兜底用 AbstainProvider（框架推荐）：不把对话规则伪装成工业预测
    provider = ResilientProvider(tiered, AbstainProvider(profile))

    policy = PolicyConfig(
        question_roles=profile.roles,          # 0.1.5：roles 必须显式传进 policy
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
        provider=provider, policy=policy,
        state_builder=StateBuilder(ood_threshold=policy.ood_review_threshold),
    )


# ==================================================================== 请求构造
def build_request(conversation_id: str, features: dict[str, float],
                  current_message: str = "设备状态上报",
                  ood_score: float | None = None,
                  distribution_id: str | None = None,
                  deployment_positive_rate: float | None = None) -> DecisionRequest:
    """统一构造请求（含特征单位，0.1.5 的单位契约要求）。"""
    state: dict[str, Any] = {"structured_features": dict(features),
                             "feature_units": dict(FEATURE_UNITS)}
    if ood_score is not None:
        state["ood_score"] = ood_score
    channel: dict[str, Any] = {}
    if distribution_id is not None:
        channel["distribution_id"] = distribution_id
    if deployment_positive_rate is not None:
        channel["deployment_positive_rate"] = deployment_positive_rate
    return DecisionRequest(conversation_id=conversation_id,
                           current_message=current_message,
                           conversation_state=state,
                           channel_metadata=channel)


# ==================================================================== HTTP
class Handler(JevHandler):
    """复用 0.1.5 的路由，只加看板 / 批量 / 留出集评测三个附加端点。"""

    engine: DecisionEngine | None = None
    service: Service | None = None
    meta: dict[str, Any] = {}
    cfg: SentinelConfig | None = None

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
        if handle_frontend_get(self, ROOT):
            return
        path = urlparse(self.path).path
        if path in ("/", "/dashboard", "/index.html"):
            try:
                with open(DASHBOARD, encoding="utf-8") as fh:
                    self._html(200, fh.read())
            except OSError:
                self._json(200, {"service": "sentinel × jev 0.1.5",
                                 "note": "dashboard.html 缺失",
                                 "endpoints": ["/healthz", "/v1/meta", "/v1/decide",
                                               "/v1/decide/batch", "/v1/eval/dataset"]})
            return
        if path == "/healthz" and self.engine is not None:
            self._json(200, {"ok": True, "provider": self.engine.provider_name,
                             "degraded": self.engine.degraded, **self.meta})
            return
        if path == "/v1/meta" and self.engine is not None:
            self._json(200, self.meta)
            return
        if path == "/v1/eval/dataset" and self.engine is not None:
            qs = parse_qs(urlparse(self.path).query)
            self._eval_dataset(int(qs.get("n", ["500"])[0]),
                               int(qs.get("seed", ["7"])[0]),
                               float(qs.get("urgent", [str(self.cfg.urgent_score)])[0]))
            return
        super().do_GET()

    # ------------------------------------------------- 内置评测（真实留出集）
    def _eval_dataset(self, n: int, seed: int, urgent: float | None = None) -> None:
        """在 AI4I **留出测试集**上跑端到端；urgent 可临时覆盖 policy 的紧急度阈值。"""
        import numpy as np
        rows = SH.holdout_rows(self.meta["data_path"])
        rng = np.random.default_rng(seed)
        k = min(max(1, n), len(rows))
        sel = rng.choice(len(rows), k, replace=False)
        requests, truths = [], []
        for i in sel:
            row = rows[int(i)]
            requests.append(build_request(
                f"eval-{int(i)}-{seed}",
                {key: float(row[key]) for key in SH.RAW_KEYS}))
            truths.append(int(row["failure"]))

        original = self.engine.policy
        if urgent is not None and urgent != original.urgent_score:
            self.engine.policy = replace(original, urgent_score=urgent)

        t0 = time.perf_counter()
        results, lat = [], []
        for request in requests:
            t1 = time.perf_counter()
            results.append(self.engine.decide(request).to_dict())
            lat.append((time.perf_counter() - t1) * 1000.0)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.engine.policy = original

        actions: dict[str, int] = {}
        AUTO_FAULT, AUTO_NORMAL = ("draft_suggestion",), ("answer_from_context",)
        REVIEW = ("human_review", "retrieve_evidence", "ask_clarification",
                  "abstain", "stop")
        buckets = {"auto_fault": 0, "auto_normal": 0, "review": 0}
        auto_total = auto_correct = 0
        fault_total = fault_auto = fault_review = fault_miss = 0
        normal_total = normal_review = 0
        reason_hist: dict[str, int] = {}
        cross: dict[str, int] = {}
        uncalibrated = 0

        for truth, result in zip(truths, results):
            action = result.get("recommended_action") or {}
            kind = action.get("kind")
            codes = action.get("reason_codes") or ["(none)"]
            actions[kind] = actions.get(kind, 0) + 1
            for code in codes:
                reason_hist[code] = reason_hist.get(code, 0) + 1
            key = f"{kind} ← {codes[0]}"
            cross[key] = cross.get(key, 0) + 1
            if any(d.get("calibration_status") == "uncalibrated"
                   for d in result.get("decisions", [])):
                uncalibrated += 1

            if kind in AUTO_FAULT:
                buckets["auto_fault"] += 1
            elif kind in AUTO_NORMAL:
                buckets["auto_normal"] += 1
            else:
                buckets["review"] += 1
            if kind in AUTO_FAULT or kind in AUTO_NORMAL:
                auto_total += 1
                auto_correct += int((kind in AUTO_FAULT) == bool(truth))
            if truth == 1:
                fault_total += 1
                if kind in AUTO_FAULT:
                    fault_auto += 1
                elif kind in AUTO_NORMAL:
                    fault_miss += 1
                else:
                    fault_review += 1
            elif kind in REVIEW:
                normal_total += 1
                normal_review += 1
            else:
                normal_total += 1

        total = len(truths)
        auto_n = buckets["auto_fault"] + buckets["auto_normal"]
        self._json(200, {
            "dataset": "AI4I 2020 · 留出测试集（训练未见）",
            "framework": FRAMEWORK_VERSION,
            "n": total, "seed": seed, "urgent_score": urgent,
            "automation_rate": round(auto_n / max(total, 1), 4),
            "auto_count": auto_n, "review_count": total - auto_n,
            "auto_precision": round(auto_correct / auto_total, 4) if auto_total else None,
            "actions": dict(sorted(actions.items(), key=lambda kv: -kv[1])),
            "buckets": buckets,
            "reason_hist": dict(sorted(reason_hist.items(), key=lambda kv: -kv[1])),
            "action_reason_cross": dict(sorted(cross.items(), key=lambda kv: -kv[1])),
            "fault_total": fault_total,
            "fault_auto_dispatch_rate": round(fault_auto / fault_total, 4) if fault_total else None,
            "fault_recall_reviewed_or_auto": (round((fault_auto + fault_review) / fault_total, 4)
                                              if fault_total else None),
            "fault_miss_rate": round(fault_miss / fault_total, 4) if fault_total else None,
            "fault_miss_count": fault_miss,
            "normal_total": normal_total,
            "normal_false_review_rate": (round(normal_review / normal_total, 4)
                                         if normal_total else None),
            "uncalibrated_responses": uncalibrated,
            "latency_ms": round(elapsed_ms, 3),
            "latency_per_item_ms": round(elapsed_ms / max(total, 1), 6),
            "throughput_per_s": round(total / max(elapsed_ms / 1000.0, 1e-9), 1),
            "latency_p50_engine_ms": round(sorted(lat)[len(lat) // 2], 4) if lat else None,
            "latency_p95_engine_ms": (round(sorted(lat)[int(len(lat) * 0.95)], 4)
                                      if lat else None),
        })

    # ---------------------------------------------------------- POST
    def do_POST(self) -> None:                           # noqa: N802
        if urlparse(self.path).path == "/v1/decide/batch":
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

        auto_kinds = ("answer_from_context", "draft_suggestion")
        auto_n = sum(replay["actions"].get(k, 0) for k in auto_kinds)
        auto_total = auto_correct = 0
        for item, result in zip(items, replay["results"]):
            expected = item.get("expected") if isinstance(item, dict) else None
            kind = (result.get("recommended_action") or {}).get("kind")
            if kind not in auto_kinds:
                continue
            auto_total += 1
            if isinstance(expected, dict) and "fault" in expected:
                auto_correct += int(bool(expected["fault"]) == (kind == "draft_suggestion"))
        self._json(200, {
            **replay,
            "automation_rate": round(auto_n / max(len(items), 1), 4),
            "auto_count": auto_n, "review_count": len(items) - auto_n,
            "auto_evaluated": auto_total, "auto_correct": auto_correct,
            "auto_precision": round(auto_correct / auto_total, 4) if auto_total else None,
            "latency_ms": round(elapsed_ms, 3),
            "latency_per_item_ms": round(elapsed_ms / max(len(items), 1), 6),
            "throughput_per_s": round(len(items) / max(elapsed_ms / 1000.0, 1e-9), 1),
        })


# ==================================================================== main
def main() -> None:
    import torch

    ap = argparse.ArgumentParser(description="哨兵 × jev 0.1.5 集成决策服务")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "ai4i2020.csv"))
    ap.add_argument("--cache", default=os.path.join(ROOT, "heads.pt"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--escalate", action="store_true", help="开启低置信升级")
    ap.add_argument("--escalate-floor", type=float, default=0.80)
    ap.add_argument("--generative", default="local", choices=["local", "stepfun"],
                    help="升级层：本地 OpenAI 兼容端点 或 StepFun 官方 API")
    ap.add_argument("--generative-model",
                    default="modelscope.cn/unsloth/Qwen3-8B-GGUF:latest")
    ap.add_argument("--generative-endpoint",
                    default="http://127.0.0.1:11434/v1/chat/completions")
    ap.add_argument("--generative-max-tokens", type=int, default=2048)
    args = ap.parse_args()

    t0 = time.perf_counter()
    heads, cached = SH.SentinelHeads.load_or_train(
        args.data, args.device, args.cache, retrain=args.retrain)
    SH.export_meta(heads, os.path.join(ROOT, "heads_meta.json"))
    train_s = time.perf_counter() - t0

    support = head_profile_support(args.data)
    cfg = SentinelConfig(escalate=bool(args.escalate),
                         escalate_floor=args.escalate_floor,
                         generative=args.generative,
                         generative_model=args.generative_model,
                         generative_endpoint=args.generative_endpoint,
                         generative_max_tokens=args.generative_max_tokens)
    profile = industrial_profile()
    engine = build_engine(heads, cfg, class_support=support, profile=profile)

    Handler.service = Service(engine=engine)
    Handler.engine = engine
    Handler.cfg = cfg
    Handler.meta = {
        "framework": f"jev-decision-service {FRAMEWORK_VERSION}",
        "decision_head": f"sentinel-mlp@{args.device}",
        "torch": torch.__version__, "device": args.device,
        "data_path": args.data,
        "temperatures": {k: round(v, 4) for k, v in heads.temps.items()},
        "class_support": support,
        "train_seconds": round(train_s, 2), "cache_hit": cached,
        "escalate": cfg.escalate, "escalate_floor": cfg.escalate_floor,
        "generative_tier": (args.generative if cfg.escalate else None),
        "generative_endpoint": (cfg.generative_endpoint if cfg.escalate else None),
        "generative_max_tokens": (cfg.generative_max_tokens if cfg.escalate else None),
        "question_roles": profile.roles,
        "feature_contract": {"names": list(SH.RAW_KEYS),
                             "units": FEATURE_UNITS,
                             "bounds": {k: list(v) for k, v in FEATURE_BOUNDS.items()}},
        "policy": {"auto_confidence": 0.80, "urgent_score": cfg.urgent_score,
                   "next_action_threshold": 0.70},
    }

    print(f"[启动] 框架 jev {FRAMEWORK_VERSION} | 决策头 sentinel-mlp @ {args.device} "
          f"| 缓存命中={cached} | 用时 {train_s:.2f}s")
    print(f"[启动] 逐头温度 {({k: round(v, 3) for k, v in heads.temps.items()})} "
          f"| 升级层={'开(' + args.generative + ')' if cfg.escalate else '关'}")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[就绪] http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def head_profile_support(data_csv: str) -> dict[str, int]:
    counts = {t: 0 for t in SH.TYPES}
    for row in SH.load_rows(data_csv):
        counts[row["type"]] = counts.get(row["type"], 0) + 1
    return counts


if __name__ == "__main__":
    main()
