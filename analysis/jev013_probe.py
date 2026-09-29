#!/usr/bin/env python3
"""0.1.3 综合探针：验证新功能 + 回归旧 bug。

覆盖：
  A. 开源模型 provider 的提示词/解析器匹配（用 mock LLM 复现真实模型输出格式）
  B. StateBuilder 的 OOD 检测
  C. 校准（DecisionHeadProfile + 先验校正）数值是否与手算一致
  D. Policy 对 OOD / 稀有类 / 未校准 的反应
  E. TieredProvider 的分层升级
  F. StructuredHeadProvider 能否接入我们自己的决策头
"""
from __future__ import annotations

import dataclasses
import json
import math
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, "/tmp/jev0113/v013/src")

from jev_service.calibration import DecisionHeadProfile, apply_profile  # noqa: E402
from jev_service.models import (DecisionRequest, Evidence, Prediction,  # noqa: E402
                                Turn)
from jev_service.policy import PolicyConfig, choose_action  # noqa: E402
from jev_service.provider import (OpenSourceProvider, ProviderError,  # noqa: E402
                                 RuleBasedProvider, StructuredHeadProvider,
                                 TieredProvider)
from jev_service.state_builder import StateBuilder  # noqa: E402

PASS, FAIL = "  [OK]", "  [!!]"


# ------------------------------------------------------------------ mock LLM
class MockHandler(BaseHTTPRequestHandler):
    mode = "value"        # value | native

    def log_message(self, *a):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        from jev_service.questions import question_schema
        schema = question_schema()
        answers = {}
        for qid, spec in schema.items():
            if spec["type"] == "choice":
                opts = spec["options"]
                # 真实模型常见写法：用 "value" 键（与解析器期望的 "choice" 不一致）
                if MockHandler.mode == "value":
                    answers[qid] = {"type": "choice", "value": opts[0],
                                    "probabilities": {o: round(1 / len(opts), 4) for o in opts}}
                else:
                    answers[qid] = {"type": "choice", "choice": opts[0],
                                    "probabilities": {o: round(1 / len(opts), 4) for o in opts}}
            elif spec["type"] == "score":
                levels = spec["levels"]
                key = "value" if MockHandler.mode == "value" else "score"
                answers[qid] = {"type": "score", key: 1,
                                "probabilities": {lv: round(1 / len(levels), 4) for lv in levels}}
            else:
                if MockHandler.mode == "value":
                    answers[qid] = {"type": "noul", "value": True,
                                    "probabilities": {"yes": 0.7, "no": 0.3}}
                else:
                    answers[qid] = {"type": "noul", "noul": 0.7}
        body = {"choices": [{"message": {"content": json.dumps({"answers": answers})}}]}
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def start_mock():
    srv = HTTPServer(("127.0.0.1", 0), MockHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}/v1/chat/completions"


def req(msg="你是不是忘了我昨天说过的事？", **state):
    return DecisionRequest(conversation_id="probe", current_message=msg,
                           conversation_state=state)


def main():
    print("=" * 82)
    print("0.1.3 综合探针")
    print("=" * 82)

    # ------------------------------------------------ A. 开源 provider
    print("\n【A】开源模型 provider：提示词与解析器是否匹配")
    srv, endpoint = start_mock()

    for mode, label in [("value", '模型输出 "value" 键（真实模型常见写法）'),
                        ("native", '模型输出 "choice"/"score"/"noul" 键')]:
        MockHandler.mode = mode
        provider = OpenSourceProvider(endpoint=endpoint, model="mock", timeout=5.0)
        try:
            preds = provider.evaluate({"current_message": "测试", "conversation_state": {}})
            print(f"{PASS} {label} → 解析成功，得到 {len(preds)} 条预测")
        except ProviderError as exc:
            print(f"{FAIL} {label} → ProviderError: {exc}")
        except Exception as exc:                                  # noqa: BLE001
            print(f"{FAIL} {label} → {type(exc).__name__}: {exc}")

    # ------------------------------------------------ B. StateBuilder / OOD
    print("\n【B】StateBuilder：OOD 与结构化特征识别")
    sb = StateBuilder(ood_threshold=0.70)
    cases = [
        ("正常文本输入", dict(), {"current_message": "帮我看下昨天那条记录"}),
        ("OOD 分数 0.85（超阈值）", dict(), {"current_message": "风马牛不相及的一句话",
                                        "ood_score": 0.85}),
        ("OOD 分数 0.30（安全）", dict(), {"current_message": "正常一句",
                                        "ood_score": 0.30}),
        ("合法结构化特征", dict(), {"current_message": "设备报警",
                                "structured_features": {"a": 1.0, "b": 2.0}}),
        ("非法结构化特征（含字符串）", dict(), {"current_message": "设备报警",
                                          "structured_features": {"a": "x"}}),
    ]
    for label, kw, st in cases:
        r = DecisionRequest(conversation_id="p", current_message=st.pop("current_message"),
                            conversation_state=st)
        state, report = sb.build(r, [])
        print(f"  {label:<26} feature_status={report.feature_status:<8} "
              f"routing={report.routing_hint:<11} warnings={list(report.warnings)}")

    # ------------------------------------------------ C. 校准数值
    print("\n【C】校准：先验校正是否与手算一致")
    prof = DecisionHeadProfile(name="sentinel-mlp", version="1.0", calibrated=True,
                               temperature=1.0, train_positive_rate=0.0312,
                               calibration_distribution="natural")
    p = Prediction("is_real_fault", "noul", True, {"yes": 0.3232, "no": 0.6768}, None, "head")
    deploy = 0.5
    rep = apply_profile([p], prof, {"deployment_positive_rate": deploy})
    # 手算
    logit = math.log(0.3232 / (1 - 0.3232))
    logit += math.log((deploy / (1 - deploy)) / (0.0312 / (1 - 0.0312)))
    expected = 1 / (1 + math.exp(-logit))
    got = p.probabilities["yes"]
    ok = abs(got - expected) < 1e-6
    print(f"  校正前 0.3232 → 服务算出 {got:.6f}，手算 {expected:.6f}  "
          f"{PASS if ok else FAIL}")
    print(f"  校准状态={p.calibration_status}  report.status={rep.status}  "
          f"warnings={rep.warnings}")

    # 温度缩放单独验证
    p2 = Prediction("is_real_fault", "noul", True, {"yes": 0.9, "no": 0.1}, None, "head")
    prof_t = DecisionHeadProfile(name="t", calibrated=True, temperature=2.0,
                                 train_positive_rate=None)
    apply_profile([p2], prof_t, {})
    lg = math.log(0.9 / 0.1) / 2.0
    exp2 = 1 / (1 + math.exp(-lg))
    print(f"  温度 T=2.0：0.9 → {p2.probabilities['yes']:.6f}，手算 {exp2:.6f}  "
          f"{PASS if abs(p2.probabilities['yes']-exp2) < 1e-6 else FAIL}")

    # 分布漂移检测
    p3 = Prediction("is_real_fault", "noul", True, {"yes": 0.5, "no": 0.5}, None, "head")
    rep3 = apply_profile([p3], prof, {"distribution_id": "line-B"})
    print(f"  分布漂移检测：calibration_distribution=natural + 部署=line-B → "
          f"shifted={rep3.distribution_shifted}, status={rep3.status}, "
          f"warnings={rep3.warnings}")

    # ------------------------------------------------ D. Policy 反应
    print("\n【D】Policy：OOD / 稀有类 / 未校准 是否触发保护")
    base = [Prediction("next_action", "choice", "answer_from_context",
                       {"answer_from_context": 0.9, "ask_clarification": 0.1}, 0.9, "x")]
    scenarios = [
        ("OOD 警告", dict(warnings=["out_of_distribution"])),
        ("稀有类警告", dict(warnings=["rare_classes:TWF,RNF"])),
        ("分布漂移警告", dict(warnings=["distribution_shift"])),
        ("无警告", dict()),
    ]
    for label, kw in scenarios:
        preds = [Prediction(p.id, p.type, p.value, dict(p.probabilities), p.confidence,
                            p.provider, p.calibration_status) for p in base]
        a = choose_action(preds, "普通消息", 1, "open-source", PolicyConfig(), **kw)
        print(f"  {label:<14} → action={a.kind:<19} blocked={a.blocked} "
              f"reasons={a.reason_codes}")

    # 未校准时的保护
    preds = [Prediction(p.id, p.type, p.value, dict(p.probabilities), p.confidence,
                        p.provider, "uncalibrated") for p in base]
    a = choose_action(preds, "普通消息", 1, "open-source", PolicyConfig())
    print(f"  {'未校准概率':<14} → action={a.kind:<19} blocked={a.blocked} "
          f"reasons={a.reason_codes}")

    # abstain 路径
    preds = [Prediction("intent", "choice", "abstain", {"abstain": 1.0}, None, "x")]
    a = choose_action(preds, "普通消息", 1, "open-source", PolicyConfig())
    print(f"  {'模型输出 abstain':<14} → action={a.kind:<19} blocked={a.blocked} "
          f"reasons={a.reason_codes}")

    # ------------------------------------------------ E. TieredProvider
    print("\n【E】TieredProvider：低置信是否升级到生成式")

    class FakeSlow:
        name = "fake-slow"
        last_report = None
        def evaluate(self, state):
            return [Prediction("next_action", "choice", "human_review",
                               {"human_review": 0.99}, 0.99, "fake-slow",
                               "provider_calibrated")]

    class FakeFast:
        name = "fake-fast"
        last_report = None
        def __init__(self, conf):
            self.conf = conf
        def evaluate(self, state):
            return [Prediction("next_action", "choice", "answer_from_context",
                               {"answer_from_context": 0.6, "x": 0.4}, self.conf,
                               "fake-fast", "temperature_scaled")]

    for conf, label in [(0.95, "快速头高置信 0.95"), (0.40, "快速头低置信 0.40")]:
        tp = TieredProvider(fast=FakeFast(conf), generative=FakeSlow(),
                            fast_confidence_floor=0.8)
        out = tp.evaluate({})
        print(f"  {label:<20} → route={tp.last_route:<10} reason={tp.last_reason}")

    # ------------------------------------------------ F. StructuredHeadProvider
    print("\n【F】StructuredHeadProvider：能否接入我们自己的决策头")

    def our_head(features):
        """模拟哨兵决策头：吃 9 维数值特征，吐三问答案。"""
        from jev_service.questions import question_schema
        schema = question_schema()
        wear = float(features.get("tool_wear_min", 0))
        fault = wear > 200
        answers = {}
        for qid, spec in schema.items():
            if spec["type"] == "choice":
                opts = spec["options"]
                pick = opts[0]
                answers[qid] = {"type": "choice", "choice": pick,
                                "probabilities": {o: round(1 / len(opts), 4) for o in opts}}
            elif spec["type"] == "score":
                levels = spec["levels"]
                answers[qid] = {"type": "score", "score": 3 if fault else 0,
                                "probabilities": {lv: round(1 / len(levels), 4) for lv in levels}}
            else:
                answers[qid] = {"type": "noul", "noul": 0.95 if fault else 0.02}
        return answers

    prof2 = DecisionHeadProfile(name="sentinel-mlp", version="d2", calibrated=True,
                                temperature=1.1, train_positive_rate=0.0312,
                                calibration_distribution="natural",
                                class_support={"TWF": 6, "RNF": 6}, min_class_support=1)
    shp = StructuredHeadProvider(evaluator=our_head, profile=prof2)
    try:
        st = {"conversation_state": {"structured_features": {"tool_wear_min": 243}}}
        preds = shp.evaluate(st)
        print(f"{PASS} 注入自定义决策头 → {len(preds)} 条预测；"
              f"校准状态={preds[0].calibration_status}；"
              f"report={shp.last_report.status}, warnings={shp.last_report.warnings}")
    except Exception as exc:                                      # noqa: BLE001
        print(f"{FAIL} 注入失败：{type(exc).__name__}: {exc}")

    # 缺特征时是否优雅报错
    try:
        shp.evaluate({"conversation_state": {}})
        print(f"{FAIL} 缺特征时应报错，但通过了")
    except ProviderError as exc:
        print(f"{PASS} 缺结构化特征 → ProviderError(retryable={exc.retryable}): {exc}")

    srv.shutdown()
    print("\n" + "=" * 82)


if __name__ == "__main__":
    main()
