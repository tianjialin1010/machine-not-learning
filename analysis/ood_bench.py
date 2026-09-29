#!/usr/bin/env python3
"""补充测试：各版本对 OOD（分布外输入）的防护能力。
对同一批样本注入 ood_score，看各版本是否拦截。"""
import csv, inspect, json, math, os, sys
from collections import Counter
import numpy as np

BASE = sys.argv[1]
sys.path.insert(0, os.path.join(BASE, "src"))
from jev_service.calibration import DecisionHeadProfile
from jev_service.models import DecisionRequest
from jev_service.engine import DecisionEngine
from jev_service.policy import PolicyConfig, choose_action
from jev_service.provider import StructuredHeadProvider
from jev_service.questions import QuestionProfile
from jev_service.state_builder import StateBuilder

DATA = "/home/USER/jev-service/data/ai4i2020.csv"
SEV = ["观察", "计划", "尽快", "立即"]
PROF = QuestionProfile(name="ind", version="v1", questions={
    "risk_level": {"type": "score", "levels": SEV, "question": "优先级"},
    "needs_evidence": {"type": "noul", "question": "查手册?"},
    "needs_clarification": {"type": "noul", "question": "需确认?"},
    "allows_side_effect": {"type": "noul", "question": "自动派单?"},
    "next_action": {"type": "choice", "question": "下一步",
                    "options": ["answer_from_context", "retrieve_evidence",
                                "ask_clarification", "draft_suggestion", "human_review"]}})


def ev(features):
    return {
        "risk_level": {"type": "score", "score": 0,
                       "probabilities": {SEV[i]: (0.97 if i == 0 else 0.01) for i in range(4)}},
        "needs_evidence": {"type": "noul", "noul": 0.1},
        "needs_clarification": {"type": "noul", "noul": 0.05},
        "allows_side_effect": {"type": "noul", "noul": 0.9},
        "next_action": {"type": "choice", "choice": "answer_from_context",
                        "probabilities": {o: (0.9 if o == "answer_from_context" else 0.025)
                                          for o in PROF.questions["next_action"]["options"]}},
    }


def main():
    profile = DecisionHeadProfile(name="p", version="1", calibrated=True,
                                  temperature=1.0, calibration_distribution="d",
                                  label_provenance="model")
    provider = StructuredHeadProvider(evaluator=ev, profile=profile,
                                      questions=PROF, name="t")
    engine = DecisionEngine(provider=provider, policy=PolicyConfig(urgent_score=3.0))

    rows = list(csv.DictReader(open(DATA, encoding="utf-8-sig")))
    rng = np.random.default_rng(7)
    sel = rng.choice(len(rows), 200, replace=False)
    res = Counter()
    for i in sel:
        f = {"air_temp_c": float(rows[i]["Air temperature [K]"]) - 273.15,
             "process_temp_c": float(rows[i]["Process temperature [K]"]) - 273.15,
             "rpm": float(rows[i]["Rotational speed [rpm]"]),
             "torque_nm": float(rows[i]["Torque [Nm]"]),
             "tool_wear_min": float(rows[i]["Tool wear [min]"])}
        req = DecisionRequest(conversation_id=f"o{i}", current_message="设备状态上报",
                              conversation_state={"structured_features": f,
                                                  "ood_score": 0.9})
        resp = engine.decide(req)
        res[resp.recommended_action.kind] += 1
    blocked = res.get("human_review", 0) + res.get("abstain", 0)
    print(json.dumps({"version": os.path.basename(BASE.rstrip('/')),
                      "n": len(sel), "ood_blocked": blocked,
                      "ood_block_rate": round(blocked / len(sel), 4),
                      "actions": dict(res)}, ensure_ascii=False))


main()
