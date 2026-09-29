"""验证：opensource provider 即使格式修好，能否走到自动化分支。"""
import os, sys
BASE = sys.argv[1]
sys.path.insert(0, os.path.join(BASE, "src"))
from jev_service.calibration import DecisionHeadProfile
from jev_service.provider import OpenSourceProvider
from jev_service.policy import PolicyConfig, choose_action

EP = "http://127.0.0.1:11434/v1/chat/completions"
MDL = "modelscope.cn/unsloth/Qwen3-8B-GGUF:latest"
STATE = {"current_message": "你是不是忘了我昨天说过的事？",
         "conversation_state": {}, "retrieved_evidence": []}

print("  【情形 1】默认 profile（calibrated=False）")
p1 = OpenSourceProvider(endpoint=EP, model=MDL, timeout=180)
preds = p1.evaluate(STATE)
print(f"    calibration_status = {preds[0].calibration_status}")
a = choose_action(preds, STATE["current_message"], 0, "open-source", PolicyConfig(),
                  warnings=list(p1.last_report.warnings))
print(f"    policy → action={a.kind}  blocked={a.blocked}  reasons={a.reason_codes}")

print()
print("  【情形 2】注入我们 D2 已标定的 profile（T=1.14, π_train=0.0312）")
prof = DecisionHeadProfile(name="sentinel-mlp", version="d2", calibrated=True,
                           temperature=1.14, train_positive_rate=0.0312,
                           calibration_distribution="ai4i-natural",
                           label_provenance="derived_rule")
p2 = OpenSourceProvider(endpoint=EP, model=MDL, timeout=180, profile=prof)
preds2 = p2.evaluate(STATE)
print(f"    calibration_status = {preds2[0].calibration_status}")
noul = next(x for x in preds2 if x.type == "noul")
print(f"    首个 noul 概率 = {noul.probabilities}")
a2 = choose_action(preds2, STATE["current_message"], 0, "open-source", PolicyConfig(),
                   warnings=list(p2.last_report.warnings))
print(f"    policy → action={a2.kind}  blocked={a2.blocked}  reasons={a2.reason_codes}")
