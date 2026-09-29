"""诊断：升级到生成式时到底发生了什么。"""
import json, os, sys, time
sys.path.insert(0, "/home/USER/sentinel-stack/vendor")
sys.path.insert(0, "/home/USER/sentinel-stack")
from jev_service.provider import OpenSourceProvider, ProviderError, TieredProvider
from jev_service.questions import QuestionProfile
import sentinel_integrated as SI

prof = SI.industrial_profile()
gen = OpenSourceProvider(
    endpoint="http://127.0.0.1:11434/v1/chat/completions",
    model="modelscope.cn/unsloth/Qwen3.8-27B-GGUF:latest",
    timeout=300.0, questions=prof, name="local-generative")

state = {"current_message": "设备状态上报",
         "conversation_state": {"structured_features": {
             "air_temp_c": 24.1, "process_temp_c": 37.6, "rpm": 1380,
             "torque_nm": 62.3, "tool_wear_min": 243}}}
t0 = time.time()
try:
    preds = gen.evaluate(state)
    print("  [成功] 耗时 %.1fs" % (time.time() - t0))
    for p in preds:
        print("   %-20s %-7s %s" % (p.id, p.type, str(p.value)[:30]))
except ProviderError as exc:
    print("  [失败] 耗时 %.1fs  retryable=%s" % (time.time() - t0, exc.retryable))
    print("   错误: %s" % exc)
