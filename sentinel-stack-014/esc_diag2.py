"""对比三种生成式配置，找出可用的升级层。"""
import json, sys, time, urllib.request
sys.path.insert(0, "/home/USER/sentinel-stack/vendor")
sys.path.insert(0, "/home/USER/sentinel-stack")
from jev_service.provider import OpenSourceProvider, ProviderError
import sentinel_integrated as SI

PROF = SI.industrial_profile()
STATE = {"current_message": "设备状态上报",
         "conversation_state": {"structured_features": {
             "air_temp_c": 24.1, "process_temp_c": 37.6, "rpm": 1380,
             "torque_nm": 62.3, "tool_wear_min": 243}}}
M27 = "modelscope.cn/unsloth/Qwen3.8-27B-GGUF:latest"
M8 = "modelscope.cn/unsloth/Qwen3-8B-GGUF:latest"


def try_provider(label, model, patch_think):
    prov = OpenSourceProvider(endpoint="http://127.0.0.1:11434/v1/chat/completions",
                              model=model, timeout=300.0, questions=PROF, name="gen")
    if patch_think:
        import jev_service.provider as P
        orig = P.urllib.request.Request

        def patched(url, data=None, headers=None, method=None):
            if data:
                obj = json.loads(data.decode())
                obj["think"] = False          # ollama 扩展参数：关闭思考链
                data = json.dumps(obj).encode()
            return orig(url, data=data, headers=headers, method=method)
        P.urllib.request.Request = patched
    t0 = time.time()
    try:
        preds = prov.evaluate(STATE)
        print("  %-34s [成功] %.1fs" % (label, time.time() - t0))
        for p in preds:
            print("      %-20s %-7s %s" % (p.id, p.type, str(p.value)[:26]))
        return True
    except ProviderError as exc:
        print("  %-34s [失败] %.1fs  %s" % (label, time.time() - t0, str(exc)[:90]))
        return False
    finally:
        if patch_think:
            P.urllib.request.Request = orig


print("---- A. 8B 原样 ----")
try_provider("8B(原样)", M8, False)
print("---- B. 27B 原样 ----")
try_provider("27B(原样)", M27, False)
print("---- C. 27B + think=false ----")
try_provider("27B(think=false)", M27, True)
