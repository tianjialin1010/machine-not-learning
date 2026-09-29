#!/usr/bin/env python3
"""真机对比：jev-decision-service 的 opensource 路径（原版 vs 补丁版）。

用节点上的 ollama 真模型跑，验证「提示词未要求原生键名」这个缺陷的影响。
用法：python3 jev_real_test.py <包目录>
"""
from __future__ import annotations

import os
import sys
import time

BASE = sys.argv[1]
sys.path.insert(0, os.path.join(BASE, "src"))

os.environ["JEV_PROVIDER"] = "opensource"
os.environ["OPEN_SOURCE_ENDPOINT"] = "http://127.0.0.1:11434/v1/chat/completions"
os.environ["OPEN_SOURCE_MODEL"] = os.environ.get(
    "MODEL", "modelscope.cn/unsloth/Qwen3-8B-GGUF:latest")
os.environ["OPEN_SOURCE_API_KEY"] = "ollama"

from jev_service.provider import ProviderError, build_provider  # noqa: E402

print(f"  包目录: {BASE}")
print(f"  模型:   {os.environ['OPEN_SOURCE_MODEL']}")

provider = build_provider()
state = {
    "current_message": "你是不是忘了我昨天说过的事？",
    "conversation_state": {},
    "retrieved_evidence": [],
    "recent_turns": [],
}

t0 = time.perf_counter()
try:
    preds = provider.evaluate(state)
    dt = (time.perf_counter() - t0) * 1000
    print(f"  [通过] 耗时 {dt:.0f} ms，解析出 {len(preds)} 条类型化决策：")
    for x in preds:
        probs = dict(sorted((x.probabilities or {}).items(),
                            key=lambda kv: -kv[1])[:2])
        print(f"      {x.id:<20} {x.type:<7} {str(x.value)[:24]:<26} "
              f"{probs}  [{x.calibration_status}]")
except ProviderError as exc:
    print(f"  [失败] ProviderError: {exc}")
    print(f"         retryable={getattr(exc, 'retryable', '?')}")
except Exception as exc:                                       # noqa: BLE001
    print(f"  [失败] {type(exc).__name__}: {exc}")
