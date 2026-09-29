#!/usr/bin/env python3
"""节点侧补丁：修复 jev-decision-service 在 ollama 上的两个部署问题。

1) ollama 默认 num_predict=128，六个问题的嵌套 JSON 会被截断 -> 形状校验失败。
   修复：在 OpenAI 兼容请求里显式加 max_tokens。
2) 形状校验失败被标为 retryable=False，ResilientProvider 不降级，直接 500。
   修复：改为 retryable=True，让服务优雅降级到 rules 并标记 degraded。
"""
import re
import sys

PATH = "/home/USER/jev-service/src/jev_service/provider.py"

with open(PATH, encoding="utf-8") as handle:
    text = handle.read()

changes = []

# --- 补丁 1：提高生成长度上限 -------------------------------------------------
if '"max_tokens"' not in text:
    before = text
    text = text.replace(
        '            "temperature": 0,\n',
        '            "temperature": 0,\n            "max_tokens": 1024,\n',
        1,
    )
    changes.append("max_tokens" if text != before else "max_tokens(未匹配)")

# --- 补丁 2：形状校验失败改为可重试，触发降级 ---------------------------------
pairs = [
    (
        'raise ProviderError(f"invalid {provider} answer shape: {exc}", retryable=False)',
        'raise ProviderError(f"invalid {provider} answer shape: {exc}", retryable=True)',
    ),
    (
        'raise ProviderError(f"{provider} response contained unexpected or missing decision ids", retryable=False)',
        'raise ProviderError(f"{provider} response contained unexpected or missing decision ids", retryable=True)',
    ),
    (
        'raise ProviderError(f"{provider} returned an unsupported choice for {prediction.id}", retryable=False)',
        'raise ProviderError(f"{provider} returned an unsupported choice for {prediction.id}", retryable=True)',
    ),
]
for old, new in pairs:
    if old in text:
        text = text.replace(old, new, 1)
        changes.append("retryable=True x1")

if not changes:
    print("无需修改（补丁可能已应用）")
    sys.exit(0)

with open(PATH, "w", encoding="utf-8") as handle:
    handle.write(text)

print("已应用补丁:")
for item in changes:
    print("  -", item)
