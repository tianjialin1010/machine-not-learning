#!/usr/bin/env python3
"""给 jev-decision-service 的补丁：修复开源模型（vLLM/Ollama）路径不可用的问题。

背景
----
`_parse_jev_answer` 只接受原生键名（choice / score / noul），
但 `OpenSourceProvider` 的 system prompt 从未要求模型输出这些键名，
只说 "using the declared choice options, numeric score levels, or noul yes probability"。
真实模型（实测 Qwen3-27B / 8B）会输出 {"type":"choice","value":"..."}，
于是必然抛 `ProviderError: choice is missing from probabilities for <id>`。

另外 ollama 的 OpenAI 兼容接口默认只生成 128 token，
六个问题的嵌套 JSON 会被截断——所以必须显式给 max_tokens。

本补丁做四件事
--------------
1. 重写 system prompt：明确要求原生键名、给出三种形状与示例、禁止 value 键；
2. payload 增加 max_tokens（默认 1024，可用参数覆盖）；
3. 新增 max_tokens 字段到 dataclass，便于不同后端调参；
4. 在交给校验器前清洗模型输出（_normalize_raw_answers），兜住真机实测到的
   四类格式瑕疵：value 键名、概率和不为 1、score 越界、概率值非数值。
   任何一类都会让整单请求以 retryable=False 硬失败。

用法
----
    python3 patch_opensource_provider.py /path/to/src/jev_service/provider.py
    python3 patch_opensource_provider.py --check /path/to/provider.py   # 仅检查是否已修补
"""
from __future__ import annotations

import sys

OLD_SYSTEM = '''        system = (
            "You are a typed decision classifier inside a software system. Do not write a reply. "
            "Return JSON only with an answers object. Answer every question using the declared "
            "choice options, numeric score levels, or noul yes probability. Probabilities must "
            "sum to 1. Confidence is a raw self-estimate and is not calibrated.\\n\\n"
            f"Question schema:\\n{schema}"
        )'''

NEW_SYSTEM = '''        system = (
            "You are a typed decision classifier inside a software system. Do not write a reply.\\n"
            "Return JSON only (no markdown fences) with a top-level 'answers' object, one entry per "
            "question id, using EXACTLY these shapes and key names:\\n"
            '  - choice: {"type":"choice","choice":"<one of the declared options>",'
            '"probabilities":{"<option>":<p>,...}}\\n'
            '  - score:  {"type":"score","score":<number>,"probabilities":'
            '{"<level>":<p>,...}}\\n'
            '  - noul:   {"type":"noul","noul":<0.0-1.0>}\\n'
            "Rules: never emit a \\'value\\' key; every probabilities object must sum to 1; "
            "the chosen option/score must be one of the declared ones; "
            "answer EVERY question id listed in the schema.\\n"
            'Example output: {"answers":{"needs_evidence":{"type":"noul","noul":0.8}}}\\n\\n'
            f"Question schema:\\n{schema}"
        )'''

OLD_PAYLOAD = '''        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},'''

NEW_PAYLOAD = '''        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},'''

# 第 4 处：LLM 输出的概率和不精确为 1（实测 8B 会给出 0.85/0.4 这类值），
# 而校验器只容忍 0.01 偏差且 retryable=False，会把整单请求打死。
# 这里在交给校验器之前做温和归一化，改动局部、不触碰共用解析器。
OLD_NORMALIZE_ANCHOR = '''def _parse_json_object(content: str) -> dict[str, Any]:'''

NEW_NORMALIZE_ANCHOR = '''def _normalize_raw_answers(raw_answers: dict[str, Any]) -> dict[str, Any]:
    """把模型输出清洗成校验器能接受的形状（不改语义，只做容错）。

    真机实测（Qwen3-8B via ollama）暴露四类格式瑕疵，任何一类都会让整单请求
    以 retryable=False 硬失败：

      1. 用 "value" 键而不是 "choice"/"score"——解析器只认原生键名；
      2. 概率和不为 1（如 0.4 + 0.3 + 0.2），校验器只容忍 0.01 偏差；
      3. score 超出 [0, 选项数-1]（模型把四档理解成 1~4）；
      4. 概率值缺失或非数值。

    这里逐项兜住，让后续的合法性校验仍能正常发挥作用。
    """
    for answer in raw_answers.values():
        if not isinstance(answer, dict):
            continue
        answer_type = str(answer.get("type", ""))
        probabilities = answer.get("probabilities")
        if not isinstance(probabilities, dict) or not probabilities:
            continue

        # (1) value -> choice / score
        if "value" in answer:
            if answer_type == "choice" and "choice" not in answer:
                answer["choice"] = answer["value"]
            elif answer_type == "score" and "score" not in answer:
                answer["score"] = answer["value"]

        # (4) 概率值转数值
        try:
            values = {str(key): float(value) for key, value in probabilities.items()}
        except (TypeError, ValueError):
            continue
        if any(not math.isfinite(value) or value < 0 for value in values.values()):
            continue

        # (2) 归一化
        total = sum(values.values())
        if total > 0 and abs(total - 1.0) > 1e-9:
            values = {key: value / total for key, value in values.items()}
        answer["probabilities"] = values

        # (3) score 限幅到声明区间；同时把「选项名」形式的 score 映射为下标
        if answer_type == "score" or "score" in answer:
            raw_score = answer.get("score")
            if isinstance(raw_score, str) and raw_score in values:
                raw_score = list(values).index(raw_score)
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(score):
                continue
            upper = float(max(0, len(values) - 1))
            answer["score"] = min(max(score, 0.0), upper)

    return raw_answers


def _parse_json_object(content: str) -> dict[str, Any]:'''

OLD_CALL = '''            raw_answers = parsed.get("answers", parsed)
            if not isinstance(raw_answers, dict):
                raise ProviderError("open-source response did not contain an answers object", retryable=True)
            predictions = _validate_answer_set(raw_answers, self.name, self.questions)'''

NEW_CALL = '''            raw_answers = parsed.get("answers", parsed)
            if not isinstance(raw_answers, dict):
                raise ProviderError("open-source response did not contain an answers object", retryable=True)
            raw_answers = _normalize_raw_answers(raw_answers)
            predictions = _validate_answer_set(raw_answers, self.name, self.questions)'''

OLD_FIELD = '''    api_key: str = "EMPTY"
    timeout: float = 30.0
    name: str = "open-source"'''

NEW_FIELD = '''    api_key: str = "EMPTY"
    timeout: float = 30.0
    max_tokens: int = 1024          # ollama 默认仅 128，多问嵌套 JSON 会被截断
    name: str = "open-source"'''


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    check_only = "--check" in sys.argv
    if not args:
        print(__doc__)
        return 2
    path = args[0]
    try:
        text = open(path, encoding="utf-8").read()
    except OSError as exc:
        print(f"无法读取 {path}: {exc}")
        return 2

    if "never emit a" in text and "max_tokens" in text and "_normalize_raw_answers" in text:
        print("已修补过，跳过。")
        return 0
    if check_only:
        print("尚未修补：system prompt 未要求原生键名 / 缺 max_tokens / 缺概率归一化")
        return 1

    applied = []
    for name, old, new in [("system prompt", OLD_SYSTEM, NEW_SYSTEM),
                           ("payload max_tokens", OLD_PAYLOAD, NEW_PAYLOAD),
                           ("dataclass 字段", OLD_FIELD, NEW_FIELD),
                           ("输出清洗函数", OLD_NORMALIZE_ANCHOR, NEW_NORMALIZE_ANCHOR),
                           ("接入清洗调用", OLD_CALL, NEW_CALL)]:
        if old in text:
            text = text.replace(old, new, 1)
            applied.append(name)
        else:
            print(f"  [跳过] {name}：未匹配到原文（可能已改过）")

    if not applied:
        print("没有可应用的部分，文件未改动。")
        return 1

    open(path, "w", encoding="utf-8").write(text)
    print(f"已应用 {len(applied)} 处修改：{', '.join(applied)}")
    print(f"写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
