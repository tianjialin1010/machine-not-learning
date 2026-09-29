#!/usr/bin/env python3
"""jev-decision-service 完整修复补丁（3 个文件、7 处修改）。

【重要】0.1.4（2026-09-27）已自行修复本补丁覆盖的全部问题，
补丁现在只适用于 0.1.0 ~ 0.1.3。在 0.1.4+ 上执行会全部报「未匹配」，
属于预期行为，不是错误。执行 `--check` 可确认。

覆盖三类问题
------------
【A】opensource 路径完全不可用（provider.py，5 处）
  真机实测 0/3 成功。解析器只认原生键名，而提示词从未要求；
  另有 max_tokens 截断、概率和不为 1、score 越界三类格式瑕疵。

【B】启用校准后「紧急度保护」静默失效（calibration.py，1 处）—— 严重
  apply_profile 的多元分支把 value 无条件改写成 argmax 的**标签名**，
  score 类型因此从数值 3.0 变成字符串 '立即'；
  而 policy 里判断紧急度用的是 isinstance(value, (float, int))，
  类型一变整体失效 —— 高危报警会被当作普通消息直接回复。
  实测：severity=3 时，calibrated=False 走 human_review，
        calibrated=True 变成 answer_from_context。

【C】「稀有类」警告是静态全局的，声明即永久转人工（policy.py，1 处）—— 严重
  apply_profile 只要发现 profile 里有任意样本数低于阈值的类，
  就**无条件**往 warnings 里塞 rare_classes；policy 随即拦截一切请求。
  即「越诚实标注数据局限，系统越不可用」。
  实测：同一台健康设备，未声明→answer_from_context；声明了→human_review+blocked。
  合理做法：仅当**本次预测命中**稀有类时才告警。

用法
----
    python3 patch_jev_service.py /path/to/package_root          # 含 src/ 的包根目录
    python3 patch_jev_service.py --check /path/to/package_root  # 只检测
    python3 patch_jev_service.py /path/to/src/jev_service/provider.py   # 单文件亦可

补丁幂等，可重复执行；应用后请跑 `PYTHONPATH=src python -m unittest discover -s tests`。

版本适配
--------
  0.1.0 ~ 0.1.3  适用（这四版 opensource 路径均为 0/3 失败）
  0.1.4 及以后   不需要，官方已自行修复（真机实测 3/3 通过）
"""
from __future__ import annotations

import os
import sys

# ============================================================ 【A】provider.py
A_OLD_SYSTEM = '''        system = (
            "You are a typed decision classifier inside a software system. Do not write a reply. "
            "Return JSON only with an answers object. Answer every question using the declared "
            "choice options, numeric score levels, or noul yes probability. Probabilities must "
            "sum to 1. Confidence is a raw self-estimate and is not calibrated.\\n\\n"
            f"Question schema:\\n{schema}"
        )'''

A_NEW_SYSTEM = '''        system = (
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

A_OLD_PAYLOAD = '''        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},'''

A_NEW_PAYLOAD = '''        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},'''

A_OLD_FIELD = '''    api_key: str = "EMPTY"
    timeout: float = 30.0
    name: str = "open-source"'''

A_NEW_FIELD = '''    api_key: str = "EMPTY"
    timeout: float = 30.0
    max_tokens: int = 1024          # ollama 默认仅 128，多问嵌套 JSON 会被截断
    name: str = "open-source"'''

A_OLD_ANCHOR = '''def _parse_json_object(content: str) -> dict[str, Any]:'''

A_NEW_ANCHOR = '''def _normalize_raw_answers(raw_answers: dict[str, Any]) -> dict[str, Any]:
    """把模型输出清洗成校验器能接受的形状（只做容错，不改语义）。

    真机实测（Qwen3-8B via ollama）暴露四类格式瑕疵，任何一类都会让整单请求
    以 retryable=False 硬失败：value 键名、概率和不为 1、score 越界、概率非数值。
    """
    for answer in raw_answers.values():
        if not isinstance(answer, dict):
            continue
        answer_type = str(answer.get("type", ""))
        probabilities = answer.get("probabilities")
        if not isinstance(probabilities, dict) or not probabilities:
            continue

        if "value" in answer:
            if answer_type == "choice" and "choice" not in answer:
                answer["choice"] = answer["value"]
            elif answer_type == "score" and "score" not in answer:
                answer["score"] = answer["value"]

        try:
            values = {str(key): float(value) for key, value in probabilities.items()}
        except (TypeError, ValueError):
            continue
        if any(not math.isfinite(value) or value < 0 for value in values.values()):
            continue

        total = sum(values.values())
        if total > 0 and abs(total - 1.0) > 1e-9:
            values = {key: value / total for key, value in values.items()}
        answer["probabilities"] = values

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
            answer["score"] = min(max(score, 0.0), float(max(0, len(values) - 1)))

    return raw_answers


def _parse_json_object(content: str) -> dict[str, Any]:'''

A_OLD_CALL = '''            raw_answers = parsed.get("answers", parsed)
            if not isinstance(raw_answers, dict):
                raise ProviderError("open-source response did not contain an answers object", retryable=True)
            predictions = _validate_answer_set(raw_answers, self.name, self.questions)'''

A_NEW_CALL = '''            raw_answers = parsed.get("answers", parsed)
            if not isinstance(raw_answers, dict):
                raise ProviderError("open-source response did not contain an answers object", retryable=True)
            raw_answers = _normalize_raw_answers(raw_answers)
            predictions = _validate_answer_set(raw_answers, self.name, self.questions)'''

# ============================================================ 【B】calibration.py
B_OLD = '''        elif profile.calibrated and prediction.probabilities:
            keys = list(prediction.probabilities)
            logits = [math.log(max(prediction.probabilities[key], 1e-9)) for key in keys]
            scaled = _softmax([value / max(profile.temperature, 1e-6) for value in logits])
            prediction.probabilities = {key: round(value, 6) for key, value in zip(keys, scaled)}
            prediction.value = max(prediction.probabilities, key=prediction.probabilities.get)
            prediction.calibration_status = "temperature_scaled"'''

B_NEW = '''        elif profile.calibrated and prediction.probabilities:
            keys = list(prediction.probabilities)
            logits = [math.log(max(prediction.probabilities[key], 1e-9)) for key in keys]
            scaled = _softmax([value / max(profile.temperature, 1e-6) for value in logits])
            prediction.probabilities = {key: round(value, 6) for key, value in zip(keys, scaled)}
            # 注意：只有 choice 类型才把 value 设为概率最高的**标签名**。
            # score 类型的 value 必须保持数值——policy 用
            # isinstance(tension.value, (float, int)) 判断紧急度，
            # 一旦被改写成字符串，紧急度保护会静默失效。
            if prediction.type == "choice":
                prediction.value = max(prediction.probabilities,
                                       key=prediction.probabilities.get)
            elif prediction.type == "score":
                best = max(range(len(keys)), key=lambda index: scaled[index])
                prediction.value = float(best)
            prediction.calibration_status = "temperature_scaled"'''

# ============================================================ 【C】policy.py
C_OLD = '''    if "rare_classes" in " ".join(warnings):
        reasons.append("rare_class_support_insufficient")
        return ActionRecommendation("human_review", True, reasons, None, provider_status, True)'''

C_NEW = '''    # 稀有类告警只在**本次预测确实命中**稀有类时拦截；
    # 否则只要 profile 声明过任何小样本类，所有请求都会被永久转人工
    # ——「越诚实标注数据局限，系统越不可用」。
    rare_labels = set()
    for warning in warnings:
        if warning.startswith("rare_classes:"):
            rare_labels |= {item for item in warning.split(":", 1)[1].split(",") if item}
    if rare_labels:
        predicted_labels = set()
        for prediction in predictions:
            if prediction.type == "choice" and isinstance(prediction.value, str):
                predicted_labels.add(prediction.value)
            elif prediction.type == "score" and prediction.probabilities:
                predicted_labels.add(max(prediction.probabilities,
                                         key=prediction.probabilities.get))
        if predicted_labels & rare_labels:
            reasons.append("rare_class_support_insufficient")
            return ActionRecommendation("human_review", True, reasons, None,
                                        provider_status, True)'''

PATCHES = {
    "provider.py": [
        ("opensource system prompt（原生键名）", A_OLD_SYSTEM, A_NEW_SYSTEM, "never emit a"),
        ("payload max_tokens", A_OLD_PAYLOAD, A_NEW_PAYLOAD, '"max_tokens": self.max_tokens'),
        ("dataclass max_tokens 字段", A_OLD_FIELD, A_NEW_FIELD, "max_tokens: int = 1024"),
        ("输出清洗函数", A_OLD_ANCHOR, A_NEW_ANCHOR, "_normalize_raw_answers"),
        ("接入清洗调用", A_OLD_CALL, A_NEW_CALL,
         "raw_answers = _normalize_raw_answers("),
    ],
    "calibration.py": [
        ("score 类型 value 不应被改写为标签名", B_OLD, B_NEW,
         'prediction.type == "score"'),
    ],
    "policy.py": [
        ("稀有类告警改为「命中才拦截」", C_OLD, C_NEW, "rare_labels = set()"),
    ],
}

MARKERS = ["never emit a", "max_tokens", "_normalize_raw_answers",
           "prediction.type == \"score\"", "rare_labels"]


def resolve(package_root: str) -> dict[str, str]:
    if package_root.endswith(".py"):
        return {os.path.basename(package_root): package_root}
    base = os.path.join(package_root, "src", "jev_service")
    if not os.path.isdir(base):
        base = package_root
    return {name: os.path.join(base, name) for name in PATCHES}


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    check_only = "--check" in sys.argv
    if not args:
        print(__doc__)
        return 2

    targets = resolve(args[0])
    missing = [n for n, p in targets.items() if not os.path.isfile(p)]
    if missing:
        print(f"找不到文件：{missing}")
        return 2

    if check_only:
        for name, path in targets.items():
            text = open(path, encoding="utf-8").read()
            markers = [m for _, _, _, m in PATCHES[name]]
            done = sum(1 for m in markers if m in text)
            print(f"  {name:<18} 已应用 {done}/{len(markers)}")
        return 0

    total = 0
    for name, path in targets.items():
        text = open(path, encoding="utf-8").read()
        applied, skipped = [], []
        for label, old, new, marker in PATCHES[name]:
            if marker in text:
                skipped.append(f"{label}（已应用）")
                continue
            if old in text:
                text = text.replace(old, new, 1)
                applied.append(label)
            else:
                skipped.append(f"{label}（未匹配）")
        if applied:
            open(path, "w", encoding="utf-8").write(text)
        total += len(applied)
        print(f"  {name:<18} 应用 {len(applied)} 处" +
              (f"；跳过 {len(skipped)}" if skipped else ""))
        for label in applied:
            print(f"      + {label}")

    print(f"\n合计应用 {total} 处修改。")
    print("建议随后执行：PYTHONPATH=src python -m unittest discover -s tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
