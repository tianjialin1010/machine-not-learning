#!/usr/bin/env python3
"""生成《jev-decision-service 0.1.3 opensource 路径缺陷与修复》技术反馈单 PDF。"""
from __future__ import annotations

import sys

sys.path.insert(0, "/Users/jialin_tian/WorkBuddy/2026-09-20-02-38-44")
from mkpdf import build_pdf  # noqa: E402

OUT = ("/Users/jialin_tian/WorkBuddy/2026-09-20-02-38-44/"
       "jev 0.1.3 opensource路径缺陷与修复（反馈单）.pdf")

blocks = []

blocks.append({"h2": "0　结论"})
blocks.append({"p": "0.1.3 的框架、校准、OOD、分层路由这几块做得很好，"
                    "但在<b>真机上 opensource 路径（对接 vLLM / Ollama 本地模型）无法工作</b>。"
                    "我们用节点上的 ollama + Qwen3-8B 实测：原版<b>连续 3 次全部失败</b>；"
                    "打好补丁后<b>连续 3 次全部成功</b>。"})
blocks.append({"kpi": [("0 / 3", "原版真机成功率"), ("3 / 3", "补丁后成功率"),
                      ("5 处", "补丁修改点"), ("13.2 s", "单次决策耗时")]})
blocks.append({"note": "根因不是模型不行，而是<b>解析器要求的键名和提示词要求的键名不一致</b>。"
                       "单测只喂了手工构造的标准格式，所以一直没暴露。", "level": "bad"})

blocks.append({"h2": "1　缺陷一：提示词与解析器键名不匹配"})
blocks.append({"table": {
    "header": ["位置", "现状", "问题"],
    "rows": [
        ["_parse_jev_answer",
         "只接受 {\"type\":\"choice\",\"choice\":...} / \"score\" / \"noul\"",
         "—（这是对的，Jev 原生契约）"],
        ["OpenSourceProvider 的 system prompt",
         "只说 “Answer every question using the declared choice options, numeric score "
         "levels, or noul yes probability”",
         "从未要求模型把类型的值放在 choice / score / noul 这些键上"],
        ["结果",
         "模型按字面理解，输出 {\"type\":\"choice\",\"value\":\"clarification\",...}",
         "解析器找不到 choice 键 → ProviderError: choice is missing from probabilities"],
    ],
    "widths": [1.35, 2.1, 1.45],
}})
blocks.append({"p": "<b>证据</b>：逐字节比对显示，<code>_parse_jev_answer</code> 在 0.1.0 与 0.1.3 中"
                    "完全相同（2574 字节），而提示词也一字未改——这个缺陷从 0.1.0 一路带到了 0.1.3。"})
blocks.append({"p": "<b>修复</b>：重写 system prompt，明确列出三种形状与键名、给出示例、"
                    "并显式禁止 <code>value</code> 键。"})

blocks.append({"h2": "2　缺陷二：真机上还有三类格式瑕疵"})
blocks.append({"p": "把键名问题修好之后，真机又连续暴露了三类新的失败。"
                    "这说明<b>不是改一句话就能解决，而需要一组输出容错</b>："})
blocks.append({"table": {
    "header": ["#", "真机现象", "原代码行为", "补丁做法"],
    "rows": [
        ["1", "模型输出 value 键而非 choice/score 键",
         "ProviderError: choice is missing from probabilities",
         "提示词明确要求原生键名（治本）"],
        ["2", "概率和不严格为 1（如 0.4 + 0.3 + 0.2 + 0.1 ≠ 1）",
         "校验器只容忍 0.01 偏差 → ProviderError: "
         "choice probabilities do not sum to one，且 retryable=False",
         "交给校验器前温和归一化"],
        ["3", "score 越界（4 档被模型理解成 1~4，给出 4）",
         "ProviderError: score is out of range",
         "限幅到 [0, 选项数−1]；另支持把选项名映射为下标"],
        ["4", "多问嵌套 JSON 被截断（ollama 默认只生成 128 token）",
         "解析失败，形状校验不通过",
         "payload 显式传 max_tokens（默认 1024）"],
    ],
    "widths": [0.22, 1.4, 1.9, 1.4],
}})
blocks.append({"note": "这四类错误原代码<b>全部标为 retryable=False</b>，意味着任何一处模型输出瑕疵"
                       "都会让整单请求硬失败，而不会降级到 rules provider。"
                       "建议：格式类错误标为 retryable=True，让 ResilientProvider 能兜底。",
               "level": "warn"})

blocks.append({"h2": "3　缺陷三：格式修好后，仍然走不到自动化"})
blocks.append({"p": "这是更深一层的问题，和格式无关。OpenSourceProvider 的默认 profile 是 "
                    "<code>calibrated=False</code>，于是所有预测的 calibration_status 恒为 "
                    "<code>uncalibrated</code>；而 PolicyConfig 默认 "
                    "<code>require_calibration_for_auto=True</code>，会因此触发 "
                    "<code>uncalibrated_probability</code> 并降级为 ask_clarification。"})
blocks.append({"p": "<b>结论：不注入标定过的 DecisionHeadProfile，opensource 路径永远无法自动处置，"
                    "只能反复追问。</b>这是设计上的安全默认，但必须显式说明，否则接入方会困惑。"})
blocks.append({"table": {
    "header": ["情形", "calibration_status", "policy 动作", "是否被拦"],
    "rows": [
        ["默认 profile（calibrated=False）", "uncalibrated",
         "human_review（因 risk_level 偏高）", "是"],
        ["注入哨兵 D2 已标定 profile（T=1.14，π_train=0.0312）",
         "temperature_scaled",
         "ask_clarification（原因变为需要澄清，不再是「未校准」）", "否"],
    ],
    "widths": [2.0, 1.1, 1.7, 0.6],
}})
blocks.append({"note": "实测证据：注入标定 profile 后，noul 概率被正确校正为 "
                       "yes=0.6777 / no=0.3223，calibration_status 变为 temperature_scaled，"
                       "<b>policy 不再因未校准而拦截</b>。哨兵 D2 已经产出了这组标定参数"
                       "（温度 T 与训练先验 π），可以直接提供给 opensource provider 使用。",
               "level": "good"})

blocks.append({"h2": "4　真机证据（节点 gx10-902c，ollama + Qwen3-8B）"})
blocks.append({"table": {
    "header": ["被测版本", "第 1 次", "第 2 次", "第 3 次", "成功率"],
    "rows": [
        ["0.1.3 原版", "失败", "失败", "失败", "0 / 3"],
        ["0.1.3 + 补丁", "通过 13.3 s", "通过 13.2 s", "通过 13.2 s", "3 / 3"],
    ],
    "num_cols": [1, 2, 3, 4],
    "bold_rows": [1],
    "widths": [1.4, 1.0, 1.0, 1.0, 0.8],
}})
blocks.append({"h3": "原版典型报错（3 次一致）"})
blocks.append({"table": {
    "header": ["调用", "报错"],
    "rows": [["provider.evaluate(state)",
              "ProviderError: invalid answer for intent: expected object（retryable=False）"]],
    "widths": [1.5, 3.3],
}})
blocks.append({"h3": "补丁版输出（一次调用并行产出 6 条类型化决策）"})
blocks.append({"table": {
    "header": ["决策 id", "类型", "取值", "主要概率"],
    "rows": [
        ["intent", "choice", "clarification", "clarification 0.70 / information_request 0.20"],
        ["risk_level", "score", "0.85", "high 0.40 / moderate 0.30"],
        ["needs_evidence", "noul", "True", "yes 0.80 / no 0.20"],
        ["needs_clarification", "noul", "True", "yes 0.90 / no 0.10"],
        ["allows_side_effect", "noul", "False", "no 1.00 / yes 0.00"],
        ["next_action", "choice", "ask_clarification", "ask_clarification 0.60 / retrieve_evidence 0.20"],
    ],
    "widths": [1.3, 0.7, 1.15, 2.0],
}})

blocks.append({"h2": "5　补丁清单与用法"})
blocks.append({"table": {
    "header": ["#", "修改点", "作用"],
    "rows": [
        ["1", "OpenSourceProvider 的 system prompt",
         "明确要求原生键名（choice / score / noul），给出三种形状与示例，显式禁止 value 键"],
        ["2", "payload 增加 max_tokens",
         "默认 1024，避免 ollama 默认 128 token 截断多问 JSON"],
        ["3", "dataclass 增加 max_tokens: int = 1024",
         "不同后端可调参"],
        ["4", "新增 _normalize_raw_answers()",
         "兜住 value 键名 / 概率和不为 1 / score 越界 / 概率非数值四类瑕疵"],
        ["5", "在 _validate_answer_set 之前调用清洗函数",
         "只做容错，不改校验器的合法性判断"],
    ],
    "widths": [0.22, 1.9, 2.7],
}})
blocks.append({"p": "<b>用法</b>：<code>python3 patch_opensource_provider.py "
                    "/path/to/src/jev_service/provider.py</code>，幂等可重复执行；"
                    "加 <code>--check</code> 只检测是否已修补。补丁后 29 个单元测试全绿。"})

blocks.append({"h2": "6　顺带一个对作品有用的论据"})
blocks.append({"p": "这次真机测试还顺手提供了一个很有力的对比：<b>通用大模型的输出是不可控的</b>。"
                    "同一个模型、同一个提示词，连续三次调用暴露了四类不同的格式瑕疵"
                    "（键名错、概率和不为一、score 越界、token 截断），需要一整套容错代码才能接住。"})
blocks.append({"table": {
    "header": ["输出可控性", "垂类决策头（哨兵）", "通用大模型（opensource 路径）"],
    "rows": [
        ["键名与结构", "类型化输出，必然合法", "自由文本，需解析与容错"],
        ["概率约束", "softmax 保证和为 1", "模型自由给出，常不为 1"],
        ["取值范围", "由输出层维度保证", "可能越界（如 4 档给出 4）"],
        ["长度限制", "不受影响", "受 max_tokens 约束，可能截断"],
        ["接入代价", "直接可用", "需 4 类容错 + 重试 + 降级"],
    ],
    "widths": [1.2, 1.8, 1.9],
}})
blocks.append({"note": "这一条可以直接写进作品的「架构选型」论证："
                       "高频、结构化、要求确定性的判断交给决策头（输出可控、亚毫秒、已校准）；"
                       "只有罕见、模糊、需要语言理解的任务才升级到生成式——"
                       "而升级路径必须配备输出容错与降级，否则线上会被模型的格式瑕疵击穿。",
               "level": "info"})

p = build_pdf(
    OUT,
    "jev-decision-service 0.1.3 · opensource 路径缺陷与修复",
    "技术反馈单　|　真机环境：NVIDIA DGX Spark（GB10）+ ollama + Qwen3-8B　|　2026-09-24",
    blocks,
    footer="补丁：patch_opensource_provider.py　|　真机脚本：jev_real_test.py / calib_check.py　|　"
           "全部结论均来自 gx10-902c 节点实测",
)
print("已生成:", p)
