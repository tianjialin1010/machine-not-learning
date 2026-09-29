#!/usr/bin/env python3
"""生成《jev 0.1.4 → 0.1.5 升级与效果对比报告》PDF。"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mkpdf import build_pdf                                    # noqa: E402

OUT = "/Users/jialin_tian/WorkBuddy/2026-09-20-02-38-44/jev 0.1.5 升级与效果对比报告.pdf"

B: list[dict] = []

# ------------------------------------------------------------------ 1
B.append({"h2": "1　结论摘要"})
B.append({"p": "0.1.5 是一次<b>结构性升级</b>（代码 +47%，测试 35 → 70 项），"
               "方向完全正确：它把我上一轮报的三个框架级问题<b>全部修掉</b>，"
               "并新增了工业场景真正需要的四类能力。端到端主指标与 0.1.4 持平，"
               "代价是单条延迟上升约 21%。"})
B.append({"kpi": [("+47%", "代码量（2114 → 3107 行）"),
                  ("70", "单元测试（0.1.4 为 35）"),
                  ("3 / 3", "上轮问题全部修复"),
                  ("88.3%", "自动化率（与 0.1.4 持平）")]})
B.append({"table": {
    "header": ["维度", "0.1.4", "0.1.5", "判断"],
    "rows": [
        ["代码量 / 测试", "2114 行 / 35 项", "3107 行 / 70 项", "结构升级"],
        ["模块拆分", "provider 单体 550 行",
         "拆出 routing / normalization / http_provider / errors", "可维护性↑"],
        ["上轮 3 个问题", "全部存在", "全部修复", "见第 3 节"],
        ["StepFun 官方接入", "无", "StepFunProvider + json_schema", "新能力"],
        ["问题集注入", "无（代码内）", "JEV_QUESTION_PROFILE + roles", "新能力"],
        ["特征契约", "无", "维度/名称/量程/单位四道校验", "新能力"],
        ["离线评测工具", "无", "evaluation.py（分组指标 + Brier + ECE）", "新能力"],
        ["自动化率（主口径）", "88.3%", "88.3%", "持平"],
        ["单条延迟 p50", "0.444 ms", "0.536 ms", "慢 21%（见 4.3）"],
    ],
    "widths": [1.1, 1.1, 1.3, 0.9],
}})

# ------------------------------------------------------------------ 2
B.append({"h2": "2　0.1.5 新增了什么"})
B.append({"table": {
    "header": ["新增能力", "对工业场景的意义"],
    "rows": [
        ["QuestionProfile.roles（risk/action/evidence/clarification/side_effect）",
         "领域问题可以叫 intervention_risk，不必再迁就 policy 的硬编码 id——"
         "我之前靠「故意沿用框架 id」绕过的那个约束，现在从设计上解掉了"],
        ["DecisionHeadProfile.temperatures（逐问题温度）",
         "多决策头可各自标定，不必再把温度强行挤进一个字段"],
        ["prior_correction_head + prior_shift_assumption",
         "先验修正只作用于指定 Noul，并要求显式声明 label_shift 假设（修了上轮问题 2）"],
        ["feature_count / names / bounds / units",
         "结构化头有了输入契约：维度、名称、量程、单位四道校验，"
         "不匹配时 accepts() 返回 False 交给文本模型，而不是报错"],
        ["provider_disagreement → 强制人工复核",
         "快头与生成式判断不一致时自动进复核，不再凭其中一个下结论"],
        ["raw_value / raw_probabilities 保留",
         "原始风险与校准后风险可追溯；原始高风险不会因温度缩放或升级而消失"],
        ["provider_calls 调用审计（脱敏）",
         "记录模型、尝试次数、耗时、结果与错误码，不含认证头与输入原文"],
        ["StepFunProvider（json_schema strict + 有限退避）",
         "云端升级层就绪；无密钥时不静默降级成规则，直接明确失败"],
        ["TieredProvider.accepts() 路由 + 分歧/降级遥测",
         "升级失败不再被伪装成 fast_head_accepted（修了上轮问题 1）"],
    ],
    "widths": [1.9, 2.1],
}})

# ------------------------------------------------------------------ 3
B.append({"h2": "3　上轮所报 3 个问题的修复验证"})
B.append({"h3": "3.1　降级遥测被静默掩盖 → 已修复"})
B.append({"table": {
    "header": ["版本", "触发方式", "实际观测"],
    "rows": [
        ["0.1.4", "升级层失败（27B 返回空内容）",
         "route=fast，route_reason=fast_head_accepted，耗时 89 秒——"
         "完全看不出发生过升级，也看不出失败"],
        ["0.1.5（不可重试）", "升级层返回 404",
         "明确报错：{\"error\":\"local-generative returned HTTP 404\", "
         "\"code\":\"http_error\",\"provider_status\":404,\"retryable\":false}——"
         "配置错误直接暴露，不伪装成成功"],
        ["0.1.5（可重试）", "升级层连接被拒",
         "route_reason=rare_class_support_insufficient;generative_failed:transport_error，"
         "warnings 含 provider_fallback:transport_error——降级原因可追溯"],
    ],
    "widths": [0.9, 0.9, 2.2],
}})
B.append({"h3": "3.2　先验修正误伤全部 Noul → 已修复"})
B.append({"table": {
    "header": ["版本", "deployment_positive_rate=0.5 时的四个 Noul 概率", "判断"],
    "rows": [
        ["0.1.4", "fault 0.0027→0.0774（预期）；needs_evidence 0.052→0.631、"
                  "needs_clarification 0.050→0.620、allows_side_effect 0.900→0.996（全部误伤）",
         "追问阈值 0.70 被逼近，随时误触发"],
        ["0.1.5", "fault 0.0016→0.0465（预期）；其余三项 0.0521 / 0.0500 / 0.9000 "
                  "<b>完全不变</b>",
         "作用域正确"],
    ],
    "widths": [0.6, 2.4, 1.0],
}})
B.append({"h3": "3.3　Score 概率键名不校验 → 已修复（且比建议更严谨）"})
B.append({"p": "0.1.5 的处理分三层：① 命名等级（观察/计划/尽快/立即）自动转下标并留"
               "`score_labels_to_indices` 警告；② <b>标定后的 Score 强制使用规范数字键</b>，"
               "否则直接抛错；③ choice 缺失的合法标签补零并告警，<b>额外标签或类型错配直接拒绝</b>。"
               "我们的 evaluator 现在直接输出 \"0\"~\"3\"，实测 normalization_warnings 为空。"})

# ------------------------------------------------------------------ 4
B.append({"h2": "4　端到端效果对比（同数据、同决策头、同种子）"})
B.append({"p": "两版使用<b>同一套决策头权重</b>（heads.pt 复制沿用）、同一批 AI4I 留出集样本"
               "（n=1000，seed=7）、同一条完整框架链路。"
               "0.1.4 侧保留我原来的集成写法，0.1.5 侧改用新能力（roles / 逐头温度 / "
               "先验作用域 / 特征契约）——即「各版各自的最佳写法」。"})
B.append({"h3": "4.1　主指标"})
B.append({"table": {
    "header": ["指标", "0.1.4", "0.1.5", "变化"],
    "rows": [
        ["自动化率", "88.3%", "88.3%", "持平"],
        ["自动部分准确率", "100.0%", "100.0%", "持平"],
        ["故障漏放率", "0.0%", "0.0%", "持平"],
        ["正常设备误转率", "9.1%", "9.1%", "持平"],
        ["未标定响应数", "—（该字段尚未存在）", "0", "全部经标定"],
        ["引擎延迟 p50 / p95", "0.444 / 0.474 ms", "0.536 / 0.564 ms", "慢 21%"],
        ["吞吐", "2,020 条/秒", "1,848 条/秒", "−8.5%"],
    ],
    "widths": [1.4, 1.2, 1.2, 1.0],
    "bold_rows": [0, 2],
}})
B.append({"h3": "4.2　阈值扫描：安全拐点位置一致"})
B.append({"table": {
    "header": ["urgent_score", "0.1.4 自动化率", "0.1.5 自动化率", "0.1.4 漏放", "0.1.5 漏放"],
    "rows": [
        ["1.6", "39.8%", "39.7%", "0.0%", "0.0%"],
        ["2.0", "47.9%", "42.3%", "0.0%", "0.0%"],
        ["2.5（默认）", "88.3%", "88.3%", "0.0%", "0.0%"],
        ["3.0", "92.7%", "90.5%", "0.0%", "0.0%"],
        ["3.05", "94.1%", "94.1%", "10.0%", "10.0%"],
    ],
    "num_cols": [1, 2, 3, 4],
    "bold_rows": [2, 4],
    "widths": [1.0, 1.1, 1.1, 0.8, 0.8],
}})
B.append({"note": "拐点位置两版完全一致（都在 3.0↔3.05 之间的悬崖）。"
                  "中间两档 0.1.5 略保守（42.3% vs 47.9%、90.5% vs 92.7%），"
                  "原因是 0.1.5 的风险门控<b>同时比较校准后值与原始值</b>"
                  "（`value` 与 `raw_value` 都在判据里），触发面更大——"
                  "属于「原始高风险不因校准而消失」的预期代价，方向正确。",
               "level": "info"})
B.append({"h3": "4.3　延迟为何上升 21%"})
B.append({"p": "0.1.5 在每次决策里新增了这些固定开销："
               "① 对完整请求（含特征、证据、元数据）做 SHA-256 幂等指纹；"
               "② 每个预测保留 raw_value / raw_probabilities 副本；"
               "③ warnings 去重与 route_reason 拼接；"
               "④ provider_calls 审计快照的深拷贝。"
               "绝对值仍是亚毫秒（0.536 ms/条），对本场景无实质影响；"
               "但若追求极限吞吐，①④ 是可优化项。"})

# ------------------------------------------------------------------ 5
B.append({"h2": "5　0.1.5 新能力的实测记录"})
B.append({"table": {
    "header": ["能力", "验证方式", "实测结果"],
    "rows": [
        ["roles 映射", "问题命名为 intervention_risk / next_action 等，"
                       "PolicyConfig 只传 roles", "策略层零改动，五条保护全部生效"],
        ["逐头温度", "temperatures={fault:0.916, fault_type:0.939, intervention_risk:0.320}",
         "evaluator 只吐原始概率，框架完成标定；fault 0.99299 → 0.995538"],
        ["先验作用域", "prior_correction_head + label_shift + 部署先验 0.5",
         "仅 fault 移动，其余 Noul 不变（见 3.2）"],
        ["特征契约", "缺 feature_units / 维度不符 / 超量程",
         "三种非法输入均→ route=abstain，动作 abstain，不报 500"],
        ["跨模型分歧", "floor=0.995 强制升级到本地 8B（22 秒）",
         "生成式判 intervention_risk=2.0 vs 快头 3.0 → provider_disagreement → "
         "强制 human_review"],
        ["调用审计", "同上", "provider_calls=1，含模型、尝试次数、耗时、结果、错误码"],
        ["原始风险可追溯", "对比 raw_value 与 value",
         "raw=3.0 / value=3.0 等；升级后仍保留快头预测（last_safety_predictions）"],
        ["StepFun 接入", "—", "<b>未验证</b>：无 STEPFUN_API_KEY。"
                              "0.1.5 自己的验证记录同样标注 NOT_RUN"],
    ],
    "widths": [0.9, 1.7, 1.9],
}})

# ------------------------------------------------------------------ 6
B.append({"h2": "6　本次仍发现的 3 个残留问题（都不阻断）"})
B.append({"table": {
    "header": ["#", "问题", "影响", "建议"],
    "rows": [
        ["1", "degraded 布尔未透传：TieredProvider 已置 degraded=True，"
              "但外层 ResilientProvider.degraded 仍为 False，engine.degraded 取的是后者",
         "降级只能靠 route_reason / warnings 发现，"
         "单看 degraded 字段会误判为健康",
         "ResilientProvider._collect 里同时带上 degraded"],
        ["2", "accepts() 吞掉了具体原因：StructuredHeadProvider.accepts() 内部捕获 "
              "ProviderError 后只返回 False，具体是维度/单位/量程哪一项不匹配丢失",
         "调用方只看到 provider_unavailable，难以自查",
         "accepts() 记录 last_reject_reason（如 invalid_features:unit_mismatch）并透出"],
        ["3", "OpenSourceProvider 默认 max_tokens=1024：对 7 问嵌套 JSON 会截断，"
              "finish_reason=length → incomplete_output（可重试）",
         "自定义多问 profile 接本地模型时首次必失败",
         "按问题数估算默认值，或在文档中提示；我已把服务默认值设为 2048"],
    ],
    "widths": [0.25, 1.9, 1.3, 1.1],
}})

# ------------------------------------------------------------------ 7
B.append({"h2": "7　关于 0.1.5 对我方多数据集报告的核对：我的回应"})
B.append({"p": "0.1.5 附带的 docs/industrial-report-review.md 对我 9/28 的"
               "《Jev架构多工业数据集能力评测报告》提了 7 条方法论质疑。"
               "逐条核完：<b>其中 5 条成立，我认可并会修正</b>；2 条需补充数据后确认。"})
B.append({"table": {
    "header": ["#", "对方质疑", "我的判断"],
    "rows": [
        ["1", "「同一模型两种分布」但 BAcc 变化很大；单纯改类别比例不会改变 BAcc",
         "<b>成立，是我的表述问题</b>。均衡口径是<b>重训</b>（训练集下采样后重训），"
         "不是同一模型换评测集。原报告措辞会造成误读，需改为「重训后的均衡口径」"],
        ["2", "高维表现差不能直接归因网络容量不足，需学习曲线与容量对照",
         "<b>成立</b>。原报告写「高维必须扩容」属越界归因。"
         "应改为「64 维隐层在 474 维输入下欠拟合，容量对照实验待补」"],
        ["3", "自然分布 ECE 很低不等于可安全直接阈值路由",
         "<b>成立</b>。应同时给出高风险召回、Brier 与自动放行的错误率——"
         "本次 0.1.5 验收里我已补上「自动部分准确率」「故障漏放率」"],
        ["4", "先验修正只在 label_shift 假设下成立；且 SKAB 均衡 ECE 反而更低，"
              "不能说所有数据集都恶化",
         "<b>成立，且我原文有事实错误</b>。SKAB 均衡 ECE=0.0160 是所有数据集里最低的，"
         "我原文「5 个数据集一致复现 ECE 普遍恶化」不成立，必须更正为"
         "「除 SKAB 外」并说明 SKAB 的分布形态差异"],
        ["5", "SKAB / TEP 是时序数据，随机行划分可能泄漏（应按时间/设备/批次分组）",
         "<b>成立，这是最严重的一条</b>。我的 multi_dataset_bench 用了随机行划分，"
         "SKAB 与 TEP 的结果存在泄漏风险，需改为按时序分组重跑后才可用作成绩"],
        ["6", "0.0006–0.0072 ms/条可能是批量摊销口径，不能与生成模型端到端时延比较",
         "<b>成立</b>。应在报告中明确「推理核心单条口径 vs 端到端口径」，"
         "并避免跨口径大倍数对比。本次 0.1.5 验收已统一为完整链路口径"],
        ["7", "五任务跑通只支持「接口可适配」的工程结论，不能把效果归因于架构",
         "<b>成立</b>。应把结论限定为工程适配性；架构有效性需消融实验"],
    ],
    "widths": [0.25, 1.75, 2.0],
}})
B.append({"note": "这份核对质量很高——尤其第 4、5 条指出了我原报告的<b>事实错误与泄漏风险</b>。"
                  "建议尽快出 v2 修正版：修正 SKAB 表述、明确「重训」口径、"
                  "把 SKAB/TEP 改为按时序分组划分重跑、补容量与架构消融。"
                  "在评审面前主动更正，比被追问强得多。", "level": "bad"})

# ------------------------------------------------------------------ 8
B.append({"h2": "8　部署现状"})
B.append({"table": {
    "header": ["项", "状态"],
    "rows": [
        ["生产服务", "~/sentinel-stack　→ 框架 <b>jev 0.1.5</b>，端口 9000（公网 :9014）"],
        ["基线备份", "~/sentinel-stack-014　→ 框架 0.1.4，需要对照时可 PORT=9001 启动"],
        ["升级层", "默认关闭；<code>ESCALATE=1</code> 开启，"
                   "<code>--generative stepfun</code> 可切换 StepFun（需 STEPFUN_API_KEY）"],
        ["特征契约", "请求必须带 <code>conversation_state.feature_units</code>"
                     "（C / C / rpm / Nm / min）"],
        ["看板", "http://CHANGE_ME_NODE_HOST:9014/（已适配 0.1.5 契约，"
                 "并显示原始风险 vs 校准后风险）"],
        ["管理", "cd ~/sentinel-stack && PORT=9000 ./stackctl.sh start|stop|restart|status|logs|meta"],
    ],
    "widths": [0.9, 3.1],
}})

p = build_pdf(
    OUT,
    "jev 0.1.4 → 0.1.5 升级与效果对比报告",
    "框架升级验证 · 上轮问题修复确认 · 端到端效果对比 · 新能力实测 · 残留问题",
    B,
    footer="哨兵 EdgeSentinel · 框架升级验收 · 数据均来自节点 gx10-902c 实测",
)
print(f"已生成：{p}")
