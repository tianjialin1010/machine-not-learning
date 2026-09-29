#!/usr/bin/env python3
"""生成《通用大模型 vs 垂类决策头 · 对决实验报告》PDF（数据驱动）。"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, "/Users/jialin_tian/WorkBuddy/2026-09-20-02-38-44")
from mkpdf import build_pdf  # noqa: E402

SRC = "/Users/jialin_tian/WorkBuddy/2026-09-20-02-38-44/duel_result.json"
OUT = "/Users/jialin_tian/WorkBuddy/2026-09-20-02-38-44/通用大模型vs垂类决策头·对决报告.pdf"

with open(SRC, encoding="utf-8") as fh:
    R = json.load(fh)

S = R["summary"]
N = R["n_eval"]
N_FAIL = R["n_fail"]


def g(key, field, default=None):
    return S.get(key, {}).get(field, default)


def f(v, nd=3, dash="—"):
    if v is None:
        return dash
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


# ---------------------------------------------------------------- 选手排序
HEAD_KEY = "哨兵决策头 (MLP+校准)"
llm_keys = [k for k in S if k != HEAD_KEY
            and k not in ("随机基线", "全判正常（多数类）")]
order_pref = []
for tag in ["Qwen3-27B", "Qwen3-8B"]:
    for lv in ["L0", "L1", "L2", "L2-CoT"]:
        k = f"{tag}-{lv}"
        if k in S:
            order_pref.append(k)
llm_keys = order_pref + [k for k in llm_keys if k not in order_pref]
baselines = [k for k in ["全判正常（多数类）", "随机基线"] if k in S]
all_keys = [HEAD_KEY] + llm_keys + baselines

NAME_MAP = {
    HEAD_KEY: "哨兵决策头（MLP+校准）",
    "全判正常（多数类）": "基线：全判正常",
    "随机基线": "基线：随机猜",
}


SUFFIX = {"L0": "裸数值", "L1": "+知识", "L2": "+示例", "L2-CoT": "+示例·思考链(超时)"}


def nice(k):
    if k in NAME_MAP:
        return NAME_MAP[k]
    for lv in ["L2-CoT", "L0", "L1", "L2"]:     # 长后缀优先匹配
        if k.endswith("-" + lv):
            return f"{k[:-(len(lv) + 1)]} · {SUFFIX[lv]}"
    return k


# 决策头延迟采用单条调用口径（full_eval 实测 0.097 ms/条）；
# 批量 predict 会摊薄到 0.0005 ms/条，不适合与「并发单条」延迟并列比较。
HEAD_MS = 0.097


blocks = []

# ---------------------------------------------------------------- 实验设计
blocks.append({"h2": "0　实验设计（为什么要这么比）"})
blocks.append({"p": "核心问题：在工业设备状态判定这类「结构化数值 → 类型化决策」任务上，"
                    "通用大模型到底比我们训练的垂类小网络强还是弱？"})
blocks.append({"table": {
    "header": ["要素", "设定"],
    "rows": [
        ["数据", f"AI4I 2020；训练 6000 / 校准 2000 / 测试 2000（种子 20260921）"],
        ["评估集", f"均衡抽样 n={N}（故障 {N_FAIL} / 正常 {N_FAIL}）——避免「全判正常」刷分"],
        ["垂类选手", "9 维状态编码 → MLP(64,32) 三决策头 → 温度缩放校准"],
        ["通用选手", "Qwen3-27B、Qwen3-8B（本地 GGUF，100% GPU 卸载，temperature=0）"],
        ["信息量档位", "L0 裸数值 / L1 加单位与正常范围 / L2 再加故障机理 + few-shot 示例"],
        ["决策方式", "一次调用同时输出三个决策（对标 Jev 并行采样）：故障? 类型? 严重度?"],
        ["统一指标", "Accuracy / Balanced Acc / Precision / Recall / F1 / MCC / ECE / Brier / 格式合规率 / 延迟"],
    ],
    "widths": [1.05, 3.3],
}})
blocks.append({"note": "公平性说明：三方吃<b>完全相同的一批样本</b>。大模型拿的是同一组数字写成的一句话，"
                       "垂类模型拿的是同一行传感器读数。<b>刻意给大模型加了 L1/L2 两档知识注入</b>——"
                       "如果这样它仍然打不过，那结论就不是「提示没写好」，而是任务性质本身决定的。",
               "level": "info"})

# ---------------------------------------------------------------- 总表
blocks.append({"h2": "1　总表：谁更强"})
rows = []
for k in all_keys:
    s = S[k]
    rows.append([
        nice(k),
        f(s["noul_acc"]), f(s["noul_bacc"]), f(s["noul_f1"]), f(s["noul_mcc"]),
        f(s.get("type_macro_f1")), f(s.get("sev_acc")),
        f(s.get("ece"), 4), f(s.get("brier"), 4),
        f"{(s.get('format_ok') or 0) * 100:.0f}%",
        f"{HEAD_MS:.3f}" if k == HEAD_KEY else f(s.get("latency_ms"), 1),
    ])
blocks.append({"table": {
    "header": ["选手", "Acc", "B.Acc", "F1", "MCC", "类型宏F1", "严重度Acc",
               "ECE", "Brier", "合规", "延迟ms"],
    "rows": rows,
    "num_cols": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "bold_rows": [0],
    "widths": [1.35, 0.42, 0.45, 0.42, 0.42, 0.5, 0.55, 0.45, 0.45, 0.38, 0.5],
}})
blocks.append({"note": "读法：<b>B.Acc</b>（平衡准确率）是主指标——它把故障类和正常类一视同仁，"
                       "不会被「96% 都是正常样本」这种不平衡带偏。<b>MCC</b> 越接近 1 越好、0 等于瞎猜。"
                       "<b>ECE/Brier</b> 衡量概率可不可信，越低越好。<b>合规</b> 是输出能被解析成合法决策的比例。",
               "level": "info"})

# ---------------------------------------------------------------- 分任务
blocks.append({"h2": "2　分任务拆解"})
blocks.append({"h3": "2.1　Noul：是否真实故障（精确率 / 召回率）"})
rows = []
for k in all_keys:
    s = S[k]
    rows.append([nice(k), f(s["noul_prec"]), f(s["noul_rec"]), f(s["noul_f1"]),
                 f(s["noul_bacc"])])
blocks.append({"table": {
    "header": ["选手", "Precision", "Recall", "F1", "B.Acc"],
    "rows": rows,
    "num_cols": [1, 2, 3, 4],
    "bold_rows": [0],
    "widths": [1.8, 0.85, 0.85, 0.85, 0.85],
}})

blocks.append({"h3": "2.2　Choice：故障类型（6 类）与 Score：严重度（0~3）"})
rows = []
for k in all_keys:
    s = S[k]
    rows.append([nice(k), f(s.get("type_acc")), f(s.get("type_macro_f1")),
                 f(s.get("sev_acc")), f(s.get("sev_macro_f1")),
                 f(s.get("sev_mae"), 3), f(s.get("sev_within1"))])
blocks.append({"table": {
    "header": ["选手", "类型Acc", "类型宏F1", "严重度Acc", "严重度宏F1", "严重度MAE", "误差≤1级"],
    "rows": rows,
    "num_cols": [1, 2, 3, 4, 5, 6],
    "bold_rows": [0],
    "widths": [1.35, 0.6, 0.65, 0.7, 0.75, 0.7, 0.65],
}})

# ---------------------------------------------------------------- 信息量
blocks.append({"h2": "3　提示工程能不能救回来？"})
rows = []
for tag in ["Qwen3-27B", "Qwen3-8B"]:
    if f"{tag}-L0" not in S:
        continue
    vals = [f(S[f"{tag}-{lv}"]["noul_bacc"]) if f"{tag}-{lv}" in S else "—"
            for lv in ["L0", "L1", "L2"]]
    cot = f(S[f"{tag}-L2-CoT"]["noul_bacc"]) if f"{tag}-L2-CoT" in S else "—"
    rows.append([tag] + vals + [cot, f(S[HEAD_KEY]["noul_bacc"])])
blocks.append({"table": {
    "header": ["模型", "L0 裸数值", "L1 +知识", "L2 +示例", "L2 +思考链", "（垂类决策头）"],
    "rows": rows,
    "num_cols": [1, 2, 3, 4, 5],
    "widths": [1.0, 0.85, 0.8, 0.8, 0.9, 1.15],
}})

# ---------------------------------------------------------------- 校准
blocks.append({"h2": "4　概率可不可信（ECE / Brier）"})
rows = []
for k in all_keys:
    s = S[k]
    rows.append([nice(k), f(s.get("ece"), 4), f(s.get("brier"), 4)])
blocks.append({"table": {
    "header": ["选手", "ECE", "Brier"],
    "rows": rows,
    "num_cols": [1, 2],
    "bold_rows": [0],
    "widths": [2.2, 1.0, 1.0],
}})

# ---------------------------------------------------------------- 延迟
blocks.append({"h2": "5　延迟"})
rows = []
for k in all_keys:
    s = S[k]
    lm = s.get("latency_ms")
    rows.append([nice(k),
                 f"{HEAD_MS:.3f}" if k == HEAD_KEY else (f"{lm:.1f}" if lm else "—"),
                 f"{(s.get('n') or N)}"])
blocks.append({"table": {
    "header": ["选手", "单条延迟 p50 (ms)", "样本数"],
    "rows": rows,
    "num_cols": [1, 2],
    "bold_rows": [0],
    "widths": [2.2, 1.2, 0.8],
}})

# ---------------------------------------------------------------- 结论
head = S[HEAD_KEY]
b27l0 = S.get("Qwen3-27B-L0", {})
b27l2 = S.get("Qwen3-27B-L2", {})
b8l2 = S.get("Qwen3-8B-L2", {})
speedup = (b27l2.get("latency_ms") or 0) / HEAD_MS

blocks.append({"h2": "6　结论：垂类和通用，谁更适合这件事"})
blocks.append({"p": f"在“结构化传感器数值 → 类型化工业决策”这一特定任务上，"
                    f"垂类小网络<b>全面、显著</b>地优于通用大模型，"
                    f"而且差距不是靠堆参数或写提示词能弥补的。"})

blocks.append({"h3": "6.1　五条硬结论"})
blocks.append({"table": {
    "header": ["#", "结论", "数据支撑"],
    "rows": [
        ["1", "裸数值喂给通用大模型，它等于瞎猜",
         f"27B·L0 平衡准确率 {f(b27l0.get('noul_bacc'))}、F1 {f(b27l0.get('noul_f1'))}、"
         f"MCC {f(b27l0.get('noul_mcc'))}（负值 = 还不如随机猜）"],
        ["2", "领域知识和示例确实能救，但有天花板",
         f"27B 从 L0 的 {f(b27l0.get('noul_bacc'))} 提升到 L1/L2 的 {f(b27l2.get('noul_bacc'))}，"
         f"却仍低于决策头的 {f(head.get('noul_bacc'))}；MCC 差 "
         f"{f(head.get('noul_mcc'))} vs {f(b27l2.get('noul_mcc'))}（约 2.4 倍）"],
        ["3", "小模型不是“够用”，是真的不行",
         f"8B 最好档平衡准确率仅 {f(b8l2.get('noul_bacc'))}，"
         f"严重度准确率 {f(b8l2.get('sev_acc'))}；且它把样本<b>几乎全判为故障</b>"
         f"（召回 1.000、精确率 {f(b8l2.get('noul_prec'))}），属于典型“狼来了”"],
        ["4", "概率质量同样拉开差距",
         f"Brier {f(head.get('brier'), 4)} vs 27B·L2 {f(b27l2.get('brier'), 4)}、"
         f"8B·L2 {f(b8l2.get('brier'), 4)}；ECE {f(head.get('ece'), 4)} vs "
         f"{f(b27l2.get('ece'), 4)}" ],
        ["5", "开思考链在工业实时场景直接出局",
         "27B 开思考链 20 条<b>全部超过 90 秒未完成，0 条可解析</b>；"
         "而报警分诊要求的是毫秒级响应"],
    ],
    "widths": [0.25, 1.5, 2.6],
}})

blocks.append({"h3": "6.2　速度：不是一个量级"})
blocks.append({"table": {
    "header": ["选手", "单条延迟", "相对决策头"],
    "rows": [
        ["哨兵决策头", f"{HEAD_MS:.3f} ms", "1×"],
        ["Qwen3-8B（L2）", f"{b8l2.get('latency_ms', 0):.0f} ms",
         f"{b8l2.get('latency_ms', 0) / HEAD_MS:,.0f}×"],
        ["Qwen3-27B（L2）", f"{b27l2.get('latency_ms', 0):.0f} ms",
         f"{speedup:,.0f}×"],
        ["Qwen3-27B（开思考链）", "> 90 000 ms（超时）", "不可用"],
    ],
    "num_cols": [1, 2],
    "bold_rows": [0],
    "widths": [1.5, 1.5, 1.3],
}})
blocks.append({"note": f"同一块 GPU、同一批数据。决策头 {HEAD_MS} ms/条意味着一条产线上千个测点"
                       f"每秒可扫描多轮；27B 单条就要 {b27l2.get('latency_ms', 0) / 1000:.1f} 秒，"
                       f"碰到报警风暴只能排队。", "level": "info"})

blocks.append({"h3": "6.3　但通用大模型不是没用——边界要说清楚"})
blocks.append({"table": {
    "header": ["维度", "垂类决策头", "通用大模型"],
    "rows": [
        ["结构化数值判定", "强（本次实测全面领先）", "弱（裸数值近乎随机）"],
        ["非结构化文本（报修工单、维修记录、手册）", "做不了", "强，唯一可行的选择"],
        ["长尾/罕见故障（训练数据里没有的）", "无法外推", "可凭通识给出合理推断"],
        ["解释与报告生成", "做不到", "强"],
        ["延迟与成本", "亚毫秒、零增量成本", "秒级、且随调用量线性增长"],
        ["输出可控性", "类型化、必然合法、概率已校准", "自由文本，需解析与校验"],
    ],
    "widths": [1.7, 1.5, 1.7],
}})
blocks.append({"note": "这正是三层路由的意义，也是我们必须诚实说明的边界：<b>高频、结构化、要求确定性的判断"
                       "交给 System One（决策头）；罕见、模糊、需要语言理解与生成的才升级到生成式模型。</b>"
                       "把 27B 拿去做每秒上千条的报警分诊，既慢又不准——本次实验就是证据。",
               "level": "good"})

blocks.append({"note": "方法学提醒：随机基线的 ECE = 0.0000（因为它恒定报 0.5，恰好匹配 50/50 的评估集），"
                       "但它的准确率只有 0.457。<b>ECE 低 ≠ 有用</b>，概率质量必须与判别能力一起看。",
               "level": "warn"})

p = build_pdf(
    OUT,
    "通用大模型 vs 垂类决策头 · 对决实验报告",
    f"硬件：NVIDIA DGX Spark（GB10）　数据：AI4I 2020　评估集：n={N} 均衡抽样　日期：2026-09-22",
    blocks,
    footer="复现：cd ~/jev-service &amp;&amp; python3 duel_bench.py　|　全部数据源自 gx10-902c 节点真机实测",
)
print("已生成:", p)
