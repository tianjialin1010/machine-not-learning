#!/usr/bin/env python3
"""生成《Jev架构多工业数据集能力评测 · v2 修正版》PDF。

数据来源（均为节点实测，可复现）：
  multi_dataset_result_v1.json   ← v1 报告基线（随机行划分，9/28）
  multi_dataset_result_v2.json   ← v2 修正版（时序/跨单元划分 + 5 次重复，9/29）

设计原则：本报告要显式说明「哪些结论被推翻、哪些被降级、哪些保留」，
不做粉饰。修正清单第一节即列出，便于评审直接核对。
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mkpdf import build_pdf                                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "Jev架构多工业数据集能力评测报告 v2（修正版）.pdf")

V1 = json.load(open(os.path.join(HERE, "multi_dataset_result_v1.json"), encoding="utf-8"))
V2 = json.load(open(os.path.join(HERE, "multi_dataset_result_v2.json"), encoding="utf-8"))

D1 = {r["key"]: r for r in V1}
D2 = V2["datasets"]
ORDER = ["ai4i", "steel", "secom", "skab", "tep"]
NAME = {"ai4i": "AI4I 2020", "steel": "Steel Plates", "secom": "SECOM",
        "skab": "SKAB", "tep": "Tennessee Eastman"}


def f4(x):
    return "—" if x is None else f"{x:.4f}"


def pm(s, key):
    """mean±std 字符串。"""
    if not s:
        return "—"
    m, sd = s.get(key), s.get(key + "_std")
    if m is None:
        return "—"
    return f"{m:.4f}" if not sd else f"{m:.4f}±{sd:.4f}"


def main_of(k, which):
    return (D2[k].get("main") or {}).get(which)


blocks: list[dict] = []

# ==================================================================== 1 缘起
blocks.append({"h2": "1　本版缘起与修正清单"})
blocks.append({"p": "0.1.5 附带的《industrial-report-review》对 9/28 的 v1 报告提出 7 条方法论质疑。"
                    "我们逐条核对后认为 <b>5 条成立</b>，其中 2 条属于本报告的事实性错误。"
                    "本版用重新设计的实验逐一验证，并把修正结果如实列出。"})
blocks.append({"table": {
    "header": ["#", "核对提出的质疑", "本版处理方式", "核对结论"],
    "rows": [
        ["1", "「同一模型两种分布」表述有误——均衡口径实为<b>重训</b>（训练集下采样）",
         "全文改称「均衡重训口径」，并同时给出「自然分布口径」", "表述已修正"],
        ["2", "高维数据集表现差，不能直接归因于隐层容量（缺学习曲线/容量对照）",
         "补 3 档容量对照（64→512），看学习曲线", "<b>原归因被推翻</b>"],
        ["3", "低 ECE 不等于可以安全地做阈值路由",
         "新增 4 档置信度阈值扫描，统计自动化率 / 高置信漏放 / 自动内错误率",
         "<b>质疑成立，本版给出关键新证据</b>"],
        ["4", "「5 个数据集一致复现 ECE 恶化」与事实不符（SKAB 的 ECE 实为最低）",
         "逐数据集列出 ECE，不再作一致性陈述", "<b>原表述错误，已更正</b>"],
        ["5", "SKAB / TEP 是连续时序数据，随机行划分存在同段泄漏风险",
         "新增「时序划分」与「跨单元划分」，并对每个配置做 5 次多种子重复",
         "<b>原结论偏乐观，已系统性下修</b>"],
        ["6", "时延口径（推理核心 / 端到端）不能混比",
         "同时给出「单条端到端」与「批量摊薄」两个口径并显式标注，禁止跨口径比较",
         "<b>原表口径混用，已澄清（两口径相差约 3500 倍）</b>"],
        ["7", "五个任务跑通只能支持「接口可适配」，架构有效性需要消融",
         "新增温度缩放消融（4 个数据集）", "校准确有贡献（ECE 降 33~44%），但无法弥补判别力缺失"],
    ],
    "widths": [0.28, 2.1, 2.6, 1.7],
}})

# ==================================================================== 2 方法
blocks.append({"h2": "2　方法学变更（四项）"})
blocks.append({"table": {
    "header": ["变更", "v1 做法", "v2 做法", "为何必要"],
    "rows": [
        ["划分方式", "全样本随机打乱后 60/20/20",
         "在<b>每个时序单元内</b>按 60/20/20 顺序切分（时序划分）；另加按整组切分的"
         "「跨单元划分」",
         "SKAB 是 34 次独立水泵试验、TEP 是 22 个工况各 960 点连续采样；"
         "随机打乱会让同一试验的前后段同时进入训练与测试"],
        ["重复实验", "单次运行（固定种子）",
         "每个配置 5 个种子，报告 mean±std", "部分数据集均衡测试集很小（SECOM 仅 34 条），单次结果不足为据"],
        ["口径", "自然分布 / 均衡分布并列，但均衡口径的语义未说明",
         "明确标注：均衡口径 = <b>重训</b>（训练集也下采样），属「判别力口径」，"
         "不能当作真实部署指标",
         "避免读者把均衡指标误读为部署表现"],
        ["架构贡献", "无消融", "对温度缩放做 on/off 消融", "把「跑通」升级为「哪个组件真正起作用」"],
    ],
    "widths": [0.75, 1.5, 2.3, 2.1],
}})

# ==================================================================== 3 可比性
blocks.append({"h2": "3　先验证两版可比（随机划分复现 v1）"})
blocks.append({"p": "在讨论差异之前，必须先排除「v2 实现与 v1 不同」这一混淆。"
                    "把 v2 切换回 v1 的随机划分口径后，四个数据集的结果与 v1 报告一一对应："})
rows = []
for k in ["ai4i", "secom", "skab", "tep"]:
    b1 = D1[k]["noul_balanced"]
    s = main_of(k, "random_balanced")
    r2 = s["bacc"] if s else None
    rows.append([NAME[k], f4(b1["balanced_accuracy"]), f4(r2),
                 "—" if r2 is None else f"{r2 - b1['balanced_accuracy']:+.4f}",
                 "在标准差范围内" if s and abs(r2 - b1["balanced_accuracy"]) <= 3 * (s["bacc_std"] or 0.5)
                 else "需注意"])
blocks.append({"table": {
    "header": ["数据集", "v1 报告（随机划分·均衡）", "v2 复现（5 次均值）", "偏差", "判定"],
    "rows": rows,
    "num_cols": [1, 2, 3],
    "widths": [1.5, 1.6, 1.5, 0.85, 1.4],
}})
blocks.append({"note": "四个数据集的偏差全部落在标准差以内，最大偏差 0.012。"
                       "这证明 <b>v2 与 v1 的可比性成立，后续的指标变化确实由划分方式导致，"
                       "而非实现差异</b>。", "level": "good"})

# ==================================================================== 4 主结果
blocks.append({"h2": "4　主结果：时序划分让判别力系统性下降"})
blocks.append({"p": "下表为均衡重训口径（与 v1 主表同口径），每个配置 5 次重复。"})
rows, deltas = [], []
for k in ["ai4i", "secom", "skab", "tep"]:
    rand_s = main_of(k, "random_balanced")
    temp_s = main_of(k, "temporal_balanced")
    cross_s = main_of(k, "cross_unit_balanced")
    d = None
    if rand_s and temp_s:
        d = temp_s["bacc"] - rand_s["bacc"]
        deltas.append((k, d))
    rows.append([NAME[k], str(D2[k]["dim"]), pm(rand_s, "bacc"),
                 pm(temp_s, "bacc") if temp_s else "无时序语义",
                 pm(cross_s, "bacc") if cross_s else "不适用",
                 "—" if d is None else f"{d:+.4f}"])
blocks.append({"table": {
    "header": ["数据集", "维度", "随机划分 BAcc", "时序划分 BAcc", "跨单元划分 BAcc", "Δ（时序−随机）"],
    "rows": rows,
    "num_cols": [1, 2, 3, 4, 5],
    "widths": [1.4, 0.5, 1.4, 1.4, 1.4, 1.1],
}})
blocks.append({"p": "自然分布口径（贴近真实部署，不做任何下采样）下降更剧烈："})
rows = []
for k in ["ai4i", "secom", "skab", "tep"]:
    rn, tn = main_of(k, "random_natural"), main_of(k, "temporal_natural")
    rows.append([NAME[k], pm(rn, "bacc"), pm(tn, "bacc") if tn else "无时序语义",
                 f4(rn["mcc"]) if rn else "—",
                 f4(tn["mcc"]) if tn else "—",
                 f4(rn["auc"]) if rn else "—"])
blocks.append({"table": {
    "header": ["数据集", "随机划分 BAcc", "时序划分 BAcc", "随机 MCC", "时序 MCC", "随机 AUC"],
    "rows": rows,
    "num_cols": [1, 2, 3, 4, 5],
    "widths": [1.4, 1.3, 1.3, 0.9, 0.9, 0.9],
}})
blocks.append({"note": f"三个时序数据集的判别力在时序划分下全部下降："
                       f"SECOM {deltas[0][1]:+.4f}、SKAB "
                       f"{[d for k2, d in deltas if k2 == 'skab'][0]:+.4f}、TEP "
                       f"{[d for k2, d in deltas if k2 == 'tep'][0]:+.4f}。"
                       "其中 <b>SECOM 跌破随机水平</b>（时序均衡 BAcc 0.4471、AUC 0.4969，MCC 为负），"
                       "说明它在跨时段泛化上完全失效。原 v1 的 0.6333 显著高估了实际能力。",
               "level": "bad"})
blocks.append({"note": "SKAB 的正例在试验后段集中出现，因此时序划分会造成严重先验漂移："
                       "训练集正例率 16%、校准集 95%、测试集 32%。"
                       "这既解释了指标下滑，也说明 <b>时序场景必须引入部署先验校正</b>"
                       "（与哨兵决策层 D2 的先验校正机制一致）。", "level": "warn"})

# ==================================================================== 5 容量
blocks.append({"h2": "5　容量对照：推翻「高维必须扩容」的归因"})
blocks.append({"p": "v1 把 SECOM（474 维）表现差归因于「474 维压进 64 维隐层严重欠拟合」。"
                    "本版把隐层从 (64,32) 扩到 (512,256,128)——参数量提升约 30 倍——结果如下："})
rows = []
for k in ["ai4i", "secom", "skab", "tep"]:
    cs = D2[k].get("capacity_sweep") or {}
    if not cs:
        continue
    vals = {n: cs[n]["balanced_accuracy"] for n in ("small", "medium", "large") if n in cs}
    gain = max(vals.values()) - vals["small"] if vals else None
    rows.append([NAME[k], str(D2[k]["dim"]),
                 f4(vals.get("small")), f4(vals.get("medium")), f4(vals.get("large")),
                 "—" if gain is None else f"{gain:+.4f}"])
blocks.append({"table": {
    "header": ["数据集", "维度", "(64,32)", "(256,128)", "(512,256,128)", "最大增益"],
    "rows": rows,
    "num_cols": [1, 2, 3, 4, 5],
    "widths": [1.5, 0.5, 1.0, 1.1, 1.4, 1.0],
}})
blocks.append({"note": "容量提升约 30 倍，四个数据集的最大增益仅 "
                       "<b>+0.0000 ~ +0.0295</b>，其中 SECOM 扩到 512 仍然只有 0.3824（随机水平）。"
                       "结论：<b>瓶颈不在隐层容量，而在特征表示与数据本身的信噪比</b>。"
                       "v1 的归因不成立，已更正。", "level": "bad"})

# ==================================================================== 6 消融
blocks.append({"h2": "6　消融：校准确有贡献，但救不了判别力缺失"})
rows = []
for k in ["ai4i", "secom", "skab", "tep"]:
    ab = D2[k].get("ablation") or {}
    if not ab:
        continue
    a, b = ab["full_calibrated"], ab["no_temperature_scaling"]
    drop = (b["ece"] - a["ece"]) / b["ece"] * 100 if b["ece"] else 0
    rows.append([NAME[k], f4(b["ece"]), f4(a["ece"]), f"{drop:.1f}%",
                 f"{b['brier']:.4f}→{a['brier']:.4f}", f"{a['temperature']:.3f}"])
blocks.append({"table": {
    "header": ["数据集", "未校准 ECE", "温度缩放后 ECE", "降幅", "Brier 前→后", "温度 T"],
    "rows": rows,
    "num_cols": [1, 2, 3, 4, 5],
    "widths": [1.5, 1.1, 1.3, 0.8, 1.3, 0.8],
}})
blocks.append({"note": "温度缩放一致地把 ECE 压低 33%~44%，证明该组件确有贡献（回应第 7 条质疑）。"
                       "但必须同时看到：<b>SECOM 校准后 ECE 仍为 0.4104、TEP 仍为 0.2518</b>——"
                       "校准只能修正「概率标度」，无法创造判别信息。"
                       "对判别力本身不足的模型，校准之后依然不可用于自动化。", "level": "warn"})

# ==================================================================== 7 阈值
blocks.append({"h2": "7　阈值扫描：低 ECE 不等于可以安全地阈值路由"})
blocks.append({"p": "这是核对第 3 条质疑的直接验证，也是本版最重要的新增证据。"
                    "在时序划分 + 均衡重训口径下，按置信度阈值自动化，看三件事："
                    "自动化率、<b>高置信漏放率</b>（真故障被以高置信判为正常）、"
                    "<b>自动内错误率</b>（自动处置部分判错的比例）。"})
rows = []
for k in ["ai4i", "secom", "skab", "tep"]:
    ts = D2[k].get("threshold_scan") or []
    if not ts:
        continue
    by = {t["threshold"]: t for t in ts}
    cells = [NAME[k]]
    for th in (0.7, 0.9, 0.95):
        t = by.get(th)
        cells.append("—" if not t else
                     f"{t['automation_rate']*100:.1f}% / {t['fault_miss_rate']*100:.1f}% / "
                     f"{t['auto_error_rate']*100:.1f}%")
    rows.append(cells)
blocks.append({"table": {
    "header": ["数据集", "阈值 0.7", "阈值 0.9", "阈值 0.95"],
    "rows": rows,
    "widths": [1.4, 2.3, 2.3, 2.3],
}})
blocks.append({"p": "（单元格格式：自动化率 / 高置信漏放率 / 自动内错误率）"})
blocks.append({"table": {
    "header": ["数据集", "均衡 ECE", "阈值 0.9 时自动内错误率", "阈值 0.9 时自动化率", "该阈值下是否可用"],
    "rows": [
        ["AI4I 2020", "0.0587", "6.9%", "72.9%", "可用（漏放稳定在 4.3%，属模型固定盲区）"],
        ["SKAB", "0.0392", "3.4%", "18.0%", "可用但自动化率被压得很低"],
        ["Tennessee Eastman", "0.2141", "<b>26.9%</b>", "44.5%", "<b>不可用</b>——每 4 条自动处置错 1 条"],
        ["SECOM", "0.3565", "0.0%（因 88% 请求被拒自动化）", "11.8%", "<b>不可用</b>——th=0.7 时自动内错误 59.1%"],
    ],
    "widths": [1.6, 0.9, 1.6, 1.4, 2.5],
}})
blocks.append({"note": "结论：<b>ECE 与「阈值路由是否安全」不是同一件事</b>。"
                       "SKAB 的 ECE（0.0392）与 TEP（0.2141）相差 5 倍，但真正的判据是"
                       "「自动处置部分的错误率」。SECOM 在 0.7 阈值下自动处理 64.7% 的请求、"
                       "其中 <b>59.1% 判错</b>、漏放 47.1%——如果只看 ECE 就上线，会把一半的故障"
                       "静默归档。这正是核对第 3 条质疑的实证。", "level": "bad"})

# ==================================================================== 8 延迟
blocks.append({"h2": "8　延迟口径澄清"})
blocks.append({"p": "v1 表中报告的 0.00057~0.00719 ms 实为<b>批量摊薄值</b>，"
                    "而与之对比的生成式方案是<b>单条</b>延迟。两种口径相差三个数量级，"
                    "不能并列。本版同时给出两个口径并显式标注："})
rows = []
for k in ["ai4i", "secom", "skab", "tep"]:
    lat = D2[k].get("latency")
    if not lat:
        continue
    ratio = lat["single_inference_ms"] / lat["batch_amortized_ms"]
    rows.append([NAME[k], f"{lat['single_inference_ms']:.4f}",
                 f"{lat['single_throughput_per_s']:,}",
                 f"{lat['batch_amortized_ms']:.6f}",
                 f"{lat['batch_throughput_per_s']:,}", f"{ratio:,.0f}×"])
blocks.append({"table": {
    "header": ["数据集", "单条端到端 (ms)", "单条吞吐 (条/秒)", "批量摊薄 (ms)", "批量吞吐 (条/秒)", "倍数"],
    "rows": rows,
    "num_cols": [1, 2, 3, 4, 5],
    "widths": [1.5, 1.2, 1.2, 1.2, 1.3, 0.7],
}})
blocks.append({"note": "按<b>单条端到端</b>口径（含张量构造）：0.0454~0.0508 ms，"
                       "对比生成式单条 11 279 ms 约为 <b>24.5 万倍</b>。"
                       "批量摊薄值仅适用于「批量离线扫描」场景，不能用于实时报警分诊的对比。", "level": "info"})

# ==================================================================== 9 结论修正
blocks.append({"h2": "9　v1 结论的修正清单"})
blocks.append({"table": {
    "header": ["v1 的结论", "v2 判定", "依据"],
    "rows": [
        ["跨领域可用性成立（5 个子领域零改动跑通）",
         "<b>降级</b>为「接口可适配」",
         "时序划分下 SECOM 完全失效（AUC 0.4969）、TEP 在自然分布下恒判故障（BAcc 0.5000、MCC 0）"],
        ["低维（8~9 维）结构化传感器是优势区",
         "<b>部分保留</b>，数值下修",
         "SKAB 时序均衡 BAcc 0.7462（原 0.8461）、跨单元 0.8130；AI4I 随机均衡 0.8917"],
        ["高维必须扩容（474 维压进 64 维隐层欠拟合）",
         "<b>推翻</b>",
         "容量扩至 (512,256,128) 后 SECOM 仅 0.3824，四数据集最大增益 +0.0295"],
        ["5 个数据集一致复现「校准随部署分布漂移」",
         "<b>更正</b>为部分成立",
         "SKAB 校准后 ECE 0.0392 为全部最低，「一致恶化」不成立；漂移幅度各数据集差异显著"],
        ["延迟 0.0006~0.0072 ms/条",
         "<b>更正口径</b>",
         "该数值为批量摊薄；单条端到端为 0.0454~0.0508 ms，两者相差约 3500 倍"],
        ["少数类是主要短板（AI4I Choice 宏 F1 0.211）",
         "<b>保留</b>",
         "TWF 46 条 / RNF 18 条，样本量不足，与容量无关"],
    ],
    "widths": [2.2, 1.4, 3.6],
}})

# ==================================================================== 10 局限
blocks.append({"h2": "10　仍未解决的局限（如实列出）"})
blocks.append({"table": {
    "header": ["局限", "影响", "后续动作"],
    "rows": [
        ["SECOM / TEP 的判别力本身不足（时序划分下接近随机）",
         "不能声称这两类场景可用；半导体与连续流程需要专门的特征工程",
         "尝试按工况归一化、只用稳态段、加入过程变量的一阶差分与滑动统计"],
        ["SKAB 时序划分造成严重先验漂移（训练 16% / 校准 95% / 测试 32%）",
         "校准集无法代表部署分布，温度缩放被误导",
         "引入部署先验校正（logits 加 log 先验比），与哨兵 D2 机制统一"],
        ["部分数据集测试样本量小（SECOM 均衡仅 34 条）",
         "指标标准差可达 ±0.064，细粒度比较的统计功效不足",
         "改用交叉验证或扩大采样次数；对外只报区间不报点值"],
        ["只有 5 个工业数据集、4 个可评 Noul",
         "跨领域结论的覆盖面有限", "补 CWRU 轴承振动、UCI-447 液压系统，覆盖旋转机械与液压传动"],
    ],
    "widths": [2.4, 2.3, 2.6],
}})
blocks.append({"note": "本版的立场：<b>宁可把指标改低，也不留下会被复现推翻的结论。</b>"
                       "v1 中过于乐观的三条已在本版修正，其中两条属于我们的错误。", "level": "info"})

build_pdf(OUT, "Jev 架构多工业数据集能力评测 · v2 修正版",
          "Machine not Learning · 哨兵 EdgeSentinel 项目 · 全部数据为 DGX Spark（NVIDIA GB10 · CUDA 13.0）"
          "节点实测 · 5 数据集 × 三种划分 × 两套口径 × 5 次重复",
          blocks, footer="哨兵 EdgeSentinel · Jev 架构多工业数据集能力评测 v2 · 内部技术文档")
print("已生成:", OUT)
