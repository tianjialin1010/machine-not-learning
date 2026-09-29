# 哨兵 EdgeSentinel · 工业报警智能处置中枢

**Machine not Learning** 队 · 第三届 NVIDIA DGX Spark 黑客松 · Agent Skills 开发挑战赛

> 把「判断」从模型手里拿回来：不用生成式模型写回复，而是让它输出**带校准概率的类型化决策**，
> 再由确定性代码决定自动处置、转人工还是升级。全部推理跑在 DGX Spark 本地，数据不出厂。

---

## 一、核心成果（均为节点实测，可复现）

### 1.1 留出集端到端（AI4I 2020，n=1000，走完整框架链路）

| 指标 | 数值 |
|---|---|
| 自动化率 | **88.3%** |
| 自动处置部分准确率 | **100%** |
| 故障漏放率 | **0%** |
| 正常设备误转人工 | 9.1% |
| 引擎延迟 | p50 **0.536 ms** / p95 0.564 ms |
| 吞吐 | 1,848 条/秒 |

### 1.2 阈值—自动化率—安全权衡（拐点是悬崖）

| urgent_score | 自动化率 | 故障漏放 | 正常误转 |
|---|---|---|---|
| 1.6 | 39.8% | 0% | 59.1% |
| 2.0 | 47.9% | 0% | 50.7% |
| **2.5** | **88.3%** | **0%** | 9.1% |
| 3.0 | 92.7% | 0% | 4.5% |
| 3.05 | 94.1% | **10.0%** | 3.4% |

阈值从 3.0 抬到 3.05 只换来 1.4 个百分点的自动化，漏放率却从 0% 跳到 10%。

### 1.3 垂类决策头 vs 通用大模型（同任务、同数据）

| | 哨兵决策头 | Qwen3-27B（加示例） | Qwen3-27B（裸数值） |
|---|---|---|---|
| 平衡准确率 | **0.829** | 0.643 | 0.493 |
| MCC | **0.692** | 0.286 | **−0.085** |
| 严重度准确率 | **0.986** | 0.829 | 0.079 |
| Brier | **0.1202** | 0.2855 | 0.4491 |
| 单条延迟 | **0.046 ms** | 11,279 ms | 10,594 ms |

另有独立实验：让 27B 自报故障概率，**平均自报置信度 0.912，实际准确率仅 0.475**（ECE 0.4375）——
这是「RLHF 系统性过度自信」的实测证据，也是本项目立论的基础。

### 1.4 跨 5 个工业子领域评测（v2 修正版，时序划分 + 5 次重复）

| 数据集 | 领域 | 维度 | 均衡重训口径 BAcc |
|---|---|---|---|
| AI4I 2020 | 机加工 | 9 | 0.8917 ± 0.0171 |
| SKAB | 水务/水泵 | 8 | 0.7462 ± 0.0047 |
| Tennessee Eastman | 化工流程 | 52 | 0.6385 ± 0.0269 |
| SECOM | 半导体 | 474 | 0.4471 ± 0.0638 |
| Steel Plates | 冶金 | 27 | （仅多分类，宏 F1 0.7498） |

> v2 版修正了 v1 的三处过于乐观的结论（含两处事实错误），并公开了「低 ECE ≠ 可安全阈值路由」
> 的反例。详见 `docs/Jev架构多工业数据集能力评测报告 v2（修正版）.pdf`。

---

## 二、架构：三层路由

```
报警事件（结构化传感器特征）
        │
        ▼
┌───────────────────────────────────────────┐
│ 第一层 · 决策头（DGX Spark GB10 GPU）      │
│   三个并行决策头：                          │
│     Noul  是否真实故障    → 概率            │
│     Choice 故障类型       → 分布            │
│     Score  维护优先级     → 分布 + 期望值   │
│   逐头温度标定 + 一致性绑定                 │
│   延迟 0.046 ms / 条                        │
└───────────────┬───────────────────────────┘
                │ 置信度 = max(p, 1−p)
        ┌───────┴────────┐
        │                │
   置信度达标         置信度不足
        │                │
        ▼                ▼
  确定性策略路由   第二层 · 生成式模型（8B）
  （代码决定，      （罕见/模糊样本的第二意见，
   非模型决定）      需要人工复核）
        │
        ▼
  第三层 · 兜底与弃权（ProviderError → abstain + 降级标记）
```

**关键设计**：模型只负责「判断」，是否自动化处置由**代码**依据校准后的概率与阈值决定。
这不是实现细节，而是让系统可审计、可追责的前提。

三层之上还挂了一套由 `jev-decision-service 0.1.5` 提供的治理能力：
校准与先验修正、分布漂移检测、OOD 拦截、稀有类告警、决策一致性校验、调用审计。

---

## 三、目录结构

```
machine-not-learning/
├── sentinel-stack/              ★ 主服务栈（框架 0.1.5 + 决策头 + 前端 + 发布流程）
│   ├── sentinel_heads.py            模型层：数据→特征→三决策头→标定→GPU 推理
│   ├── sentinel_integrated.py       服务层：工业问题集/evaluator/三层路由/HTTP
│   ├── dashboard.html               实时看板（零外部依赖）
│   ├── stackctl.sh                  服务管理（start/stop/restart/status/logs/meta）
│   ├── frontend_static.py           前端静态资源路由
│   ├── industrial_runtime.py        工业模型运行时
│   ├── vendor/jev_service/          决策治理框架（0.1.5，团队自研）
│   ├── deploy-staging/              发布流水线
│   ├── frontend-releases/           前端发布记录（含 3D 场景模型）
│   ├── industrial-releases/         工业模型发布记录与训练脚本
│   └── deployment-backups/          部署前健康快照
├── sentinel-stack-014/          框架 0.1.4 基线栈（版本对照用）
├── jev-service/                 决策头原型与早期实验
├── jevtest/                     框架 5 个版本副本 + 真机测试脚本
├── _up015/                      0.1.5 栈的解压副本
├── analysis/                    评测脚本与结果数据
│   ├── multi_dataset_bench_v2.py    多数据集评测（时序划分/容量对照/消融/阈值扫描）
│   ├── duel_bench.py                垂类头 vs 通用大模型对决
│   ├── gpu_head.py / prior_calib.py CUDA 决策头 / 先验校正
│   ├── sentinel_jev_bridge.py       决策头接入框架的桥接
│   └── *.json                       全部实验原始结果
├── docs/                        技术文档与报告（12 份 PDF + 方案与征文）
├── tools/                       集群连接工具（已脱敏，凭据需自行填入）
├── spark_bench.sh               节点性能测评
├── extra_bench.sh               内存带宽 / CPU 多核 / 端口映射实测
└── ollama_bench.sh              Ollama 推理吞吐实测
```

---

## 四、快速开始

### 4.1 环境要求

- NVIDIA DGX Spark（GB10，统一内存架构）或任意 CUDA GPU
- Python 3.12、PyTorch 2.14+（CUDA 13.0）
- Ollama（用于生成式升级层，可选）

### 4.2 启动决策服务

```bash
cd sentinel-stack

# 训练决策头并启动（首次会训练并缓存权重，约 3 秒）
PORT=9000 ./stackctl.sh start

# 健康检查
curl -s http://127.0.0.1:9000/healthz

# 实时看板
open http://127.0.0.1:9000/
```

### 4.3 调用决策接口

```bash
curl -X POST http://127.0.0.1:9000/v1/decide \
  -H 'content-type: application/json' \
  -d '{
    "conversation_id": "demo-1",
    "current_message": "设备状态上报",
    "conversation_state": {
      "structured_features": {
        "air_temp_c": 24.1, "process_temp_c": 37.6,
        "rpm": 1380, "torque_nm": 62.3, "tool_wear_min": 243
      },
      "feature_units": {
        "air_temp_c": "C", "process_temp_c": "C",
        "rpm": "rpm", "torque_nm": "Nm", "tool_wear_min": "min"
      }
    }
  }'
```

返回带概率分布的三种类型化决策 + 策略层给出的动作建议（`auto_dispatch` / `auto_close` /
`human_review` / `retrieve_evidence`）与触发原因码。

### 4.4 运行评测

```bash
# 框架内留出集端到端评测（在服务运行时）
curl "http://127.0.0.1:9000/v1/eval/dataset?n=1000&seed=7"

# 多数据集评测（需先准备数据集，见 analysis/multi_dataset_bench_v2.py 内的下载地址）
python3 analysis/multi_dataset_bench_v2.py --repeats 5
```

---

## 五、技术文档索引（`docs/`）

| 文档 | 内容 |
|---|---|
| `Jev架构多工业数据集能力评测报告 v2（修正版）.pdf` | ★ 跨 5 个工业子领域评测，含结论修正清单 |
| `哨兵×jev 集成验收报告.pdf` | 决策头接入框架的验收（留出集 + 阈值扫描 + 7 项安全保护） |
| `通用大模型vs垂类决策头·对决报告.pdf` | 垂类头 vs 27B/8B 的全指标对决 |
| `哨兵模型评估报告.pdf` | 完整指标集（Precision/Recall/F1/MCC/AUC/ECE/Brier） |
| `哨兵D2成果报告.pdf` | 先验校正 + CUDA 决策头 + 服务化 |
| `jev 各版本性能横向对比报告.pdf` | 框架 0.1.1 / 0.1.3 / 0.1.4 端到端性能对照 |
| `jev 0.1.3 六项缺陷定位与修复报告.pdf` | 含两处「静默失效」缺陷的定位与修复 |
| `jev 0.1.4 验收报告.pdf` / `jev 0.1.5 升级与效果对比报告.pdf` | 版本验收记录 |
| `Spark节点性能测评报告.html` / `Spark节点诊断报告.html` | 节点硬件实测与连通性诊断 |
| `征文-哨兵十日谈.docx` | 赛事征文（开发历程） |
| `工业场景公开数据集整理.docx` | 9 大类 40+ 工业公开数据集清单 |

---

## 六、说明与声明

### 6.1 关于凭据（重要）

本仓库为**公开仓库**，所有敏感信息已在导出时**统一脱敏**：

- 集群 SSH 密码、公网/内网 IP、平台入口地址 → 全部替换为 `CHANGE_ME_*` 占位符
- `tools/` 下的连接脚本需自行填入真实凭据后使用（通过环境变量或直接编辑）
- 部署记录中的绝对路径已归一化为 `/home/USER/...`

**请勿在提交中回填任何真实凭据、IP 或 API Key。**

### 6.2 组件来源

- `sentinel-stack/vendor/jev_service/`：决策治理框架 `jev-decision-service` **0.1.5**，团队自研
- 决策头、服务层、评测脚本：本仓库 `analysis/` 与 `sentinel-stack/` 内实现
- 前端与发布流程：`sentinel-stack/deploy-staging/`、`frontend-releases/`、`industrial-releases/`

### 6.3 数据集

未随仓库分发（体积与许可原因）。所用公开数据集：

| 数据集 | 来源 |
|---|---|
| AI4I 2020 Predictive Maintenance | UCI ML Repository #601 |
| Steel Plates Faults | UCI ML Repository #198 |
| SECOM | UCI ML Repository #179 |
| SKAB | GitHub `waico/SKAB` |
| Tennessee Eastman | GitHub `camaramm/tennessee-eastman-profBraatz` |

各加载器内含下载地址与预处理逻辑，见 `analysis/multi_dataset_bench_v2.py`。

### 6.4 关于本仓库的历史快照

`sentinel-stack/frontend-releases/` 与 `industrial-releases/` 保留了开发过程的多轮发布快照
（含 3D 场景模型 `.glb`），因此仓库体积较大。这是为了让评审能看到完整的迭代轨迹；
如仅需运行，只需 `sentinel-stack/` 顶层的代码与 `vendor/`。

---

*本仓库内容由 Machine not Learning 队维护。文中全部性能数据来自 DGX Spark 节点的实测结果，
除特别标注外均为可复现脚本产生。*
