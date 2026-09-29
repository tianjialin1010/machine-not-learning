# jev-decision-service 0.1.3 问题清单（技术对接用）

> 全部行号基于 0.1.3 原始代码。验证环境：NVIDIA DGX Spark 节点 + ollama + Qwen3-8B 真机。
> 一键修复脚本：`patch_jev_service.py`（3 个文件、7 处修改，幂等，可重复执行）

---

## 问题 5【严重·安全】启用校准后，「紧急度保护」静默失效

**这是最需要优先处理的一条。不崩溃、不报错，但把最该拦的报警放过去了。**

### 位置

- 出问题的地方：`src/jev_service/calibration.py:83`
- 受害的地方：`src/jev_service/policy.py:66`

### 现象

`severity = 3`（最高级）时：

| 配置 | `risk_level` 的 value | policy 最终动作 | 紧急度保护 |
|---|---|---|---|
| `calibrated=False` | `3.0` (float) | `human_review` | 生效 |
| `calibrated=True` | `'立即'` (str) | `answer_from_context` | **失效** |

### 根因

`calibration.py` 第 78–83 行，多元校准分支：

```python
elif profile.calibrated and prediction.probabilities:
    keys = list(prediction.probabilities)
    logits = [math.log(max(prediction.probabilities[key], 1e-9)) for key in keys]
    scaled = _softmax([value / max(profile.temperature, 1e-6) for value in logits])
    prediction.probabilities = {key: round(value, 6) for key, value in zip(keys, scaled)}
    prediction.value = max(prediction.probabilities, key=prediction.probabilities.get)   # ← 第 83 行
    prediction.calibration_status = "temperature_scaled"
```

第 83 行**无条件**把 `value` 写成"概率最高的标签名"。对 `choice` 类型这是对的；对 `score` 类型就错了——它的 `value` 必须是数值。

然后 `policy.py:66` 是这样判断紧急度的：

```python
if high_risk or (tension and isinstance(tension.value, (float, int)) and float(tension.value) >= config.urgent_score):
```

`tension` 取的是 `risk_level`（`score` 类型）。`value` 变成 `'立即'` 之后，`isinstance(..., (float, int))` 为 `False`，**整个条件被跳过**，直接落到后面的"低风险"分支。于是最高级别报警被当作普通消息回复。

### 修法

`calibration.py:83` 按类型分别处理：

```python
            if prediction.type == "choice":
                prediction.value = max(prediction.probabilities,
                                       key=prediction.probabilities.get)
            elif prediction.type == "score":
                best = max(range(len(keys)), key=lambda index: scaled[index])
                prediction.value = float(best)
```

不需要改 `policy.py`——它本身没问题，是上游把类型弄坏了。

---

## 问题 6【严重·可用性】「稀有类」告警是静态全局的，声明即永久转人工

### 位置

- 告警生成：`src/jev_service/calibration.py:67-68`
- 拦截逻辑：`src/jev_service/policy.py:58-59`

### 现象

同一台**健康设备**（无故障）：

| profile 是否声明 class_support | warnings | policy 动作 | 是否拦截 |
|---|---|---|---|
| 未声明 | `['label_provenance:derived_rule']` | `answer_from_context` | 否 |
| 声明了 `{"TWF": 6, "HDF": 29, "RNF": 6}` | `[..., 'rare_classes:RNF,TWF']` | `human_review` | **是** |

### 根因

`calibration.py:67-68`：

```python
    if profile.min_class_support and profile.class_support:
        rare = [label for label, support in profile.class_support.items() if support < profile.rare_class_threshold]
        if rare:
            warnings.append("rare_classes:" + ",".join(sorted(rare)))
```

这里只检查 **profile 里声明的类别样本数**，跟**本次输入、本次预测**毫无关系。只要 profile 里存在任意一个样本数低于阈值的类，这条告警就永远存在。

`policy.py:58-59` 又是无条件拦截：

```python
    if "rare_classes" in " ".join(warnings):
        reasons.append("rare_class_support_insufficient")
        return ActionRecommendation("human_review", True, reasons, None, provider_status, True)
```

**净效果：越诚实地标注"我们某个类样本不足"，系统就越不可用——所有请求被永久转人工，自动化率归零。**

### 修法

改为"仅当本次预测确实命中稀有类时才拦截"：

```python
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
                                        provider_status, True)
```

注：这段依赖问题 5 已修复（`score` 的 `value` 与 `probabilities` 一致）。

---

## 问题 1~4：opensource 路径（`JEV_PROVIDER=opensource`）完全跑不通

**真机实测：原版 0/3 成功，修复后 3/3 成功。**

### 问题 1【根因】提示词要求的键名，与解析器接受的键名不一致

- 提示词：`provider.py:276-280`
- 解析器：`provider.py:155` `_parse_jev_answer`

提示词只说：

```
"Return JSON only with an answers object. Answer every question using the declared "
"choice options, numeric score levels, or noul yes probability."
```

**从没要求把值放在 `choice` / `score` / `noul` 这些键上。** 而解析器（第 155 行起）只认这三个原生键：

```python
    if answer_type == "choice" or "choice" in answer:
        ...
        choice = answer.get("choice")
        if choice is None or str(choice) not in probabilities:
            raise ProviderError(f"choice is missing from probabilities for {identifier}", retryable=False)   # 第 170 行
```

真实模型按字面理解，输出 `{"type":"choice","value":"clarification",...}`，解析器找不到 `choice` 键 → 抛错。

> 补充：`_parse_jev_answer` 在 0.1.0 与 0.1.3 中**逐字节完全相同**（2574 字节），提示词也一字未改——这个缺陷从 0.1.0 一直带到 0.1.3。

**修法**：重写 system prompt，明确列出三种形状与键名、给示例、显式禁止 `value` 键。

### 问题 2 概率和不严格为 1

- 校验位置：`provider.py:166-167`

```python
        if abs(sum(probabilities.values()) - 1) > 0.01:
            raise ProviderError(f"choice probabilities do not sum to one for {identifier}", retryable=False)
```

真机实测模型常给出 `{"high":0.4,"moderate":0.3,...}` 这类和不严格为 1 的分布（偏差经常超过 0.01）→ 抛错。

**修法**：交给校验器之前做归一化。

### 问题 3 `score` 越界

- 校验位置：`provider.py:177-179`

```python
        score = float(answer.get("score", 0.0))
        if not math.isfinite(score) or score < 0 or score > max(0, len(probabilities) - 1):
            raise ProviderError(f"score is out of range for {identifier}", retryable=False)
```

四档量表模型会理解成 `1~4`，给出 `4` → 超出 `[0, 3]` → 抛错。

**修法**：限幅到合法区间；另支持把"选项名"形式的 score 映射为下标。

### 问题 4 多问嵌套 JSON 被截断

- payload：`provider.py:282-286`

```python
        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [...],
        }
```

**没有设置 `max_tokens`。** ollama 的 OpenAI 兼容接口默认只生成 **128 token**，六个问题的嵌套 JSON 会被截断 → 形状校验失败。

**修法**：payload 增加 `"max_tokens": self.max_tokens`，并在 dataclass 上增加 `max_tokens: int = 1024`。

### 共同点（建议一并处理）

以上四类错误**全部标为 `retryable=False`**，意味着模型任何一处输出瑕疵都会让整单请求硬失败，**而不会降级到 rules provider**。

建议把"格式类错误"改为 `retryable=True`，让 `ResilientProvider` 能兜底。

---

## 修复方式

```bash
# 应用到包根目录（含 src/）
python3 patch_jev_service.py /path/to/jev-decision-service

# 只检测已应用情况
python3 patch_jev_service.py --check /path/to/jev-decision-service

# 也支持直接指向单个文件
python3 patch_jev_service.py /path/to/src/jev_service/provider.py
```

补丁幂等，重复执行不会重复插入。应用后请执行：

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

预期：29 个测试全部通过。

## 修改文件清单

| 文件 | 处数 | 内容 |
|---|---|---|
| `provider.py` | 5 | system prompt 键名、payload `max_tokens`、dataclass 字段、输出清洗函数、清洗调用 |
| `calibration.py` | 1 | `score` 类型的 `value` 保持数值 |
| `policy.py` | 1 | 稀有类告警改为"命中才拦截" |

## 验证脚本（真机）

| 脚本 | 用途 |
|---|---|
| `jev_real_test.py` | 原版 vs 补丁版在 ollama 真模型上的成功率对比 |
| `calib_check.py` | 校验注入 `DecisionHeadProfile` 后能否走到自动化分支 |
| `sentinel_jev_bridge.py` | 哨兵决策头接入 `StructuredHeadProvider` 的端到端演示 |
