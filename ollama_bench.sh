#!/usr/bin/env bash
###############################################################################
#  Ollama 推理实测：token 吞吐 / 首字延迟 / 显存占用
#  用法：bash ollama_bench.sh [模型名]
#  不带参数则自动选用 ollama list 里第一个模型
###############################################################################
set -uo pipefail
MODEL="${1:-$(ollama list | awk 'NR==2{print $1}')}"
[[ -z "$MODEL" ]] && { echo "没有可用模型"; exit 1; }

echo "模型: $MODEL"
echo "== 预热 =="
curl -s http://localhost:11434/api/generate \
  -d "{\"model\":\"$MODEL\",\"prompt\":\"你好\",\"stream\":false,\"options\":{\"num_predict\":8}}" \
  >/dev/null

PROMPT="请用大约 300 字的中文，解释一下为什么统一内存架构在端侧跑大模型时有优势。"
echo "== 正式测速（生成 256 token）=="
RESP=$(curl -s http://localhost:11434/api/generate -d "{
  \"model\":\"$MODEL\",
  \"prompt\":\"$PROMPT\",
  \"stream\":false,
  \"options\":{\"num_predict\":256,\"temperature\":0.6}
}")

python3 - "$RESP" <<'PY'
import json,sys
try:
    d=json.loads(sys.argv[1])
except Exception as e:
    print("解析失败:",e); print(sys.argv[1][:500]); raise SystemExit

ec=d.get("eval_count",0); ed=d.get("eval_duration",1)/1e9
pc=d.get("prompt_eval_count",0); pd=d.get("prompt_eval_duration",1)/1e9
total=d.get("total_duration",0)/1e9

print("  ── 生成阶段 ──")
print("  输出 token 数 : %d" % ec)
print("  生成耗时      : %.2f s" % ed)
print("  生成速度      : %.1f tokens/s   ← 核心指标" % (ec/ed if ed else 0))
print("  ── 预填阶段 ──")
print("  输入 token 数 : %d" % pc)
print("  预填耗时      : %.2f s" % pd)
print("  预填速度      : %.1f tokens/s" % (pc/pd if pd else 0))
print("  ── 汇总 ──")
print("  端到端总耗时  : %.2f s" % total)
print()
print("  参考：DGX Spark(GB10, 273 GB/s) 跑 27B 级 GGUF 模型，")
print("        30 tokens/s 以上属于正常，低于 15 需排查是否跑在 CPU 上。")
PY

echo
echo "== 推理后 GPU 状态 =="
nvidia-smi --query-gpu=name,memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv 2>/dev/null
echo
echo "== 推理后内存 =="
free -h | head -2
