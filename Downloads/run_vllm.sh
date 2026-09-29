#!/usr/bin/env bash
# ============================================================
# DGX Spark (arm64) 本机 vLLM 服务管理脚本
#   NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 (NemotronH, NVFP4)
#
# 用法:
#   bash run_vllm.sh start           启动服务(后台,nohup)
#   bash run_vllm.sh stop            停止服务
#   bash run_vllm.sh status          查看运行状态
#   bash run_vllm.sh logs            查看/跟随日志 (Ctrl+C 退出跟随)
#   bash run_vllm.sh test            跑一次对话测试
# ============================================================
set -e

VENV="$HOME/envs/vllm"
MODEL_DIR="/home/${USER:-$(whoami)}/models/Nemotron-3.5-Lightning-30B-A3B-NVFP4"
PORT="8000"
SERVED_NAME="nemotron"
LOG="/tmp/vllm_serve.log"
MAX_LEN="${MAX_LEN:-32768}"
GPU_UTIL="${GPU_UTIL:-0.9}"

PY="$VENV/bin/python"
CMD="$VENV/bin/vllm"

is_running() { pgrep -f "vllm serve" >/dev/null 2>&1; }

case "${1:-start}" in
  start)
    [ -d "$MODEL_DIR" ] || { echo "❌ 模型目录不存在: $MODEL_DIR"; exit 1; }
    if is_running; then echo "⚠️  服务已在运行 (端口 $PORT)。如要重启先 stop。"; exit 0; fi
    echo "== 启动 vLLM ($($PY -c 'import vllm;print(vllm.__version__)'))"
    echo "   模型: $MODEL_DIR"
    echo "   日志: $LOG"
    nohup "$CMD" serve "$MODEL_DIR" \
        --served-model-name "$SERVED_NAME" \
        --max-model-len "$MAX_LEN" \
        --gpu-memory-utilization "$GPU_UTIL" \
        --trust-remote-code \
        > "$LOG" 2>&1 &
    echo "== 已后台启动 PID=$!，等待服务就绪..."
    for i in $(seq 1 60); do
        if curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/v1/models" 2>/dev/null | grep -q 200; then
            echo "✅ 服务就绪: http://localhost:$PORT/v1/models (模型: $SERVED_NAME)"
            exit 0
        fi
        sleep 5
    done
    echo "⚠️  5分钟内未就绪，请用 logs 查看: $LOG"
    ;;
  stop)
    pkill -f "vllm serve" 2>/dev/null && echo "✅ 已停止 vLLM" || echo "未发现运行中的 vLLM"
    ;;
  status)
    if is_running; then
        echo "✅ vLLM 运行中 (端口 $PORT)"
        curl -s "http://localhost:$PORT/v1/models" | head -c 300; echo ""
    else
        echo "❌ vLLM 未运行"
    fi
    ;;
  logs)
    tail -f "$LOG"
    ;;
  test)
    curl -s "http://localhost:$PORT/v1/chat/completions" \
         -H "Content-Type: application/json" \
         -d "{\"model\":\"$SERVED_NAME\",\"max_tokens\":96,\"messages\":[{\"role\":\"user\",\"content\":\"用一句话介绍你自己\"}]}"
    echo ""
    ;;
  *)
    echo "未知命令: $1"; grep '^#   bash' "$0" | sed 's/^#   //'; exit 1 ;;
esac
