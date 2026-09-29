#!/usr/bin/env bash
###############################################################################
#  jev-decision-service 节点管理脚本
#  用法: ./jevctl.sh start | stop | restart | status | logs [端口]
#
#  端口规划（对应该节点的公网映射）:
#    8888 (公网 8014) — opensource 模式，调用本地 ollama 27B 真实模型出决策
#                       失败自动降级为 rules（ResilientProvider 已内置）
#    9000 (公网 9014) — rules 模式，纯本地规则，毫秒级响应，永远可用
###############################################################################
set -uo pipefail

DIR="$HOME/jev-service"
MODEL="modelscope.cn/unsloth/Qwen3.8-27B-GGUF:latest"
ENDPOINT="http://127.0.0.1:11434/v1/chat/completions"

start_one() {
  local port=$1 mode=$2
  if ss -tln 2>/dev/null | grep -q ":${port} "; then
    echo "  端口 $port 已在运行，跳过"
    return
  fi
  JEV_PROVIDER="$mode" \
  OPEN_SOURCE_ENDPOINT="$ENDPOINT" \
  OPEN_SOURCE_MODEL="$MODEL" \
  OPEN_SOURCE_API_KEY="EMPTY" \
  PYTHONPATH=src \
  setsid python3 -m jev_service.server --host 0.0.0.0 --port "$port" \
    </dev/null > "$DIR/jev-$port.log" 2>&1 &
  sleep 1
  echo "  已启动 :$port  ($mode)"
}

case "${1:-status}" in
  start)
    cd "$DIR" || exit 1
    echo "启动 JEV 决策服务..."
    start_one 8888 opensource
    start_one 9000 rules
    sleep 2
    echo
    echo "健康检查:"
    curl -s -m 10 "http://127.0.0.1:8888/healthz"; echo
    curl -s -m 10 "http://127.0.0.1:9000/healthz"; echo
    ;;
  stop)
    echo "停止所有 JEV 服务..."
    pkill -f "^python3 -m jev_service.server" 2>/dev/null
    sleep 1
    echo "  已停止"
    ;;
  restart)
    "$0" stop
    sleep 1
    "$0" start
    ;;
  status)
    echo "监听状态:"
    ss -tlnp 2>/dev/null | grep -E ":(8888|9000) " || echo "  (无服务在运行)"
    echo
    echo "健康检查:"
    curl -s -m 10 "http://127.0.0.1:8888/healthz"; echo "   <- :8888 (公网 8014)"
    curl -s -m 10 "http://127.0.0.1:9000/healthz"; echo "   <- :9000 (公网 9014)"
    ;;
  logs)
    port="${2:-8888}"
    tail -n 40 "$DIR/jev-$port.log" 2>/dev/null || echo "无日志: $DIR/jev-$port.log"
    ;;
  *)
    echo "用法: $0 start|stop|restart|status|logs [端口]"
    ;;
esac
