#!/bin/bash
# 哨兵独立决策服务（sentinel_server.py）管理脚本
#   ./sentinelctl.sh start|stop|restart|status|logs [端口]
# 说明：这是**不依赖 jev 框架**的独立版（自带 /v1/decide/batch 与看板）。
#       与 jev 深度集成的版本由 stackctl.sh 管理。
set -u
PORT="${1:-9000}"
ACTION="${1:-status}"
[ "$ACTION" = "logs" ] && PORT="${2:-9000}"
DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$DIR/sentinel-$PORT.log"
PAT="sentinel_[s]erver.py"        # 字符类写法，避免匹配到本脚本命令行
PY="${PY:-python3}"

is_running() { pgrep -f "$PAT" >/dev/null 2>&1; }

case "$ACTION" in
  start)
    if is_running; then echo "[已运行] PID $(pgrep -f "$PAT" | tr '\n' ' ')"; exit 0; fi
    cd "$DIR" || exit 1
    setsid nohup "$PY" -u sentinel_server.py --port "$PORT" \
      >"$LOG" 2>&1 </dev/null &
    for _ in $(seq 1 40); do
      sleep 1
      curl -s -m 2 -o /dev/null "http://127.0.0.1:$PORT/healthz" && {
        echo "[启动成功] 端口 $PORT"; exit 0; }
    done
    echo "[启动失败]"; tail -15 "$LOG"; exit 1 ;;
  stop)
    if ! is_running; then echo "[未运行]"; exit 0; fi
    pkill -f "$PAT"; sleep 2
    is_running && { pkill -9 -f "$PAT"; sleep 1; }
    echo "[已停止]" ;;
  restart) "$0" stop; sleep 2; "$0" start ;;
  status)
    if is_running; then
      echo "[运行中] PID $(pgrep -f "$PAT" | tr '\n' ' ')"
      curl -s -m 3 "http://127.0.0.1:$PORT/healthz"; echo
    else echo "[未运行]"; fi ;;
  logs) tail -"${3:-40}" "$LOG" ;;
  *) echo "用法: $0 {start|stop|restart|status|logs [端口]}"; exit 1 ;;
esac
