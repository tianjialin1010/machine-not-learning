#!/bin/bash
# 哨兵 × jev 集成服务 管理脚本
#   ./sentinelctl.sh start|stop|restart|status|logs|meta
# 环境变量：
#   PORT=9000            监听端口
#   ESCALATE=1           开启「低置信升级生成式」
#   RETRAIN=1            重新训练决策头
#   EXTRA="--escalate-floor 0.75"   追加参数
set -u
PORT="${PORT:-9000}"
DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$DIR/service-$PORT.log"
PAT="sentinel_integrate[d].py"          # 字符类写法，避免匹配到本脚本自身
PY="${PY:-python3}"

is_running() { pgrep -f "$PAT" >/dev/null 2>&1; }

do_start() {
  if is_running; then
    echo "[已运行] PID $(pgrep -f "$PAT" | tr '\n' ' ')"
    return 0
  fi
  ARGS=(--port "$PORT")
  [ "${ESCALATE:-0}" = "1" ] && ARGS+=(--escalate)
  [ "${RETRAIN:-0}" = "1" ] && ARGS+=(--retrain)
  [ -n "${EXTRA:-}" ] && ARGS+=($EXTRA)
  cd "$DIR" || exit 1
  setsid nohup "$PY" -u sentinel_integrated.py "${ARGS[@]}" \
    >"$LOG" 2>&1 </dev/null &
  for _ in $(seq 1 60); do
    sleep 1
    if is_running && curl -s -m 2 -o /dev/null "http://127.0.0.1:$PORT/healthz"; then
      echo "[启动成功] 端口 $PORT  PID $(pgrep -f "$PAT" | tr '\n' ' ')"
      return 0
    fi
  done
  echo "[启动失败] 日志尾部："
  tail -20 "$LOG"
  return 1
}

do_stop() {
  if ! is_running; then echo "[未运行]"; return 0; fi
  pkill -f "$PAT"
  for _ in $(seq 1 15); do
    sleep 1
    is_running || { echo "[已停止]"; return 0; }
  done
  pkill -9 -f "$PAT"; sleep 1
  echo "[强制停止]"
}

case "${1:-status}" in
  start)   do_start ;;
  stop)    do_stop ;;
  restart) do_stop; sleep 2; do_start ;;
  status)
    if is_running; then
      echo "[运行中] PID $(pgrep -f "$PAT" | tr '\n' ' ')"
      curl -s -m 3 "http://127.0.0.1:$PORT/healthz" || echo "  (健康检查失败)"
      echo
    else
      echo "[未运行] 日志：$LOG"
    fi
    ;;
  logs)  tail -"${2:-40}" "$LOG" ;;
  meta)  curl -s -m 5 "http://127.0.0.1:$PORT/v1/meta" ;;
  *)     echo "用法: $0 {start|stop|restart|status|logs [n]|meta}"; exit 1 ;;
esac
