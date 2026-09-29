#!/bin/bash
# 哨兵 System One 决策服务 管理脚本
#   ./sentinelctl.sh start|stop|restart|status|logs
#
# 注意：pkill -f 会匹配到 bash -c 自身的命令行导致自杀，
# 因此统一用 "sentinel_[s]erver.py" 这种字符类写法规避。

DIR="$HOME/jev-service"
PORT=9000
LOG="$DIR/sentinel-$PORT.log"
PATTERN="sentinel_[s]erver.py"     # 字符类：不匹配本脚本自身的命令行

cd "$DIR" || exit 1

case "$1" in
  start)
    if pgrep -f "$PATTERN" >/dev/null; then
      echo "已在运行 (pid $(pgrep -f "$PATTERN" | head -1))"
      exit 0
    fi
    setsid nohup python3 -u sentinel_server.py --port "$PORT" \
      > "$LOG" 2>&1 < /dev/null &
    echo "启动中…（首次需训练三个决策头，约 5~10 秒）"
    for i in $(seq 1 30); do
      sleep 1
      if curl -s -m 3 "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; then
        echo "就绪，用时 ${i}s"
        curl -s "http://127.0.0.1:$PORT/healthz"; echo
        exit 0
      fi
    done
    echo "启动超时，请查看日志：$LOG"
    ;;
  stop)
    pkill -f "$PATTERN" && echo "已停止" || echo "未在运行"
    ;;
  restart)
    "$0" stop; sleep 2; "$0" start
    ;;
  status)
    if pgrep -f "$PATTERN" >/dev/null; then
      echo "运行中 (pid $(pgrep -f "$PATTERN" | head -1))，$PORT 端口"
      curl -s -m 5 "http://127.0.0.1:$PORT/healthz"; echo
    else
      echo "未运行"
    fi
    ;;
  logs)
    tail -n "${2:-30}" "$LOG"
    ;;
  *)
    echo "用法: $0 start|stop|restart|status|logs [行数]"
    exit 1
    ;;
esac
