#!/bin/zsh
###############################################################################
#  本机一键操作脚本（macOS）
#  功能：1) 检测节点是否可达  2) 可达则自动上传并执行 spark_bench.sh 并取回报告
#  用法：./spark_run.sh          # 检测 + 自动测评
#        ./spark_run.sh check    # 只做连通性检测
###############################################################################
set -uo pipefail

HOST=CHANGE_ME_NODE_HOST
PORT=6014
SUSER=USER
SPASS=CHANGE_ME_SSH_PASSWORD
BENCH="$(cd "$(dirname "$0")" && pwd)/spark_bench.sh"
MODE="${1:-run}"

banner_probe() {
  /Users/jialin_tian/.workbuddy/binaries/python/versions/3.13.12/bin/python3 - "$HOST" "$PORT" <<'PY'
import socket,sys
host,port=sys.argv[1],int(sys.argv[2])
s=socket.socket(); s.settimeout(30)
try:
    s.connect((host,port)); s.sendall(b'SSH-2.0-probe\r\n')
    d=s.recv(100)
    print("BANNER:", d[:60])
    sys.exit(0 if d.startswith(b'SSH-') else 3)
except Exception as e:
    print("NO-BANNER:", type(e).__name__)
    sys.exit(3)
finally:
    s.close()
PY
}

echo "▶ 步骤 1/3  检测 $HOST:$PORT ..."
if ! banner_probe; then
  echo "✗ 节点 SSH 服务无响应（公网网关接受 TCP 但后端不回包）"
  echo "  → 节点未就绪，联系组委会；无需反复重试，问题在服务端。"
  exit 1
fi
echo "✓ SSH 服务在线"
[[ "$MODE" == "check" ]] && exit 0

echo "▶ 步骤 2/3  上传测评脚本并执行（预计 3~8 分钟）"
/usr/bin/expect <<EXP
set timeout 1800
log_user 1
spawn ssh -p $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PreferredAuthentications=password -o PubkeyAuthentication=no $SUSER@$HOST "cat > ~/spark_bench.sh && bash ~/spark_bench.sh"
expect {
    -re "(P|p)assword:" { send "$SPASS\r"; exp_continue }
    "yes/no" { send "yes\r"; exp_continue }
    eof
}
EXP
rc=$?

echo "▶ 步骤 3/3  取回报告"
STAMP=$(date +%Y%m%d_%H%M%S)
LOCALOUT="$(dirname "$BENCH")/spark_report_$STAMP"
mkdir -p "$LOCALOUT"
/usr/bin/expect <<EXP
set timeout 600
log_user 1
spawn scp -P $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -r $SUSER@$HOST:~/spark_bench_*/report.txt $LOCALOUT/
expect {
    -re "(P|p)assword:" { send "$SPASS\r"; exp_continue }
    "yes/no" { send "yes\r"; exp_continue }
    eof
}
EXP
echo "✅ 报告已保存到 $LOCALOUT"
ls -la "$LOCALOUT"
