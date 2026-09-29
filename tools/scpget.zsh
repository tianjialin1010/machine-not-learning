#!/bin/zsh
###############################################################################
#  文件下载助手：云节点 → 本机（走 expect，自动填密码）
#  用法：./scpget.zsh <远端文件> [本地路径，默认当前目录]
###############################################################################
set -uo pipefail

REMOTE="${1:?用法: ./scpget.zsh <远端文件> [本地路径]}"
LOCAL="${2:-.}"

export SCP_HOST=CHANGE_ME_NODE_HOST
export SCP_PORT=6014
export SCP_USER=USER
export SCP_PASS=CHANGE_ME_SSH_PASSWORD
export SCP_REMOTE="$REMOTE"
export SCP_LOCAL="$LOCAL"
export SSH_TIMEOUT=${SSH_TIMEOUT:-1200}

/usr/bin/expect <<'EXP'
set timeout $env(SSH_TIMEOUT)
# 静默模式：scp 的进度条会把 expect 的 stdout 撑爆（大文件时进程会被杀），
# 因此关闭日志输出，只依赖退出码判断结果。
log_user 0
spawn scp -q -P $env(SCP_PORT) \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  -o PreferredAuthentications=password \
  -o PubkeyAuthentication=no \
  $env(SCP_USER)@$env(SCP_HOST):$env(SCP_REMOTE) $env(SCP_LOCAL)
expect {
    -re "(P|p)assword:" { send "$env(SCP_PASS)\r"; exp_continue }
    "yes/no" { send "yes\r"; exp_continue }
    eof
}
catch wait result
exit [lindex $result 3]
EXP
exit $?
