#!/bin/zsh
###############################################################################
#  文件上传助手：本机 → 云节点（走 expect，自动填密码）
#  用法：./scput.zsh <本地文件> [远端路径，默认 ~/]
###############################################################################
set -uo pipefail

LOCAL="${1:?用法: ./scput.zsh <本地文件> [远端路径]}"
REMOTE="${2:-~/}"

export SCP_HOST=CHANGE_ME_NODE_HOST
export SCP_PORT=6014
export SCP_USER=USER
export SCP_PASS=CHANGE_ME_SSH_PASSWORD
export SCP_LOCAL="$LOCAL"
export SCP_REMOTE="$REMOTE"
export SSH_TIMEOUT=${SSH_TIMEOUT:-600}

/usr/bin/expect <<'EXP'
set timeout $env(SSH_TIMEOUT)
log_user 1
spawn scp -P $env(SCP_PORT) \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  -o PreferredAuthentications=password \
  -o PubkeyAuthentication=no \
  $env(SCP_LOCAL) $env(SCP_USER)@$env(SCP_HOST):$env(SCP_REMOTE)
expect {
    -re "(P|p)assword:" { send "$env(SCP_PASS)\r"; exp_continue }
    "yes/no" { send "yes\r"; exp_continue }
    eof
}
EXP
exit 0
