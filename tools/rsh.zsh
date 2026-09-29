#!/bin/zsh
###############################################################################
# 远程执行助手：把命令写入临时文件，交给 expect 原样传给 ssh，避免引号地狱
# 用法：./rsh.zsh "远程命令"        # 命令可以是任意含引号/换行的字符串
#       ./rsh.zsh -f ./cmd.sh      # 从文件读取远程命令
###############################################################################
set -uo pipefail

export SSH_HOST=CHANGE_ME_NODE_HOST
export SSH_PORT=6014
export SSH_USER=USER
export SSH_PASS=CHANGE_ME_SSH_PASSWORD
export SSH_TIMEOUT=${SSH_TIMEOUT:-1800}

TMPCMD=$(mktemp /tmp/rsh_cmd.XXXXXX)
trap 'rm -f "$TMPCMD"' EXIT

if [[ "${1:-}" == "-f" ]]; then
  cp "$2" "$TMPCMD"
else
  printf '%s' "$1" > "$TMPCMD"
fi
export SSH_CMD_FILE="$TMPCMD"

/usr/bin/expect <<'EXP'
set timeout $env(SSH_TIMEOUT)
log_user 1
set f [open $env(SSH_CMD_FILE)]
set cmd [read $f]
close $f
spawn ssh -p $env(SSH_PORT) \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  -o PreferredAuthentications=password \
  -o PubkeyAuthentication=no \
  -o ServerAliveInterval=30 \
  $env(SSH_USER)@$env(SSH_HOST) $cmd
expect {
    -re "(P|p)assword:" { send "$env(SSH_PASS)\r"; exp_continue }
    "yes/no" { send "yes\r"; exp_continue }
    eof
}
EXP
exit 0
