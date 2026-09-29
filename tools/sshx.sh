#!/bin/zsh
# usage: sshx.sh "<remote command>"
REMOTE="$1"
HOST=CHANGE_ME_NODE_HOST
PORT=6014
USER=USER
PASS=CHANGE_ME_SSH_PASSWORD
/usr/bin/expect <<EXP
set timeout 600
log_user 1
spawn ssh -p $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PreferredAuthentications=password -o PubkeyAuthentication=no $USER@$HOST "$REMOTE"
expect {
    -re "(P|p)assword:" { send "$PASS\r"; exp_continue }
    "yes/no" { send "yes\r"; exp_continue }
    eof
}
EXP
