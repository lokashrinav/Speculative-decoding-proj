#!/bin/bash
# One-time setup on a Nexus submission node. Creates a key for this repo only,
# waits until GitHub accepts it, clones the nexus-jobs branch, starts the runner.
REPO=lokashrinav/Speculative-decoding-proj
KEY="$HOME/.ssh/specdec_deploy"
DEST="$HOME/specdec-runner"
mkdir -p "$HOME/.ssh" && chmod 700 "$HOME/.ssh"
[ -f "$KEY" ] || ssh-keygen -q -t ed25519 -N "" -C "nexus-runner" -f "$KEY"

try() { # host port -> prints ssh's reply
  ssh -T -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new \
      -o ConnectTimeout=10 -o BatchMode=yes -p "$2" "git@$1" </dev/null 2>&1
}
HOST=""; PORT=""
for hp in "github.com 22" "ssh.github.com 443"; do
  set -- $hp
  if try "$1" "$2" | grep -q -E "Permission denied|successfully authenticated"; then HOST=$1; PORT=$2; break; fi
done
if [ -z "$HOST" ]; then echo "ERROR: this node cannot reach GitHub over SSH (ports 22 and 443 both failed)."; exit 1; fi

if ! grep -q "^Host github-specdec$" "$HOME/.ssh/config" 2>/dev/null; then
  printf '\nHost github-specdec\n  HostName %s\n  Port %s\n  User git\n  IdentityFile %s\n  IdentitiesOnly yes\n  StrictHostKeyChecking accept-new\n' \
    "$HOST" "$PORT" "$KEY" >> "$HOME/.ssh/config"
  chmod 600 "$HOME/.ssh/config"
fi

echo
echo "===== COPY THE NEXT LINE AND SEND IT TO CLAUDE ====="
cat "$KEY.pub"
echo "===================================================="
echo "Waiting for the key to be added on GitHub (checks every 10 seconds, up to 30 minutes)..."
ok=0
for i in $(seq 1 180); do
  if ssh -T -o BatchMode=yes github-specdec </dev/null 2>&1 | grep -q "successfully authenticated"; then ok=1; break; fi
  sleep 10
done
[ "$ok" = 1 ] || { echo "ERROR: key was not accepted in time. Run this command again later."; exit 1; }

if [ ! -d "$DEST/.git" ]; then
  git clone -q -b nexus-jobs "git@github-specdec:$REPO.git" "$DEST" </dev/null || { echo "ERROR: clone failed"; exit 1; }
fi
if command -v tmux >/dev/null; then
  tmux kill-session -t specdec 2>/dev/null
  tmux new -d -s specdec "bash $DEST/nexus/runner.sh 2>&1 | tee -a $HOME/specdec-runner.log"
else
  nohup setsid bash "$DEST/nexus/runner.sh" >> "$HOME/specdec-runner.log" 2>&1 &
fi
echo "DONE: runner started on $(hostname). You can close this window."
