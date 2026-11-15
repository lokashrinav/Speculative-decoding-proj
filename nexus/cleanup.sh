#!/bin/bash
# Frees space in the home folder on Nexus after it filled up on Nov 15, then restarts the runner.
# It deletes two things only, and both can be recreated:
#   1. ~/specdec-code                   copies of this repository that jobs ran from
#   2. ~/specdec-data/long1/rounds1     the per-round records of the replay that filled the disk
# Before that it cancels the nine jobs queued on Nov 15 (they cannot run without space).
# Nothing else is deleted: models, checkpoints, the saved answers and the environment stay.
set -u
[ -n "${HOME:-}" ] && cd "$HOME" || { echo "ERROR: no home folder"; exit 1; }
CODE_DIR="$HOME/specdec-code"
ROUNDS="$HOME/specdec-data/long1/rounds1"
REPORT="/tmp/specdec-disk-$USER.txt"

for name in replay-probe-0 replay-probe-1 replay-probe-2 replay-probe-3 fit-gate \
            online-bank-0 online-bank-1 online-bank-2 online-bank-3; do
  scancel -u "$USER" -n "$name" 2>/dev/null
done
sleep 5
{
  echo "== cleanup at $(date -u +%FT%TZ) on $(hostname)"
  echo "== jobs still listed"; squeue -u "$USER" -o "%i %j %T %M" 2>&1
  echo "== sizes before"
  du -sh "$CODE_DIR" "$ROUNDS" 2>/dev/null
  echo "copies of the repository: $(find "$CODE_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)"
  echo "per-round record files: $(find "$ROUNDS" -type f 2>/dev/null | wc -l)"
} > "$REPORT" 2>&1
cat "$REPORT"

# A copy that a job is still running from is left alone.
busy=$(squeue -h -u "$USER" -o "%j-%i" 2>/dev/null)
if [ -d "$CODE_DIR" ]; then
  for dir in "$CODE_DIR"/*/; do
    [ -d "$dir" ] || continue
    name=$(basename "$dir")
    if printf '%s\n' "$busy" | grep -qxF "$name"; then echo "kept (in use): $name"; continue; fi
    rm -rf -- "$CODE_DIR/$name"
  done
fi
[ -d "$ROUNDS" ] && rm -rf -- "$ROUNDS"

echo "Measuring what is left (this can take a minute or two) ..."
{
  echo "== sizes after"
  du -sh "$HOME/specdec-code" "$HOME/specdec-data" "$HOME/specdec-ckpt" "$HOME/specdec-runner" "$HOME/specdec-slurm" \
         "$HOME/hf-cache" "$HOME/specdec-env" "$HOME/.cache" 2>/dev/null
  echo "== whole home folder"; du -sh "$HOME" 2>/dev/null
  echo "== can write again:"; if touch "$HOME/.specdec-write-test" 2>/dev/null; then echo yes; else echo NO; fi
} >> "$REPORT" 2>&1
sed -n '/== sizes after/,$p' "$REPORT"

# Restart the runner with the current scripts; it sends this report along.
if curl -fsSL https://raw.githubusercontent.com/lokashrinav/Speculative-decoding-proj/444ddaf1f06e50ac585bb62393cdb4c885e43b00/nexus/restart.sh | bash; then
  cp "$REPORT" "$HOME/specdec-runner/nexus/status/disk-report.txt" 2>/dev/null
fi
