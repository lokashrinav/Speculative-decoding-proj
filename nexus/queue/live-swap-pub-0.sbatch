#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=10:00:00
# Real tokens per second with the public drafter on validation prompts whose answers are 2,000 tokens
# and more: the swap (the first draft with the drafter's second choice at its four likely breaks, in
# the target's one check, NO second drafter pass), the context window (the drafter sees only the last
# 3,072 tokens, the length it was trained on), the add-on weights taught ahead of time, and these
# together. The settings with alternatives follow plain decoding's answer (nexus/live_speed.py), so
# every setting writes the same text. The settings: nexus/settings/live-swap-pub.txt (first line).
ENV="$HOME/specdec-env"
CODE="$HOME/specdec-code/$JOB_NAME-$SLURM_JOB_ID"
export HF_HOME="$HOME/hf-cache" TOKENIZERS_PARALLELISM=false
module load Python3/3.12.3 >/dev/null 2>&1 || true
# The code is copied from the runner's folder as committed there (under 1 MB), not cloned again.
mkdir -p "$CODE" && ( set -o pipefail; git -C "$HOME/specdec-runner" archive HEAD dflash nexus tests \
  | tar -x -C "$CODE" --exclude='nexus/results' --exclude='nexus/logs' --exclude='nexus/status' ) || exit 1
source "$ENV/bin/activate" || exit 1
cd "$CODE" || exit 1
export PYTHONPATH="$CODE"
git -C "$HOME/specdec-runner" rev-parse HEAD > "$RESULTS_DIR/code_commit.txt"
SETTINGS="$(head -n 1 nexus/settings/live-swap-pub.txt)"
[ -n "$SETTINGS" ] || exit 1
echo "$SETTINGS" > "$RESULTS_DIR/settings.txt"
python nexus/live_speed.py --data-dir "$HOME/specdec-data/long1" --results-dir "$RESULTS_DIR" --part validation \
  --settings "$SETTINGS" \
  --rank 16 --min-tokens 2000 --max-answers 16 --max-hours 8.5 \
  > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.err"
echo "exit $?" > "$RESULTS_DIR/exit.txt"
grep -E "Error|Traceback" -A14 "$HOME/specdec-slurm/$JOB_NAME.err" | tail -c 4000 > "$RESULTS_DIR/messages.txt"
