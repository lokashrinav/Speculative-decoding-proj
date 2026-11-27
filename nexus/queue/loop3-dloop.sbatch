#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=09:00:00
# (queued again as loop3-dloop: the first run went out of GPU memory with two views of 16 blocks; now 8 blocks, 1,536 tokens of context)
# D-Loop made here (nexus/train_loop.py --recipe dloop --full): Draft-OPD itself retrained with the paper's two views
# (an ordinary block, and a block with the true beginning filled in; prefix lengths ~ 0.7^l, decays 7, lr 2e-4), so
# that ONE set of weights makes both passes. Saved as ~/specdec-ckpt/drafter-opd-dloop. Then the probe with the
# retrained drafter as the drafter of both passes (nothing / predicted / real states), and the replay of plain and
# branches4 with it against Draft-OPD as it was (folder replay, two runs: the retrained drafter, then the original).
ENV="$HOME/specdec-env"
CODE="/tmp/specdec-code-$USER/$JOB_NAME-$SLURM_JOB_ID"
export HF_HOME="$HOME/hf-cache" TOKENIZERS_PARALLELISM=false
module load Python3/3.12.3 >/dev/null 2>&1 || true
# The code is copied from the runner's folder as committed there (under 1 MB), not cloned again.
mkdir -p "$CODE" && ( set -o pipefail; git -C "$HOME/specdec-runner" archive HEAD dflash nexus tests \
  | tar -x -C "$CODE" --exclude='nexus/results' --exclude='nexus/logs' --exclude='nexus/status' ) || exit 1
source "$ENV/bin/activate" || exit 1
cd "$CODE" || exit 1
export PYTHONPATH="$CODE"
git -C "$HOME/specdec-runner" rev-parse HEAD > "$RESULTS_DIR/code_commit.txt"
( df -h "$HOME" > "$RESULTS_DIR/df.txt" 2>&1; python nexus/train_loop.py --draft opd --data-dir "$HOME/specdec-data/long1" --more-data "$HOME/specdec-data/think1" --results-dir "$RESULTS_DIR" \
    --recipe dloop --full --beta 0.7 --tau 7 --gamma 7 --rate 2e-4 --anchors 128 --group 8 --window 1536 --passes 2 --max-hours 5 --check-answers 12 --save opd-dloop > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.err" \
  && mkdir -p "$RESULTS_DIR/probe" "$RESULTS_DIR/replay" "$RESULTS_DIR/replay-original" \
  && python nexus/restart_probe.py --draft "$HOME/specdec-ckpt/drafter-opd-dloop" --data-dir "$HOME/specdec-data/long1" --results-dir "$RESULTS_DIR/probe" \
    --part validation --max-answers 30 --max-hours 1.2 --predictor opd-m0 > /dev/null 2>> "$HOME/specdec-slurm/$JOB_NAME.err" \
  && python nexus/redraft_long.py --draft "$HOME/specdec-ckpt/drafter-opd-dloop" --data-dir "$HOME/specdec-data/long1" --results-dir "$RESULTS_DIR/replay" \
    --part validation --max-answers 8 --max-hours 1.2 --settings "plain,branches4" > /dev/null 2>> "$HOME/specdec-slurm/$JOB_NAME.err" \
  && python nexus/redraft_long.py --draft opd --data-dir "$HOME/specdec-data/long1" --results-dir "$RESULTS_DIR/replay-original" --part validation --max-answers 8 --max-hours 1.2 \
    --settings "plain,branches4" > /dev/null 2>> "$HOME/specdec-slurm/$JOB_NAME.err" )
echo "exit $?" > "$RESULTS_DIR/exit.txt"
grep -E "Error|Traceback" -A14 "$HOME/specdec-slurm/$JOB_NAME.err" | tail -c 4000 > "$RESULTS_DIR/messages.txt"

