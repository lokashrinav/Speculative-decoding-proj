#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=03:00:00
# Ablation of the pair scores told how sure the first run was (opd-p2s: all three numbers, one scale per rank): all three numbers but one scale per token, not per rank (a gate, not a shape).
# Trained exactly as opd-p2s (2 passes, 2 hours), then replayed on the same 8 validation answers
# (known there: DDTree 7.56, ptree63@opd-p2 8.20, ptree63@opd-p2s 8.65 on 4 answers so far).
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
( python nexus/train_pairs.py --draft opd --data-dir "$HOME/specdec-data/long1" --results-dir "$RESULTS_DIR" --more-data "$HOME/specdec-data/think1" --sure --sure-scalar --save opd-p2s-g --passes 2 --max-hours 2.0 --check-answers 12 > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.err" \
  && mkdir -p "$RESULTS_DIR/replay" \
  && python nexus/redraft_long.py --draft opd --data-dir "$HOME/specdec-data/long1" --results-dir "$RESULTS_DIR/replay" --part validation --max-answers 8 --max-hours 0.8 \
    --settings "ptree63@opd-p2s-g" > /dev/null 2>> "$HOME/specdec-slurm/$JOB_NAME.err" )
echo "exit $?" > "$RESULTS_DIR/exit.txt"
grep -E "Error|Traceback" -A14 "$HOME/specdec-slurm/$JOB_NAME.err" | tail -c 4000 > "$RESULTS_DIR/messages.txt"

