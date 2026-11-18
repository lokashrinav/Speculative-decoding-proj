#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=04:00:00
# How deep must a trained restart be? (nexus/train_second.py) The drafter's last 3 layers, trained as a reader that
# goes on from the first run, on the 366 answers of long1: the recipe of two layers (warm-opd-s2.pt). It was queued
# as train-depth3-opd-0, whose place was to go to the new text (nexus/gen_text.py). Saved as warm-opd-s3.pt.
# (train-depth3-opd-0 started before that change and trains the three layers itself, so THIS job writes text:
# nexus/settings/instead.txt.)
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
python nexus/train_second.py --draft opd --predictor opd-m0 --data-dir "$HOME/specdec-data/long1" --results-dir "$RESULTS_DIR" --save opd-s3 \
  --layers 3 --rank 128 --reveal --states-in-block --no-states 0.3 --real-share 0.2 --max-hours 2 --check-answers 12 \
  > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.err"
echo "exit $?" > "$RESULTS_DIR/exit.txt"
grep -E "Error|Traceback" -A14 "$HOME/specdec-slurm/$JOB_NAME.err" | tail -c 4000 > "$RESULTS_DIR/messages.txt"
