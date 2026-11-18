#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=04:00:00
# How deep must a trained restart be? (nexus/train_second.py) The drafter's last 2 layers, trained as a reader
# that goes on from the first run: the tokens known at the restart and the states are written into the block. The same
# recipe as one layer (warm-opd-s1.pt) and five layers (warm-opd-s5.pt), so that the depths can be compared: predicted
# states in half of the steps, real ones in 20%, none in 30%. Draft-OPD. Saved as warm-opd-s2.pt.
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
python nexus/train_second.py --draft opd --predictor opd-m0 --data-dir "$HOME/specdec-data/long1" --results-dir "$RESULTS_DIR" --save opd-s2 \
  --layers 2 --rank 256 --reveal --states-in-block --no-states 0.3 --real-share 0.2 --max-hours 2 --check-answers 12 \
  > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.err"
echo "exit $?" > "$RESULTS_DIR/exit.txt"
grep -E "Error|Traceback" -A14 "$HOME/specdec-slurm/$JOB_NAME.err" | tail -c 4000 > "$RESULTS_DIR/messages.txt"
