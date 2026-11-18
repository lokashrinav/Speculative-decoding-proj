#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=06:00:00
# (Queued before the script's default became --loss tokens: these jobs train on the tokens the drafter
# writes with the predicted vectors, plus a tenth of the vectors' error. train-restart-*-1 match the vectors only.)
# The same as train-restart-pub with the better drafter (Draft-OPD): a small predictor of the target's
# states a restart does not have yet, and the restart measured with it. Saved as restart-opd-v0.pt.
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
# Third predictor, the small change (Nov 18): as the second one (taught at real breaks, both kinds of restart),
# and it is also told the token after each token; after the last one that is the token the restart starts from.
# Compare tree-replay-opd-4 (the second predictor, opd-b0) and restart-replay-pub-3 (this plus two layers
# in which the missing tokens look back, opd-s0).
python nexus/train_restart.py --draft opd --data-dir "$HOME/specdec-data/long1" --results-dir "$RESULTS_DIR" --save opd-c0 \
  --coming --vector-weight 2 --passes 2 --max-hours 3 --check-answers 12 \
  > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.err"
echo "exit $?" > "$RESULTS_DIR/exit.txt"
