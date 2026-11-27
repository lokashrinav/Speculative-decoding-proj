#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=02:00:00
# Can the drafter's alternatives be read out early (nexus/early_readout.py)? The readout after every layer of Draft-OPD
# with the final norm and head, on the first runs of 24 validation answers, an anchor every 16 tokens: the right token
# in the top 1 / 2 / 8 by layer and by place, and whether the four likely breaks chosen from an early readout give
# branches whose whole prefix is right. The first of the two checks before any "branching inside the drafter" is built.
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
python nexus/early_readout.py --draft opd --data-dir "$HOME/specdec-data/long1" --results-dir "$RESULTS_DIR" --part validation --max-answers 24 --every 16 --max-anchors 256 --max-hours 1.7 \
  > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.err"
echo "exit $?" > "$RESULTS_DIR/exit.txt"
grep -E "Error|Traceback" -A14 "$HOME/specdec-slurm/$JOB_NAME.err" | tail -c 4000 > "$RESULTS_DIR/messages.txt"

