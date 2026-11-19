#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=01:30:00
# Replay of the tree the second drafter scores with the two cheap levers (the 8 validation answers of every-replay-opd-0,
# where the tree with counts held 8.31 tokens a round and the second drafter on it 8.59): a longer chain after every
# node (opd-e1cr4: 4 places asked for; used with 2, 3 and 4), 64 candidates a place (opd-e1ck64), and the second
# drafter trained on 1,000 and on 100 answers (how much text the tree's gain needs).
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
python nexus/redraft_long.py --draft opd --data-dir "$HOME/specdec-data/long1" --results-dir "$RESULTS_DIR" --part validation \
  --max-answers 8 --max-hours 1.2 --count-data "$HOME/specdec-data/think1" \
  --settings "plain,tree63,ctree63,cstree63@opd-e1c,cstree63@opd-e1cr4,cstree63t2@opd-e1cr4,cstree63t3@opd-e1cr4,cstree63@opd-e1ck64,cstree95@opd-e1cr4,cstree63@opd-e1c1000,cstree63@opd-e1c100" \
  > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.err"
echo "exit $?" > "$RESULTS_DIR/exit.txt"
grep -E "Error|Traceback" -A14 "$HOME/specdec-slurm/$JOB_NAME.err" | tail -c 4000 > "$RESULTS_DIR/messages.txt"
