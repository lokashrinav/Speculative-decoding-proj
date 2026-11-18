#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=04:00:00
# Real decoding with the second run as rewritten on Nov 18 (lean code) and with the tree in which one trained layer
# restarts at every node (nexus/live_speed.py): tokens per second on the same 8 validation answers, Draft-OPD. It
# measures the COST side: DDTree with 63 and 95 nodes, the 16 restarts through the whole drafter on DDTree, the
# one-layer tree with the whole head and with the short list, and the whole drafter at every node.
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
python nexus/live_speed.py --draft opd --data-dir "$HOME/specdec-data/long1" --results-dir "$RESULTS_DIR" --part validation \
  --rank 16 --min-tokens 2000 --max-answers 8 --max-hours 3.4 \
  --settings "plain,plain+tree63,plain+tree63_restart4pxt3@opd-m0,plain+btree63f40t2n@~opd-r1,plain+btree63f40t2v16n@~opd-r1,plain+btree63f40t2b8p@opd-m0,plain+tree95,plain+btree95f63t2n@~opd-r1" \
  > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.err"
echo "exit $?" > "$RESULTS_DIR/exit.txt"
grep -E "Error|Traceback" -A14 "$HOME/specdec-slurm/$JOB_NAME.err" | tail -c 4000 > "$RESULTS_DIR/messages.txt"
