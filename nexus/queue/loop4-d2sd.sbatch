#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=09:00:00
# (loop4: the same as loop1-d2sd, which died at 00:40 UTC Nov 27 when the home folder filled during its first save.)
# D2SD's second drafter, made here (nexus/train_loop.py --recipe d2sd --full): a SEPARATE whole drafter, started from
# Draft-OPD, all weights trained to fill the end of a block when the beginning holds the true tokens (prefix lengths
# ~ 0.85^l, loss decaying from the first masked place with tau 7, lr 1e-4), on long1's training answers and think1,
# two passes or 5 hours. The first drafter stays Draft-OPD. Saved as ~/specdec-ckpt/drafter-opd-d2sd (1.1 GB).
# Then: (1) the probe of what the second drafter is missing, restarts by the TRAINED drafter with nothing, the
# predicted states (opd-m0) and the target's real states (folder probe): does a trained second pass still gain from
# the states? (2) replay on 8 validation answers (folder replay): plain, branches4 (the unchanged drafter reused as
# the second drafter, D2SD's own control) and branches4@drafter-opd-d2sd (the trained one), tokens per round.
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
    --recipe d2sd --full --beta 0.85 --tau 7 --rate 1e-4 --anchors 128 --group 16 --passes 2 --max-hours 5 --check-answers 12 --save opd-d2sd > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.err" \
  && mkdir -p "$RESULTS_DIR/probe" "$RESULTS_DIR/replay" \
  && python nexus/restart_probe.py --draft opd --data-dir "$HOME/specdec-data/long1" --results-dir "$RESULTS_DIR/probe" --part validation --max-answers 30 --max-hours 1.2 \
    --predictor opd-m0 --second-draft drafter-opd-d2sd > /dev/null 2>> "$HOME/specdec-slurm/$JOB_NAME.err" \
  && python nexus/redraft_long.py --draft opd --data-dir "$HOME/specdec-data/long1" --results-dir "$RESULTS_DIR/replay" --part validation --max-answers 8 --max-hours 1.5 \
    --settings "plain,branches4,branches4@drafter-opd-d2sd" > /dev/null 2>> "$HOME/specdec-slurm/$JOB_NAME.err" )
echo "exit $?" > "$RESULTS_DIR/exit.txt"
grep -E "Error|Traceback" -A14 "$HOME/specdec-slurm/$JOB_NAME.err" | tail -c 4000 > "$RESULTS_DIR/messages.txt"

