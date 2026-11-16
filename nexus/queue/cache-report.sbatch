#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --cpus-per-task=4
#SBATCH --mem=8gb
#SBATCH --time=00:15:00
# Looks at how the model cache in the home folder is laid out and how much room is left, before a
# second drafter is downloaded. Reads only; writes a short report. No GPU.
ENV="$HOME/specdec-env"
CODE="$HOME/specdec-code/$JOB_NAME-$SLURM_JOB_ID"
DATA="$HOME/specdec-data/long1"
export HF_HOME="$HOME/hf-cache" TOKENIZERS_PARALLELISM=false
module load Python3/3.12.3 >/dev/null 2>&1 || true
# The code is copied from the runner's folder as committed there (under 1 MB), not cloned again.
mkdir -p "$CODE" && ( set -o pipefail; git -C "$HOME/specdec-runner" archive HEAD dflash nexus tests \
  | tar -x -C "$CODE" --exclude='nexus/results' --exclude='nexus/logs' --exclude='nexus/status' ) || exit 1
source "$ENV/bin/activate" || exit 1
cd "$CODE" || exit 1
export PYTHONPATH="$CODE"
git -C "$HOME/specdec-runner" rev-parse HEAD > "$RESULTS_DIR/code_commit.txt"
{
  echo "== versions"; python -c "import huggingface_hub, transformers; print('huggingface_hub', huggingface_hub.__version__, 'transformers', transformers.__version__)"
  python -c "import hf_xet; print('hf_xet', getattr(hf_xet, '__version__', 'installed'))" 2>&1 | tail -1
  echo "== environment"; env | grep -E "^HF_|^XDG_CACHE|^TRANSFORMERS" | sed 's/=.*hf-cache.*/=<the hf-cache folder>/'
  echo "== hf-cache, first level"; du -sh "$HOME/hf-cache"/* 2>/dev/null | sed "s|$HOME/||"
  echo "== hf-cache/hub, first level"; du -sh "$HOME/hf-cache/hub"/* 2>/dev/null | sed "s|$HOME/||"
  echo "== hf-cache/xet"; du -sh "$HOME/hf-cache/xet" 2>&1 | sed "s|$HOME/||"; ls "$HOME/hf-cache/xet" 2>&1 | head -5
  echo "== project folders"; du -sh "$HOME"/specdec-* "$HOME/hf-cache" 2>/dev/null | sed "s|$HOME/||"
  echo "== whole home folder"; du -sh "$HOME" 2>/dev/null | cut -f1
  echo "== can the node-local disks be written"; for base in /scratch1 /scratch0 /tmp; do d="$base/$USER/probe"; mkdir -p "$d" 2>/dev/null && echo "$base yes $(df -h "$base" | tail -1 | awk '{print $4}') free" || echo "$base no"; done
} > "$RESULTS_DIR/report.txt" 2>&1
echo "exit $?" > "$RESULTS_DIR/exit.txt"
grep -E "Error|Traceback" -A14 "$HOME/specdec-slurm/$JOB_NAME.err" | tail -c 4000 > "$RESULTS_DIR/messages.txt"
