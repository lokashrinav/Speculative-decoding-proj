#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --cpus-per-task=2
#SBATCH --mem=4gb
#SBATCH --time=00:20:00
# Reads only. The home folder has a limit of about 32 GB that the quota command does not show.
# This lists what uses the space (folders this project created by name; anything else by size
# only, without its name), what the model cache and the checkpoint folder contain, and whether
# the cluster offers scratch space outside the home folder.
OUT="$RESULTS_DIR/report.txt"
cd "$HOME" || exit 1
{
  echo "== $(date -u +%FT%TZ) on $(hostname)"
  echo "== whole home folder"; du -sh "$HOME" 2>/dev/null
  echo "== top-level items (project and tool folders by name, anything else by size only)"
  other=0
  for item in .[!.]* *; do
    [ -e "$item" ] || continue
    size=$(du -sk "$item" 2>/dev/null | cut -f1)
    case "$item" in
      specdec-*|hf-cache|.cache|.local|.conda|.config|.nv|.triton|.ssh|.ipython|.jupyter|.npm|.cargo|.keras|.mamba|miniconda3|anaconda3|.vscode-server|.cursor-server)
        echo "$size KB  $item" ;;
      *) other=$((other + 1)); echo "$size KB  (item $other, name not shown)" ;;
    esac
  done | sort -n | tail -40
  echo "== model cache"; du -sk hf-cache/* hf-cache/hub/* hf-cache/datasets/* 2>/dev/null | sort -n | tail -40
  echo "== checkpoints"; du -sk specdec-ckpt/* 2>/dev/null | sort -n
  echo "== saved data"; du -sk specdec-data/* specdec-data/*/* 2>/dev/null | sort -n | tail -20
  echo "== the environment"; du -sk specdec-env 2>/dev/null
  echo "== scratch space on this node and on the network"
  echo "TMPDIR=${TMPDIR:-unset}"
  df -h /tmp /scratch0 /scratch1 /fs/nexus-scratch /fs/class-projects 2>&1
  for place in /scratch0 /scratch1 /tmp "/fs/nexus-scratch/$USER" /fs/nexus-scratch /fs/class-projects; do
    if [ -d "$place" ]; then
      if [ -w "$place" ]; then echo "$place: exists, writable"; else echo "$place: exists, not writable"; fi
    else echo "$place: missing"; fi
  done
  ls -ld "/fs/nexus-scratch/$USER" 2>&1
  echo "== quota commands"; quota -s 2>&1 | head -5
} > "$OUT" 2>&1
echo "exit 0" > "$RESULTS_DIR/exit.txt"
