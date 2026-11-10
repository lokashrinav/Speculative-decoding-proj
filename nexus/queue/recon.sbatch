#!/bin/bash
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=00:10:00
# Reports what the cluster offers: limits, GPUs, disk, software, network.
R="$RESULTS_DIR/report.txt"
{
  echo "== host"; hostname; date -u
  echo "== gpu"; nvidia-smi
  echo "== qos"; show_qos 2>&1
  echo "== partition"; sinfo -p class -o "%N %G %c %m %l %T" 2>&1 | head -40
  echo "== assoc"; sacctmgr -n show assoc user="$USER" format=account,partition,qos%40,grptres%40,maxtres%40 2>&1
  echo "== cpu/mem"; nproc; free -g | head -2
  echo "== disk"; df -h "$HOME" /scratch0 /scratch1 /tmp 2>&1; quota -s 2>&1 | tail -5
  echo "== class-projects"; ls /fs/class-projects/ 2>&1 | head; ls /fs/class-projects/*/ 2>&1 | head -40
  echo "== software"; python3 --version 2>&1; which conda mamba uv pip3 tmux git 2>&1
  (module avail 2>&1 | grep -i -E "cuda|python|conda|gcc" | head -30)
  echo "== network"
  for u in https://huggingface.co https://pypi.org https://github.com https://download.pytorch.org; do
    printf "%s " "$u"; curl -s -o /dev/null -m 15 -w "%{http_code}\n" "$u" 2>&1
  done
} > "$R" 2>&1
