#!/bin/bash
# Batch job: runs every model in models.txt, one GPU each, at most 2 at a time.
# Submit with: sbatch run_models.sh
#SBATCH -A csds447
#SBATCH -p markov_gpu
#SBATCH -C "gpu4090|gpul40s"       # 7-9B models fit on either; starts on whichever is free
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH --array=1-3%2              # 3 models (lines in models.txt), max 2 running at once
#SBATCH -o logs/%x-%A_%a.out

BASE=/mnt/vstor/courses/csds447/team5
source "$BASE/venv/bin/activate"
export HF_HOME=$BASE/hf_cache

MODEL=$(sed -n "${SLURM_ARRAY_TASK_ID}p" models.txt)
nvidia-smi -L
python run_baseline.py --model "$MODEL" --out_dir "$BASE/outputs" "$@"
