#!/bin/bash
# Step B: validate the tools against published human labels / references (one GPU, ~30 min).
#   1. NLLB-200 translation on FLORES+ devtest -> results/translation_validation.csv
#   2. Llama Guard 3 and keyword judges on JBB judge_comparison -> results/judge_validation.csv
# Submit from the repo folder:  sbatch jobs/validate.sh   (can run at the same time as step A)
#SBATCH -A csds447
#SBATCH -p markov_gpu
#SBATCH -C "gpul40s|gpu4090"
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
set -e
BASE=/mnt/vstor/courses/csds447/team5
source "$BASE/venv/bin/activate"
export HF_HOME=$BASE/hf_cache HF_HUB_OFFLINE=1
umask 007
nvidia-smi -L
python translate.py --validate
python judge.py --validate
