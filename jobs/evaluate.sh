#!/bin/bash
# Step C: translate every response into English, label it with the judges, build the tables.
# Run after step A has finished:  sbatch jobs/evaluate.sh
# (or chain it:  sbatch --dependency=afterok:<generate job id> jobs/evaluate.sh)
#SBATCH -A csds447
#SBATCH -p markov_gpu
#SBATCH -C "gpul40s|gpu4090"
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%j.out
set -e
BASE=/mnt/vstor/courses/csds447/team5
source "$BASE/venv/bin/activate"
export HF_HOME=$BASE/hf_cache HF_HUB_OFFLINE=1
umask 007
OUT=$BASE/outputs
nvidia-smi -L
python translate.py --outputs $OUT/*__*__*.jsonl
python judge.py --outputs "$OUT/translated/*.jsonl"
python analyze.py
