#!/bin/bash
# Demo 3, our approach: train the cross-lingual self-alignment adapter on Qwen2.5-3B, then run the
# adapted model on the same data and settings as the baselines (MultiJail, XSTest, JBB).
# Submit from the repo folder:  sbatch jobs/self_align.sh
# Optional: MODEL=Qwen/Qwen2.5-1.5B-Instruct sbatch jobs/self_align.sh
#SBATCH -A csds447
#SBATCH -p markov_gpu
#SBATCH -C gpul40s
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
MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}
M=${MODEL##*/}
ADAPTER=$BASE/adapters/${M}-selfalign
BS=${BS:-48}
nvidia-smi -L

python self_align.py --model $MODEL --out_dir $ADAPTER
for D in "multijail" "xstest" "jbb --split both"; do
  python run_baseline.py --model $MODEL --defense self_align --adapter $ADAPTER --dataset $D --out_dir $OUT --batch_size $BS
done
echo "Done: $MODEL with $ADAPTER"
