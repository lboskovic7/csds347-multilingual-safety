#!/bin/bash
# Step A: generate responses. One array task per model in models.txt, each on its own GPU.
# Every model runs: vanilla + Self-Reminder on MultiJail, XSTest and JBB (harmful + benign),
# then Backtranslation on a MultiJail subset (BT_LIMIT prompts per language), XSTest and JBB.
# Submit from the repo folder:  sbatch jobs/generate.sh
# Optional: LIMIT=100 sbatch jobs/generate.sh   (MultiJail prompts per language; default all 315)
# Optional: MODELS_FILE=models_demo3.txt sbatch --array=1 jobs/generate.sh   (another model list)
#           DEFENSES="none" BT_LIMIT=0 ...   (vanilla only, no Backtranslation)   BS=96 (batch size)
#SBATCH -A csds447
#SBATCH -p markov_gpu
#SBATCH -C "gpul40s|gpu4090"
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 08:00:00
#SBATCH --array=1-3%3
#SBATCH -o logs/%x-%A_%a.out
set -e
BASE=/mnt/vstor/courses/csds447/team5
source "$BASE/venv/bin/activate"
export HF_HOME=$BASE/hf_cache HF_HUB_OFFLINE=1
umask 007
OUT=$BASE/outputs
MODEL=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ${MODELS_FILE:-models.txt})
M=${MODEL##*/}
MJ_LIMIT=${LIMIT:+--limit $LIMIT}
BT=${BT_LIMIT:-50}
nvidia-smi -L
# batch size by GPU memory: 48 on an L40S (48 GB), 16 on an RTX 4090 (24 GB)
MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ -z "$BS" ]; then if [ "$MEM" -gt 40000 ]; then BS=48; else BS=16; fi; fi
echo "GPU memory ${MEM} MiB -> batch size $BS"

for DEF in ${DEFENSES:-none self_reminder}; do
  python run_baseline.py --model $MODEL --defense $DEF --dataset multijail $MJ_LIMIT --out_dir $OUT --batch_size $BS
  python run_baseline.py --model $MODEL --defense $DEF --dataset xstest --out_dir $OUT --batch_size $BS
  python run_baseline.py --model $MODEL --defense $DEF --dataset jbb --split both --out_dir $OUT --batch_size $BS
done
if [ "$BT" -gt 0 ]; then
python run_baseline.py --model $MODEL --defense backtranslation --dataset multijail --limit $BT \
  --from_responses $OUT/multijail__none__$M.jsonl --out_dir $OUT --batch_size $BS
python run_baseline.py --model $MODEL --defense backtranslation --dataset xstest \
  --from_responses $OUT/xstest__none__$M.jsonl --out_dir $OUT --batch_size $BS
python run_baseline.py --model $MODEL --defense backtranslation --dataset jbb --split both \
  --from_responses $OUT/jbb-both__none__$M.jsonl --out_dir $OUT --batch_size $BS
fi
echo "Done: $MODEL"
