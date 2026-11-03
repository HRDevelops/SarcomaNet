#!/bin/bash -l
# E10 MONAI seg baseline: one config x one seed, one array task per fold.
#   slurm/submit.sh slurm/seg_baseline_array.sbatch CONFIG=configs/E10_unet.json SEED=42
# Requeue-safe: a finished fold (oof_predictions.csv present) exits immediately;
# an unfinished fold restarts from epoch 1 (no mid-run resume, by design).
#SBATCH --job-name=sts_e10
#SBATCH --account=issai
#SBATCH --partition=H100
#SBATCH --qos=issai-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-4
#SBATCH --requeue
#SBATCH --output=../logs/%x_%A_%a.out
set -euo pipefail
: "${CONFIG:?}" "${SEED:?}"
module load Python/3.11.5-GCCcore-13.2.0
source ../venv/bin/activate
FOLD=${SLURM_ARRAY_TASK_ID}
echo "commit $(python -c "import revision_utils as r;print(r.git_state())") config=$CONFIG seed=$SEED fold=$FOLD host=$(hostname) start=$(date)"
OUT=$(python -c "import json;c=json.load(open('$CONFIG'));print(f\"{c['run_root']}/{c['exp_id']}/seed$SEED/fold$FOLD\")")
if [ -f "$OUT/oof_predictions.csv" ]; then echo "already done: $OUT"; exit 0; fi
python seg_baselines.py --config "$CONFIG" --seed "$SEED" --fold "$FOLD"
echo "end $(date)"
