#!/bin/bash -l
# E11 inner runs: task t -> outer fold k = t / 4, inner split j = t % 4.
# Each task: within-inner-fold CrMAE pretraining, then E0 training for E = 50, 100, 150, 200.
#   slurm/submit.sh slurm/e11_inner_array.sbatch
# Requeue-safe: finished pieces (encoder.pth / oof_predictions.csv) are skipped.
#SBATCH --job-name=sts_e11in
#SBATCH --account=issai
#SBATCH --partition=H100
#SBATCH --qos=issai-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-19
#SBATCH --requeue
#SBATCH --output=../logs/%x_%A_%a.out
set -euo pipefail
module load Python/3.11.5-GCCcore-13.2.0
source ../venv/bin/activate
T=${SLURM_ARRAY_TASK_ID}; K=$((T / 4)); J=$((T % 4)); SEED=42
echo "commit $(python -c "import revision_utils as r;print(r.git_state())") outer=$K inner=$J host=$(hostname) start=$(date)"
C0=configs/E11/o${K}_E50.json
ENC=$(python -c "import json;from pretrain_fold import encoder_path;c=json.load(open('$C0'));print(encoder_path(c['encoder_root'],'crmae',$SEED,$J))")
[ -f "$ENC" ] || python pretrain_fold.py --config "$C0" --seed $SEED --fold $J --mode crmae
for E in 50 100 150 200; do
  CFG=configs/E11/o${K}_E${E}.json
  OUT=$(python -c "import json;c=json.load(open('$CFG'));print(f\"{c['run_root']}/{c['exp_id']}/seed$SEED/fold$J\")")
  if [ -f "$OUT/oof_predictions.csv" ]; then echo "already done: $OUT"; continue; fi
  python train.py --config "$CFG" --seed $SEED --fold $J
done
echo "end $(date)"
