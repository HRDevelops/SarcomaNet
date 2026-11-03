#!/bin/bash -l
# E11 outer refits: fold k uses configs/E11/outer_fold<k>.json (epochs = chosen E*), outer seed 42.
#   slurm/submit.sh slurm/e11_refit_array.sbatch
#SBATCH --job-name=sts_e11out
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
module load Python/3.11.5-GCCcore-13.2.0
source ../venv/bin/activate
K=${SLURM_ARRAY_TASK_ID}; CFG=configs/E11/outer_fold${K}.json
echo "commit $(python -c "import revision_utils as r;print(r.git_state())") config=$CFG host=$(hostname) start=$(date)"
OUT=$(python -c "import json;c=json.load(open('$CFG'));print(f\"{c['run_root']}/{c['exp_id']}/seed42/fold$K\")")
if [ -f "$OUT/oof_predictions.csv" ]; then echo "already done: $OUT"; exit 0; fi
python train.py --config "$CFG" --seed 42 --fold $K
echo "end $(date)"
