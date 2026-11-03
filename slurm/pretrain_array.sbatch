#!/bin/bash -l
# Within-fold self-supervised pretraining, one array task per fold.
#   sbatch --export=ALL,CONFIG=configs/E0.json,SEED=42,MODE=crmae slurm/pretrain_array.sbatch
# Requeue-safe: a finished fold (encoder.pth present) exits immediately.
#SBATCH --job-name=sts_pre
#SBATCH --account=issai
#SBATCH --partition=H100
#SBATCH --qos=issai-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-4
#SBATCH --requeue
#SBATCH --output=../logs/%x_%A_%a.out
set -euo pipefail
: "${CONFIG:?}" "${SEED:?}" "${MODE:=crmae}"
module load Python/3.11.5-GCCcore-13.2.0
source ../venv/bin/activate
FOLD=${SLURM_ARRAY_TASK_ID}
echo "commit $(python -c "import revision_utils as r;print(r.git_state())") config=$CONFIG seed=$SEED fold=$FOLD mode=$MODE host=$(hostname) start=$(date)"
ENC=$(python -c "import json,sys;from pretrain_fold import encoder_path;c=json.load(open('$CONFIG'));print(encoder_path(c['encoder_root'],'$MODE',$SEED,$FOLD))")
if [ -f "$ENC" ]; then echo "already done: $ENC"; exit 0; fi
python pretrain_fold.py --config "$CONFIG" --seed "$SEED" --fold "$FOLD" --mode "$MODE"
echo "end $(date)"
