#!/bin/bash -l
# E10 nnU-Net 3d_fullres, 250-epoch trainer, forced splits; nnU-Net fold k = frozen seed-42 fold k.
# Validation is run by nnU-Net at the end of training with checkpoint_final.pth (never checkpoint_best).
#   slurm/submit.sh slurm/nnunet_train_array.sbatch
#SBATCH --job-name=sts_nnunet
#SBATCH --account=issai
#SBATCH --partition=H100
#SBATCH --qos=issai-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --array=0-4
#SBATCH --requeue
#SBATCH --output=../logs/%x_%A_%a.out
set -euo pipefail
source slurm/nnunet_env.sh
FOLD=${SLURM_ARRAY_TASK_ID}
FD="$nnUNet_results/Dataset501_STSrev/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres/fold_$FOLD"
echo "git $(python -c "import revision_utils as r;print(r.git_state())") fold=$FOLD host=$(hostname) start=$(date)"
if [ -f "$FD/checkpoint_final.pth" ] && [ -f "$FD/validation/summary.json" ]; then echo "already done: $FD"; exit 0; fi
CONT=""; [ -f "$FD/checkpoint_latest.pth" ] && CONT="--c"      # requeue: continue the same run
nnUNetv2_train 501 3d_fullres "$FOLD" -tr nnUNetTrainer_250epochs $CONT
echo "end $(date)"
