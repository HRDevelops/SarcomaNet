#!/bin/bash -l
# E10 nnU-Net: dataset conversion + plan/preprocess (CPU) + frozen splits_final.json.
#   slurm/submit.sh slurm/nnunet_prep.sbatch
#SBATCH --job-name=sts_nnprep
#SBATCH --account=issai
#SBATCH --partition=ZEN4
#SBATCH --qos=issai-zen4
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=../logs/%x_%j.out
set -euo pipefail
source slurm/nnunet_env.sh
echo "git $(python -c "import revision_utils as r;print(r.git_state())") host=$(hostname) start=$(date)"
[ -d "$nnUNet_raw/Dataset501_STSrev" ] || python nnunet_e10.py prepare --cache_dir ../cache_48x96x96 --raw "$nnUNet_raw"
nnUNetv2_plan_and_preprocess -d 501 -c 3d_fullres -np 8 --verify_dataset_integrity
python nnunet_e10.py splits --preprocessed "$nnUNet_preprocessed"
python -c "import json;p=json.load(open('$nnUNet_preprocessed/Dataset501_STSrev/nnUNetPlans.json'))['configurations']['3d_fullres'];print('3d_fullres plan:',{k:p[k] for k in ('patch_size','batch_size','spacing','median_image_size_in_voxels') if k in p})"
echo "end $(date)"
