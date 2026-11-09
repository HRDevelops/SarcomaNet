#!/bin/bash -l
# E12 pilot (PROTOCOL §11, 2026-11-02): nnU-Net encoder features -> grade/OS heads -> evaluation and comparison.
#   slurm/submit.sh slurm/e12.sbatch            (full)
#   slurm/submit.sh slurm/e12.sbatch DRY=1      (fold 0, 3 patients, feature extraction only)
#SBATCH --job-name=sts_e12
#SBATCH --account=issai
#SBATCH --partition=H100
#SBATCH --qos=issai-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=../logs/%x_%j.out
set -euo pipefail
source slurm/nnunet_env.sh
echo "git $(python -c "import revision_utils as r;print(r.git_state())") dry=${DRY:-0} host=$(hostname) start=$(date)"
FEAT=../results/E12_nnunet_features.npz
python e12_nnunet_features.py --results "$nnUNet_results" --raw "$nnUNet_raw" --cache_dir ../cache_48x96x96 \
    --extra_dir ../nnunet/E12_extra --out "$FEAT" ${DRY:+--dry}
[ -n "${DRY:-}" ] && { echo "dry run done $(date)"; exit 0; }
deactivate; source ../venv/bin/activate
python e12_heads.py --config configs/E0.json --features "$FEAT" --seg_exp E10_nnunet --exp E12_nnunet_feat
python evaluate.py summary --run_root ../runs --exp E12_nnunet_feat --cache_dir ../cache_48x96x96 --out ../results
python evaluate.py compare --run_root ../runs --cache_dir ../cache_48x96x96 --out ../results --tag E12_vs_E0s42 \
    --ref E0_full_r2 --exps E12_nnunet_feat --ref_seeds 42
echo "end $(date)"
