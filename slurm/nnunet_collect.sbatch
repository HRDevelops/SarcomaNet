#!/bin/bash -l
# E10 nnU-Net: collect final-checkpoint validation predictions into runs/E10_nnunet, then evaluate (CPU).
#   slurm/submit.sh slurm/nnunet_collect.sbatch
#SBATCH --job-name=sts_nncollect
#SBATCH --account=issai
#SBATCH --partition=ZEN4
#SBATCH --qos=issai-zen4
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=../logs/%x_%j.out
set -euo pipefail
source slurm/nnunet_env.sh
echo "git $(python -c "import revision_utils as r;print(r.git_state())") host=$(hostname) start=$(date)"
python nnunet_e10.py collect --results "$nnUNet_results" --cache_dir ../cache_48x96x96 --run_root ../runs --ref_exp E0_full_r2
deactivate; source ../venv/bin/activate
python evaluate.py summary --run_root ../runs --exp E10_nnunet --cache_dir ../cache_48x96x96 --out ../results
echo "end $(date)"
