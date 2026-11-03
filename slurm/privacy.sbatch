#!/bin/bash -l
# E8 privacy attacks + DP-SGD head (CPU), from saved run outputs.
#   slurm/submit.sh slurm/privacy.sbatch
#SBATCH --job-name=sts_privacy
#SBATCH --account=issai
#SBATCH --partition=ZEN4
#SBATCH --qos=issai-zen4
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=../logs/%x_%j.out
set -euo pipefail
module load Python/3.11.5-GCCcore-13.2.0
source ../venv/bin/activate
echo "git $(python -c "import revision_utils as r;print(r.git_state())") host=$(hostname) start=$(date)"
python privacy_attacks.py --run_root ../runs --out ../results --exps E0_full_r2 E1_nopre E2_baseline E2_apd \
    --pair E2_baseline E2_apd --dp_exp E0_full_r2
echo "end $(date)"
