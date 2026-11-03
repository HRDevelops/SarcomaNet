#!/bin/bash -l
# E7 radiomics (CPU): feature extraction in venv_radiomics, then models on the frozen splits, then summary.
#   slurm/submit.sh slurm/radiomics.sbatch
#SBATCH --job-name=sts_radiomics
#SBATCH --account=issai
#SBATCH --partition=ZEN4
#SBATCH --qos=issai-zen4
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=../logs/%x_%j.out
set -euo pipefail
module load Python/3.11.5-GCCcore-13.2.0
FEAT=../results/E7_radiomics_features.csv
echo "git $(../venv/bin/python -c "import revision_utils as r;print(r.git_state())") host=$(hostname) start=$(date)"
[ -e "$FEAT" ] || ../venv_radiomics/bin/python radiomics_extract.py --cache_dir ../cache_48x96x96 --out "$FEAT"
source ../venv/bin/activate
python baselines_radiomics.py --config configs/E0.json --features "$FEAT" --exp E7_radiomics
python evaluate.py summary --run_root ../runs --exp E7_radiomics --cache_dir ../cache_48x96x96 --out ../results
python evaluate.py compare --run_root ../runs --cache_dir ../cache_48x96x96 --out ../results --tag E7_vs_E0 \
    --ref E0_full_r2 --exps E7_radiomics --metrics balanced_accuracy macro_auc_ovr c_pooled_rank_normalised
echo "end $(date)"
