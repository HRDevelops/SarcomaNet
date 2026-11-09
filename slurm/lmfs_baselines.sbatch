#!/bin/bash -l
# D-1 LMFS sensitivity (CPU): clinical and radiomics Cox on the LMFS endpoint, same folds, then summaries.
#   slurm/submit.sh slurm/lmfs_baselines.sbatch
#SBATCH --job-name=sts_lmfs_base
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
FEAT=../results/E7_radiomics_features.csv
[ -e "$FEAT" ] || { echo "missing $FEAT (run E7 first)"; exit 1; }
echo "git $(python -c "import revision_utils as r;print(r.git_state())") host=$(hostname) start=$(date)"
python baselines_clinical.py --config configs/E0_lmfs.json --exp E4_clinical_lmfs --features clinical
python baselines_radiomics.py --config configs/E0_lmfs.json --features "$FEAT" --exp E7_radiomics_lmfs
for e in E4_clinical_lmfs E7_radiomics_lmfs; do
  python evaluate.py summary --run_root ../runs --exp $e --cache_dir ../cache_48x96x96 --out ../results
done
echo "end $(date)"
