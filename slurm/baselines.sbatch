#!/bin/bash -l
# Non-imaging baselines (E4 clinical-only = also E6 Cox; E4 subtype-only) + their evaluation. CPU only.
#   slurm/submit.sh slurm/baselines.sbatch
#SBATCH --job-name=sts_clin
#SBATCH --account=issai
#SBATCH --partition=ZEN4
#SBATCH --qos=issai-zen4
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=../logs/%x_%j.out
set -euo pipefail
module load Python/3.11.5-GCCcore-13.2.0
source ../venv/bin/activate
echo "git $(python -c "import revision_utils as r;print(r.git_state())") host=$(hostname) start=$(date)"
python baselines_clinical.py --config configs/E0.json --exp E4_clinical --features clinical
python baselines_clinical.py --config configs/E0.json --exp E4_subtype  --features subtype
for e in E4_clinical E4_subtype; do
  python evaluate.py summary --run_root ../runs --exp $e --cache_dir ../cache_48x96x96 --out ../results
done
echo "end $(date)"
