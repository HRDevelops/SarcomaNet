#!/bin/bash -l
# CPU evaluation of one experiment from its OOF files (PROTOCOL §7).
#   slurm/submit.sh slurm/evaluate.sbatch EXP=E0_full
#SBATCH --job-name=sts_eval
#SBATCH --account=issai
#SBATCH --partition=ZEN4
#SBATCH --qos=issai-zen4
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=../logs/%x_%j.out
set -euo pipefail
: "${EXP:?}"
module load Python/3.11.5-GCCcore-13.2.0
source ../venv/bin/activate
echo "git $(python -c "import revision_utils as r;print(r.git_state())") exp=$EXP host=$(hostname) start=$(date)"
python evaluate.py summary --run_root ../runs --exp "$EXP" --cache_dir ../cache_48x96x96 --out ../results
echo "end $(date)"
