#!/bin/bash -l
# Build the 48x96x96 cache from the TCIA DICOM tree (CPU only).
#SBATCH --job-name=sts_preproc
#SBATCH --account=issai
#SBATCH --partition=ZEN4
#SBATCH --qos=issai-zen4
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=../logs/%x_%j.out
set -euo pipefail
module load Python/3.11.5-GCCcore-13.2.0
source ../venv/bin/activate
echo "commit $(python -c "import revision_utils as r;print(r.git_state())")  host $(hostname)  start $(date)"
python preprocess.py \
  --data_root "<DATA_DIR>/Soft-tissue-Sarcoma" \
  --cache_dir ../cache_48x96x96
echo "end $(date)"
