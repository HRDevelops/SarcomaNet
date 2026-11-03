#!/bin/bash -l
# Phase 2.6 dry run: E0 pipeline on seed 42 / fold 0, 2 pretrain + 3 train epochs.
# Outputs go to *_DRYRUN paths only; real runs can never pick them up.
#SBATCH --job-name=sts_dryrun
#SBATCH --account=issai
#SBATCH --partition=H100
#SBATCH --qos=issai-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=../logs/%x_%j.out
set -euo pipefail
module load Python/3.11.5-GCCcore-13.2.0
source ../venv/bin/activate
echo "commit $(python -c "import revision_utils as r;print(r.git_state())")  host $(hostname)  start $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -m pytest tests -q
python pretrain_fold.py --config configs/E0.json --seed 42 --fold 0 --mode crmae --epochs 2
python train.py        --config configs/E0.json --seed 42 --fold 0 --epochs 3
nvidia-smi --query-gpu=memory.used --format=csv
echo "end $(date)"
