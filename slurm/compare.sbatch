#!/bin/bash -l
# CPU paired comparison of variants against a reference (PROTOCOL §7), one family per job.
#   slurm/submit.sh slurm/compare.sbatch TAG=ablation_vs_E0 REF=E0_full_r2 EXPS="E2_baseline E2_crmae" [METRICS="dice"]
#SBATCH --job-name=sts_compare
#SBATCH --account=issai
#SBATCH --partition=ZEN4
#SBATCH --qos=issai-zen4
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=../logs/%x_%j.out
set -euo pipefail
: "${TAG:?}" "${REF:?}" "${EXPS:?}"
METRICS="${METRICS:-dice balanced_accuracy macro_auc_ovr c_pooled_rank_normalised}"
REF_SEEDS_ARG=""; [ -n "${REF_SEEDS:-}" ] && REF_SEEDS_ARG="--ref_seeds $REF_SEEDS"   # e.g. REF_SEEDS=42
module load Python/3.11.5-GCCcore-13.2.0
source ../venv/bin/activate
echo "git $(python -c "import revision_utils as r;print(r.git_state())") tag=$TAG ref=$REF ref_seeds=${REF_SEEDS:-all} exps=$EXPS metrics=$METRICS host=$(hostname) start=$(date)"
# shellcheck disable=SC2086
python evaluate.py compare --run_root ../runs --cache_dir ../cache_48x96x96 --out ../results \
    --tag "$TAG" --ref "$REF" --exps $EXPS --metrics $METRICS $REF_SEEDS_ARG
echo "end $(date)"
