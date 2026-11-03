#!/bin/bash
# Submit a Slurm script from the LOGIN node only if the working tree is clean,
# and record the commit in ../logs/submissions.tsv (compute nodes have no git).
#   slurm/submit.sh slurm/train_array.sbatch CONFIG=configs/E0.json SEED=42 [--dependency=aftercorr:123]
# KEY=VALUE arguments become --export variables; arguments starting with -- go to sbatch.
set -euo pipefail
script=$1; shift
if [ -n "$(git status --porcelain)" ]; then
  echo "refusing to submit: working tree is dirty"; git status --short; exit 1
fi
exports="ALL"; opts=()
for a in "$@"; do
  case "$a" in --*) opts+=("$a") ;; *) exports="$exports,$a" ;; esac
done
jid=$(sbatch --parsable "${opts[@]}" --export="$exports" "$script")
mkdir -p ../logs
printf "%s\t%s\t%s\t%s\t%s\n" "$(date -Is)" "$jid" "$(git rev-parse HEAD)" "$script" "$*" >> ../logs/submissions.tsv
echo "submitted $jid ($script $*) at $(git rev-parse --short HEAD)"
