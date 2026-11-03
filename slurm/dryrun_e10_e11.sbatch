#!/bin/bash -l
# GPU dry run for E10 MONAI nets and the E11 inner pipeline (1 epoch, 1 batch; outputs under *_DRYRUN).
#SBATCH --job-name=sts_dry2
#SBATCH --account=issai
#SBATCH --partition=H100
#SBATCH --qos=issai-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=../logs/%x_%j.out
set -euo pipefail
module load Python/3.11.5-GCCcore-13.2.0
source ../venv/bin/activate
echo "git $(python -c "import revision_utils as r;print(r.git_state())") host=$(hostname) start=$(date)"
python -m pytest -q tests
for n in unet segresnet swinunetr; do
  [ -f ../runs/E10_${n}_DRYRUN/seed42/fold0/oof_predictions.csv ] || python seg_baselines.py --config configs/E10_$n.json --seed 42 --fold 0 --epochs 2 --max_batches 2
  python -c "import json;m=json.load(open('../runs/E10_${n}_DRYRUN/seed42/fold0/run_meta.json'));print('$n params',m['n_params'])"
done
python pretrain_fold.py --config configs/E11/o0_E50.json --seed 42 --fold 0 --mode crmae --epochs 1 --max_batches 1
python train.py --config configs/E11/o0_E50.json --seed 42 --fold 0 --epochs 1 --max_batches 1
python -c "
import json,pandas as pd
m=json.load(open('../runs/E11_inner_o0_E50_DRYRUN/seed42/fold0/run_meta.json'));o=pd.read_csv('../runs/E11_inner_o0_E50_DRYRUN/seed42/fold0/oof_predictions.csv')
print('E11 inner dry: n_train',m['n_train'],'n_val',len(o),'encoder tensors',m['encoder_tensors_loaded'],'splits sha',m['splits_sha256'][:8])
outer=json.load(open('splits_inner/outer0.json'))['outer']['val_ids']; assert not set(o.pid)&set(outer); assert not set(m['val_ids'])&set(outer); print('no outer-val patients in inner run: OK')"
echo "end $(date)"
