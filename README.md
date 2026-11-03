# SarcomaNet: code for the revised evaluation

Code for *SarcomaNet* (manuscript ARRAY-D-26-03543, revised version): a multi-task 3D network for soft tissue
sarcoma segmentation, FNCLCC grading and overall-survival risk from T2FS MRI, FDG-PET and CT, evaluated under a
leak-free, pre-specified protocol (`docs/PROTOCOL.md`).

## Data

The data are public and are **not** included here. Download from The Cancer Imaging Archive (TCIA):

- the *Soft-tissue-Sarcoma* collection (DICOM images and RT-STRUCT contours);
- `INFOclinical_STS.xlsx` (clinical data).

Put both under one folder and replace `<DATA_DIR>` in `configs/*.json` and `slurm/preprocess.sbatch` with that path.

## Environment

- Python 3.11, PyTorch 2.6.0 (CUDA 12.4): `pip install -r requirements.lock`
- Radiomics baseline only: separate environment, `requirements_radiomics.lock` (PyRadiomics 3.1.0 needs numpy < 2)
- nnU-Net baseline only: separate environment with `nnunetv2==2.6.2` and the same torch version

## Pipeline

All commands run from the repository root. Outputs go to `../runs`, `../results` and `../cache_48x96x96`
(set in the configs).

1. **Preprocess** DICOM to a 48 × 96 × 96 cache (T2FS, PET, CT, tumour mask):
   `python preprocess.py --data_root <DATA_DIR>/Soft-tissue-Sarcoma --cache_dir ../cache_48x96x96`
2. **Splits.** `splits.json` holds the frozen 5-fold × 3-seed splits used in the paper (stratified on event ×
   Grade 3). `make_splits.py` regenerates them; `make_inner_splits.py` makes the inner splits for nested CV.
3. **Within-fold pretraining** (CrMAE or spatial MAE), on that fold's training patients only:
   `python pretrain_fold.py --config configs/E0.json --seed 42 --fold 0 --mode crmae`
4. **Training** (fixed 150 epochs; the last-epoch model is evaluated for all tasks):
   `python train.py --config configs/E0.json --seed 42 --fold 0`
5. **Evaluation** (pooled out-of-fold metrics with patient-bootstrap CIs; paired comparisons with Holm correction):
   `python evaluate.py summary --run_root ../runs --exp E0_full_r2 --cache_dir ../cache_48x96x96 --out ../results`
   `python evaluate.py compare --run_root ../runs --cache_dir ../cache_48x96x96 --out ../results --tag <name> --ref E0_full_r2 --exps <exp ...>`

## Experiments

| ID | Config / script | What it is |
|---|---|---|
| E0 | `configs/E0.json` | Full model (within-fold CrMAE, APD, multi-scale head, clinical vector) |
| E1, E1b | `configs/E1_nopre.json`, `configs/E1b_spatial.json` | No pretraining; spatial-patch MAE |
| E2 | `configs/E2_*.json` | Ablation: baseline, +CrMAE, +APD, +multi-scale |
| E3 | `configs/E3_imgonly.json` | Imaging only |
| E4 | `baselines_clinical.py` | Clinical-only and subtype-only models |
| E4b | `configs/E4b_clin_v1.json` | Original 6-d clinical vector |
| E7 | `radiomics_extract.py`, `baselines_radiomics.py` | Radiomics + Cox baseline |
| E8 | `privacy_attacks.py` | Cross-modal retrieval, membership inference, DP-SGD head |
| E9 | `configs/E9_*.json` | Single-task models |
| E10 | `seg_baselines.py`, `configs/E10_*.json`, `nnunet_e10.py` | MONAI U-Net / SegResNet / SwinUNETR, nnU-Net |
| E11 | `e11_nested.py`, `configs/E11/` | Number of epochs chosen by nested CV |

`slurm/` holds the Slurm scripts used on the HPC cluster (edit the account, partition and module lines for
your cluster). `tests/` holds known-answer tests for the metrics and losses: `python -m pytest tests`.

## License

[To be chosen by the authors.]

## Citation

[Add the published reference.]
