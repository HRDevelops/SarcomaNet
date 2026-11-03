#!/bin/bash
# Paired-comparison families for batch 2 + E4, as fixed in PROTOCOL §11 (2026-10-29). Run from the repo root on the login node.
set -euo pipefail
S=slurm/submit.sh; C=slurm/compare.sbatch; R=E0_full_r2
GRADE_SURV="METRICS=balanced_accuracy macro_auc_ovr c_pooled_rank_normalised"
# primary
$S $C TAG=ablation_vs_E0 REF=$R "EXPS=E2_baseline E2_crmae E2_apd E2_ms"
$S $C TAG=E1_nopre_vs_E0 REF=$R EXPS=E1_nopre
$S $C TAG=E1b_spatial_vs_E0 REF=$R EXPS=E1b_spatial
$S $C TAG=E3_imgonly_vs_E0 REF=$R EXPS=E3_imgonly
$S $C TAG=E9_seg_vs_E0 REF=$R EXPS=E9_seg METRICS=dice
$S $C TAG=E9_cls_vs_E0 REF=$R EXPS=E9_cls "METRICS=balanced_accuracy macro_auc_ovr"
$S $C TAG=E9_surv_vs_E0 REF=$R EXPS=E9_surv METRICS=c_pooled_rank_normalised
$S $C TAG=E4_vs_E0 REF=$R "EXPS=E4_clinical E4_subtype" "$GRADE_SURV"
# secondary, exploratory
$S $C TAG=components_vs_E2_baseline_EXPLORATORY REF=E2_baseline "EXPS=E2_crmae E2_apd E2_ms"
