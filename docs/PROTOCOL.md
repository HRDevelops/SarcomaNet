# PROTOCOL: evaluation protocol for the Array revision (ARRAY-D-26-03543)

**Status: APPROVED and FROZEN, 2026-10-29** (Husnain: "I approve PROTOCOL.md D-1 to D-15 as drafted"). Written and approved before any new experiment; no new result had been seen.
Every item below marked [PENDING] in the draft is now **[APPROVED 2026-10-29]**. Any later change is logged in §11 with its date and reason, and the paper reports deviations.

Evidence base for this draft: `AUDIT.md` (claim-to-code mapping, bug list B1–B17) and `PHASE0_REPORT.md`.

---

## 0. Decisions requiring approval

| ID | Decision | Recommended default | Status |
|---|---|---|---|
| D-1 | Survival endpoint | **Overall survival (OS):** event = Status "D"; time = diagnosis → last follow-up for **every** patient. Sensitivity analysis: lung-metastasis-free survival (LMFS) using sheet 2 `LungMets` and the outcome time for lung-metastasis patients, matching Vallières for the radiomics baseline (E7) | **[APPROVED 2026-10-29]** |
| D-2 | Patients with grade "N/A" (3) | **Exclude from the grading task** (loss masked, not in grade metrics or grade stratification). Grade analysis set n = 48 (Low 5 / Int 15 / High 28). Kept for segmentation, survival and pretraining | **[APPROVED 2026-10-29]** |
| D-3 | Mask failures STS_024, STS_039, STS_045 | **Exclude from segmentation loss and segmentation metrics** (`has_mask = 0`). Segmentation analysis set n = 48. Kept for grade, survival and pretraining | **[APPROVED 2026-10-29]** |
| D-4 | Code base | **May 9 lab-PC code** (`D:\sarcoma\sarcomatrine_2_0\sarcomatrine_2_0\`) as the engineering base, in a new git repo, branch `array-revision`. The **scientific spec is the submitted paper's model** (§2), not the May 9 post-hoc tuning | **[APPROVED 2026-10-29]** |
| D-5 | Checkpoint rule | **Last epoch of a fixed training budget**, all three tasks from that one checkpoint. No early stopping, no validation-based selection of any kind | **[APPROVED 2026-10-29]** |
| D-6 | Training budget | **E = 150 epochs** for every supervised variant (5 warmup + cosine to 1e-6), CrMAE 60 epochs. Fixed now, before any new run | **[APPROVED 2026-10-29]** |
| D-7 | Repeats | **5-fold CV × 3 seeds** (seed changes fold shuffle **and** initialisation): seeds 42, 1337, 2026 | **[APPROVED 2026-10-29]** |
| D-8 | Fold stratification | Stratify on **event status (OS) × high-grade (G3 vs not)**; N/A-grade patients counted as "not G3" for splitting only | **[APPROVED 2026-10-29]** |
| D-9 | Clinical vector for the main model | **Revised c:** age/100, sex, lower-limb site, **one-hot MSKCC (7)** → 10-d. **Drop MRI-to-PET interval.** Original 6-d c kept only as a sensitivity analysis | **[APPROVED 2026-10-29]** |
| D-10 | Survival loss fix (bug B8) | Cox partial likelihood computed over the **whole training set each step**: a per-epoch memory bank of detached risk scores supplies the risk set, the current batch's risk scores carry the gradient, and ties use Breslow. Fallback if unstable: May 9 batch-local version with batch 8 (documented as a limitation). **→ Fallback applied 2026-10-29, see §11** | **[APPROVED 2026-10-29]** |
| D-11 | Spatial alignment (bug B10) | **No new registration** (out of scope). State plainly that channels are independently resized, not co-registered. Report which reference series each mask was rasterised on, plus a sensitivity analysis restricted to MRI-referenced masks | **[APPROVED 2026-10-29]** |
| D-12 | Primary survival statistic | **Pooled out-of-fold C-index after within-fold rank normalisation of risk**. Secondary: raw pooled, Uno's C, per-fold with event counts | **[APPROVED 2026-10-29]** |
| D-13 | Privacy threat model and attacks | As in §7 (cross-view retrieval + membership inference; DP-SGD scoped to the grade head) | **[APPROVED 2026-10-29]** |
| D-14 | Compute budget | ≈ **100 H100 GPU-hours** (§9), in batches of ≤ 10 jobs, each batch confirmed before submission | **[APPROVED 2026-10-29]** |
| D-15 | External/auxiliary pretraining data | **None.** Within-fold, same cohort only | **[APPROVED 2026-10-29]** |

---

## 1. Analysis sets (assuming D-1 to D-3)

| Set | Patients | Used for |
|---|---|---|
| ALL | 51 | Fold assignment, CrMAE pretraining (training folds only), survival |
| SEG | 48 (ALL minus STS_024/039/045) | Segmentation loss and metrics |
| GRADE | 48 (ALL minus the 3 "N/A" grades) | Grading loss and metrics |
| SURV | 51, 15 OS events | Survival loss and metrics |

Checked 2026-10-29 (aggregate counts only): the 3 N/A-grade patients and the 3 mask failures are **disjoint**, so SEG and GRADE are both n = 48 but are different sets. OS events by grade: Low 0, Intermediate 6, High 8, N/A 1. The N/A-grade patients contribute 1 event to SURV; the mask failures contribute none.

A patient missing a label still contributes to the other tasks; their loss for the missing task is masked to zero. Per-patient flags `has_mask`, `has_grade` are written to `splits.json`. The IDs of the 3 N/A-grade patients are written to `splits.json` and not printed elsewhere.

## 2. Model specification (scientific spec = submitted paper)
Unchanged from the paper unless a bug fix requires it:
- 3D ResU-Net, `base_ch` 16, ResBlock as implemented (the paper text will be corrected to match the code, AUDIT A1/A2).
- Deep heads as in the paper: cls 486 → 256 → 64 → 3 (with D-9, input dim becomes 490); surv GAP(z) + c → 256 → 64 → 1.
- Losses as in the paper; learnable homoscedastic weighting **without** the May 9 L2 prior. The fixed-weight initialisation is dropped from the text (it is not used, AUDIT A11).
- APD: GRL + 256 → 128 → n_train discriminator, λ = 0.1, sigmoid α over the fixed budget E. **Label space = the fold's training patients only.**
- Hyperparameters: lr 3e-4, wd 1e-5, batch 4, warmup 5, cosine to 1e-6, grad clip 1.0, AMP (Cox in float32), label smoothing 0.1, augmentation as in the paper.
- Input: cache rebuilt **natively** at 48×96×96 (fixes B16).

**Not adopted from the May 9 code**, because each was chosen after looking at the same validation folds: slimmed heads, fixed weights 1/2/2, label smoothing 0.05, batch 8, composite early stopping. If Husnain wants any of them, it becomes a **pre-specified secondary variant**, reported alongside, never instead.

**Adopted from the May 9 code** (bug fixes or infrastructure only): fold-aware CrMAE (`pretrain_foldaware.py`), Cox float32, single-event batches contribute (superseded by D-10), per-patient result CSVs, pooled C-index script (to be generalised).

## 3. Training budget and checkpoint rule (D-5, D-6)
- Every supervised variant (full model, ablations, single-task, imaging-only) trains for exactly **E = 150 epochs**. CrMAE and spatial MAE pretraining: **60 epochs**, encoder = last epoch (not lowest loss).
- The model evaluated is the **final-epoch checkpoint**. Segmentation, grading and survival are all read from that one checkpoint.
- No early stopping. No `best_*.pth`. The validation fold is used **only** for the final evaluation and for logging curves (R2-4). Curves are never used to choose anything.
- Rationale for E = 150: the paper used a 200-epoch cosine schedule with stops at 64–142. 150 lets the cosine schedule reach its floor in a budget comparable to what the paper's runs actually trained. **Disclosure:** this value is informed by earlier runs on the same cohort; the paper will say so.
- Sensitivity (pre-specified, E0 only, one seed): nested selection of E ∈ {50, 100, 150, 200} by an inner 4-fold split of each training fold, to show the conclusion does not depend on E.

## 4. Splits and seeds (D-7, D-8)
- `splits.json`, generated once by a committed script with seeds 42, 1337, 2026: for each seed, 5 folds stratified on (OS event × G3). It contains patient ID → fold for each seed, plus `has_mask` and `has_grade`. Committed **before** any training. Every experiment reads it; no script calls `StratifiedKFold` itself.
- Seed s controls the fold assignment, weight init, augmentation RNG and CrMAE masking.
- Expected events per validation fold ≈ 3 (15/5). Folds with fewer than 2 events are possible and are reported, not re-drawn.

## 5. Pretraining (R1-M4, R2-3, R4.4)
- CrMAE runs **inside each fold** on that fold's training patients only, separately for each seed (5 × 3 = 15 encoders). Encoders are shared across variants with the same (seed, fold) to save compute.
- Spatial-patch MAE (E1b): same encoder, decoder and budget; patches masked across **all** channels (no modality-specific masking).
- Normalisation is per volume (no cohort statistics), so nothing else needs refitting per fold.

## 6. What every run must save
`runs/<exp_id>/seed<s>/fold<k>/`:
- `config.json` (resolved), `git_commit.txt`, `splits_sha256.txt`, `env.txt`
- `history.csv`: per-epoch train/val loss per task, val Dice/acc/C-index (logging only), learned log-variances, Cox skip rate, ReID loss
- `final.pth` (E0 and the APD/privacy variants only; deleted for other variants after predictions are written, to respect the HPC quota)
- `oof_predictions.csv`: one row per validation patient: pid, seed, fold, has_mask, has_grade, grade label, 3 class probabilities, risk score, OS time, event, Dice, IoU, HD95 (mm), ASSD (mm), voxel spacing used, mask reference series
- `oof_seg/<pid>.npz`: predicted mask (uint8), for re-computing surface metrics

All tables and figures are generated **only** from `oof_predictions.csv` files by `evaluate.py`, never from training logs.

## 7. Metrics and statistics

**Unit of analysis = patient.** Folds and seeds are not independent samples.

**Segmentation (SEG, n = 48).** Dice and IoU without smoothing; HD95 and ASSD in mm (MONAI, per-patient voxel spacing of the resampled grid, computed from the DICOM extent of the MRI reference). Patient-level mean, median and IQR; 95% bootstrap CI. Dice at 48×96×96 is stated as not comparable to native-resolution literature.

**Grading (GRADE, n = 48).** Accuracy, balanced accuracy, macro OvR AUC, per-class precision/recall/specificity/F1, pooled and per-fold confusion matrices. Reference lines: majority class (28/48 = 58.3%), a subtype-only multinomial logistic model, and a clinical-only model (E4). Pre-specified secondary: binary G3 vs not-G3 AUC (28 vs 20), because Grade 1 has n = 5.

**Survival (SURV, n = 51, 15 events).**
- Primary (D-12): pooled OOF Harrell C-index after rank-normalising risk within each (seed, fold).
- Secondary: raw pooled C, Uno's C (τ = 3 years), per-fold C with events and comparable pairs, time-dependent AUC at 1 and 2 years, integrated Brier score over [0, 2 y]. The baseline hazard (Breslow) is fitted on the training fold only.
- Kaplan–Meier: high vs low risk, threshold = median risk of the **training-fold** patients applied to the validation patients (not tuned on outcomes); log-rank p.
- Comparators on the same splits: clinical-only Cox (E6) and radiomics + penalised Cox (E7).

**Seeds.** Per-patient predictions are averaged across the 3 seeds (class probabilities averaged; risk = mean within-fold rank). Metrics are computed on these averages (primary). Per-seed metrics are reported as mean ± SD (secondary).

**Uncertainty and tests.**
- 95% CIs: patient-level bootstrap, 2000 resamples, percentile (stratified by event for survival metrics).
- Variant vs. full model: paired bootstrap over patients for ΔDice, Δbalanced-accuracy, ΔAUC, ΔC-index, with 95% CI and two-sided p. Holm correction across the 4 ablation comparisons per metric. Effect sizes reported throughout.
- Wilcoxon signed-rank on 5 fold-level values: descriptive only, with the note that n = 5 cannot give p < 0.0625 (two-sided).
- McNemar exact test for paired correct/incorrect grade predictions and for paired re-identification outcomes.
- If nothing is significant, the paper says so (R1-M3).

**Privacy (D-13; R1-M6, R2-5, R4.9).**
- Threat model: an honest-but-curious party receives a trained encoder or its embeddings, and holds one other scan view of a target patient. Their goal: link the view to the patient's identity (re-identification), or decide whether a patient was in the training set (membership inference).
- Attack 1 (retrieval): for each patient, query = embedding from a partial view (e.g. MRI channel only, other channels zeroed), gallery = embeddings of all patients from the complementary view (PET + CT). Report top-1, top-5 and mAP over the gallery (chance 1/n), with and without APD, on training patients and on held-out patients separately.
- Attack 2 (membership inference): loss- and confidence-based attack, AUC distinguishing training-fold from held-out patients.
- DP reference: DP-SGD (Opacus) on the **grade head over frozen encoder embeddings** at ε ∈ {1, 8}, δ = 1e-5. Full-network DP is not attempted (BatchNorm, memory, n ≈ 40); stated as a limitation.
- Wording rule: APD is described as a heuristic regulariser with no formal guarantee, in the abstract and the conclusion.

## 8. Experiment matrix (all under this protocol, same splits, same budget)
| ID | Experiment | Seeds | Closes |
|---|---|---|---|
| E0 | Full SarcomaNet (within-fold CrMAE + APD + multi-scale, revised c) | 3 | R1-M1, headline numbers |
| E1 | Full model without pretraining (vs E0 = leak-free CrMAE effect) | 3 | R1-M4, R2-3 |
| E1b | Spatial-patch MAE instead of CrMAE | 3 | R4.4 |
| E2 | Ablation: baseline / +CrMAE / +APD / +multi-scale (single components) | 3 | R1-M3, R4.1, R4.6, R4.7 |
| E3 | Imaging only (no clinical vector) | 3 | R1-M5 |
| E4 | Clinical only: logistic (grade), Cox (survival), subtype-only logistic (CPU) | — | R1-M5 |
| E4b | Original 6-d c (sensitivity for D-9) | 1 | R1-M5 |
| E5 | Survival analysis package on E0 OOF predictions (CPU) | — | R1-M7, R2-2 |
| E6 | Clinical-only Cox (age, sex, site, subtype; penalised) (CPU) | — | R1-M7 |
| E7 | Radiomics (PyRadiomics on GT masks, MRI + PET) + penalised Cox, nested within folds (CPU) | — | R1-M2 |
| E8 | Privacy attacks, McNemar, DP-SGD head (§7) | — | R1-M6, R2-5, R4.9 |
| E9 | Single-task models: seg-only, grade-only, survival-only | 3 | R1-M8 |
| E10 | Seg baselines on SEG set: nnU-Net 3d_fullres (forced splits, capped schedule), MONAI SwinUNETR, SegResNet, 3D U-Net | 1 | R1-M8, R4.3 |
| E11 | Nested-E sensitivity (§3); MRI-referenced-mask sensitivity (D-11) | 1 | R2-1, R4.1 |

## 9. Compute estimate (D-14)
Basis: measured **~9 s/epoch** on the RTX 4090 at 48×96×96, `base_ch` 16 (`Output_Run_v3_Diagnostic`, `outputs`, `Output_Run_20260423_Unique01` logs). H100 is assumed **no faster** (data loading is CPU-bound with `num_workers` ≤ 4); this is conservative. CrMAE ≈ 5 s/epoch (assumed; no timing log on disk).
- One 5-fold supervised run at E = 150: 5 × 150 × 9 s ≈ **1.9 GPU-h**. CrMAE for 5 folds: ≈ 0.4 GPU-h.

| Exp. | GPU-h |
|---|---|
| CrMAE encoders (3 seeds × 5 folds) | 1.2 |
| Spatial MAE encoders | 1.2 |
| E0 | 5.7 |
| E1 | 5.7 |
| E1b | 5.7 |
| E2 (4 variants) | 22.8 |
| E3 | 5.7 |
| E4b, E11 | ~6 |
| E9 (3 tasks) | 17.1 |
| E10 (nnU-Net capped ≈ 12; MONAI 3 nets ≈ 5) | ~17 |
| E8 (DP head, embedding extraction) | ~2 |
| Dry runs, failures (+15%) | ~13 |
| **Total** | **≈ 100 GPU-h** |

With up to 8 GPUs in parallel (QoS `issai-h100`), wall time is about 15–20 h plus queueing. Every individual 5-fold run is about 2 GPU-h, under the 6 GPU-h confirmation threshold. Batches of more than 10 jobs will be confirmed first, per CLAUDE.md.
**Storage:** about 0.1 GB per run without weights, about 0.4 GB with `final.pth`. Total < 15 GB, which fits the ~58 GiB free, but only if `ckpt_latest`-style files are not kept.
**If the budget must be cut:** drop E10 nnU-Net (keep MONAI), run E2/E9 with 1 seed. Floor ≈ 50 GPU-h.

## 10. Reproducibility
- New repo `sarcomanet-revision` (copy of the May 9 code, not the original folder), branch `array-revision`. `.gitignore` covers `*.pth`, `cache/`, `archive/`, `runs/**/oof_seg/`. Baseline commit before any edit; each bug fix in `BUGS.md` with before/after; tests for `evaluate.py` on synthetic data with known answers.
- Every run's directory name contains the experiment ID, seed and date. Slurm scripts are committed and record the job ID.
- Python environment: a new venv on the HPC (Python 3.11 module, PyTorch built for CUDA 12.x), pinned in `requirements.lock`.
- Privacy of the remote: confirm `HRDevelops/SarcomaNet_Trihead` is private before pushing anything; no patient-level outputs in git.

## 11. Change log of this protocol
| Date | Change | Reason | Approved by |
|---|---|---|---|
| 2026-10-29 | Draft created | Phase 1.3 | — |
| 2026-10-29 | D-1 to D-15 approved as drafted; protocol frozen | Phase 1.5 | Husnain |
| 2026-10-29 | §7 integrated Brier score window: [first observed follow-up (131 d), 2 y] instead of [0, 2 y] | IPCW Brier score (scikit-survival) is undefined before the first observed time; technical, decided before any result | logged by Claude, for Husnain's note |
| 2026-10-29 | §7 KM grouping across seeds: patient is high-risk if high in the majority of seeds (each seed: risk > median training-fold risk) | Protocol did not say how to combine seeds for KM; decided before any result | logged by Claude, for Husnain's note |
| 2026-10-29 | **D-10 fallback applied:** survival loss = batch-local Breslow Cox including single-event batches (no memory bank), supervised batch size 4 → 8. Everything else unchanged (lr, epochs, encoders, pretraining batch 4). New experiment ID `E0_full_r2`; run 1 (`E0_full`) kept as a record | Run 1 (commit 5c72189): with the memory bank the training survival loss diverged (mean 2.2 at epoch 1 → 2024 at epoch 150; risk scores ~10^4). Trigger = training-loss diagnostic, not validation performance; the fallback was pre-specified in D-10 | Husnain, 2026-10-29 |
| 2026-10-29 | §7 comparison families fixed before any comparison was run. **Primary:** (a) E2 ablation family = E2_baseline, E2_crmae, E2_apd, E2_ms each vs E0_full_r2, Holm across the 4 per metric (as pre-specified); (b) single comparisons vs E0_full_r2: E1_nopre, E1b_spatial, E3_imgonly; (c) E9 single-task models vs E0_full_r2 on their own task only (E9_seg: Dice; E9_cls: balanced accuracy, macro AUC; E9_surv: C-index), Holm across the E9 family per metric is not needed (one test each); (d) E4_clinical, E4_subtype vs E0_full_r2 on grade and C-index, Holm across the 2. **Secondary, exploratory (not pre-specified, labelled as such):** E2_crmae, E2_apd, E2_ms each vs E2_baseline (the "+component" effect), Holm across the 3 | Protocol named the tests but not how to group experiments other than the 4 ablations; decided before running any comparison (summaries of the variants were being computed, but no paired test had been run) | logged by Claude, for Husnain's note |
| 2026-10-29 | **E7 design, fixed before running.** Features: PyRadiomics 3.1.0 (`radiomics_extract.py`, own venv, `requirements_radiomics.lock`) on the cached network inputs (T2FS, PET; GT mask; spacing = resampled spacing), Original image type, shape (once) + first-order, GLCM, GLRLM, GLSZM, GLDM, NGTDM, binCount 32. PET is per-patient normalised in the cache, so PET features are not absolute SUV (stated as a limitation). Model (`baselines_radiomics.py`), all fitted on the training fold: 3 empty-mask patients imputed with training-fold medians (flagged), standardise, drop constant and |r| > 0.95 features, grade = L2 multinomial logistic (C ∈ {0.001, 0.01, 0.1, 1} by inner 4-fold CV), OS = elastic-net Cox (l1_ratio 0.5, penalty ∈ {0.1, 0.3, 1, 3} by inner 4-fold CV). Compared with E0_full_r2 on balanced accuracy, macro AUC and pooled rank C (single comparison) | Protocol named radiomics + penalised Cox but not the details | logged by Claude, for Husnain's note |
| 2026-10-29 | **E8 design, fixed before running.** `privacy_attacks.py` on saved embeddings/predictions (no new training). Retrieval as §7 (query MRI-only view, gallery PET/CT-only view of all 51, cosine; top-1, top-5, MRR; held-out and training queries separately). APD effect = E2_apd vs E2_baseline (the only pair differing in APD alone): paired bootstrap on held-out MRR + exact McNemar on per-patient top-1 (hit in ≥ 2 of 3 seeds). E0_full_r2 and E1_nopre reported descriptively. Membership inference = confidence on the true grade (rank-equivalent to the loss attack, so one AUC), training vs held-out per run, mean ± SD over 15 runs. DP head: linear 256→3 on standardised full-view E0 embeddings, SGD lr 0.1, batch 16, 100 epochs, clip 1.0, Opacus RDP accountant, ε ∈ {1, 8}, δ = 1e-5, plus the same head without noise as reference. Opacus 1.6.0 added to the main venv (torch unchanged) | Protocol fixed the attacks but not these parameters | logged by Claude, for Husnain's note |
| 2026-10-29 | Environment note: `venv_radiomics` (E7) runs Python 3.9.25 (system Python; the module was not loaded in that shell), not 3.11. Packages are pinned in `requirements_radiomics.lock`. No effect on the E7 results | Found while building the nnU-Net venv | logged by Claude, for Husnain's note |
| 2026-10-29 | **E10 design, fixed before running** (Husnain approved E10/E11). Seed 42 only. Inputs = the cached 48×96×96 SarcomaNet inputs for every baseline (like for like; not native resolution). **MONAI** (`seg_baselines.py`, monai 1.5.1 in the main venv, torch unchanged): UNet (16-256, 4 strided levels, 2 residual units), SegResNet (16 filters, blocks 1-2-2-4 / 1-1-1), SwinUNETR (feature size 24; depth zero-padded 48→64, output cropped). Identical to E0 in everything else: SEG training patients of the fold, augmentation, loss 0.5 Dice + 0.5 focal, AdamW 3e-4 / 1e-5, warm-up 5 + cosine, AMP, clip 1.0, batch 8, 150 epochs, final epoch only. **nnU-Net** v2.6.2 (`venv_nnunet311`, torch 2.6.0): 3d_fullres, `nnUNetTrainer_250epochs` (capped schedule), forced `splits_final.json` = frozen seed-42 folds restricted to SEG patients, per-image z-score normalisation (channels named so nnU-Net applies no dataset-wide CT statistics), default nnU-Net augmentation/plans, validation predictions from `checkpoint_final.pth` (never `checkpoint_best`), no postprocessing or ensembling. Comparison: Dice vs E0_full_r2 (paired bootstrap, Holm across the 4 baselines); E9_seg reported alongside | Protocol named the networks but not their settings | logged by Claude, for Husnain's note |
| 2026-10-29 | **E11 design, fixed before running.** Outer seed 42. Inner 4-fold splits of each outer training fold (`make_inner_splits.py` → `splits_inner/outer<k>.json`, same stratification as splits.json). Each E ∈ {50, 100, 150, 200} is a separate E0 run (the cosine schedule depends on E), with a CrMAE encoder pretrained on the inner training set only. Selection score on pooled inner OOF predictions: S = (mean Dice + macro AUC + pooled rank-normalised C) / 3; E* = argmax, ties → smaller E (`e11_nested.py select`, writes `configs/E11/outer_fold<k>.json`). Outer refit on the whole outer training fold for E* epochs with the existing outer-fold encoder, evaluated once on the outer validation fold (`E11_nestedE`), reported against E0_full_r2 seed 42. Compute ≈ 5 GPU-h (20 inner tasks × (pretraining + 500 epochs), 5 refits), above the ~2.5 GPU-h first estimate | Protocol fixed the grid but not the selection score or the schedule handling | logged by Claude, for Husnain's note |
| 2026-10-29 | **E10 nnU-Net: torch.compile off** (`nnUNet_compile=f`). The first training array (job 34875) failed on all 5 folds in the first iterations: triton's compiled `.so` could not be loaded from `/tmp` on the compute nodes (`failed to map segment from shared object`). No checkpoint was written, so the rerun starts clean. `torch.compile` is a speed optimisation only; the trainer, plans, splits and schedule are unchanged | Infrastructure failure | logged by Claude |
| 2026-10-29 | **E11 selection result** (`configs/E11/selection.json`, from the 80 inner runs at commit 905d03d): E* = 200, 200, 150, 50, 200 for outer folds 0-4. Outer refits use these configs unchanged | Recorded before the refits run | logged by Claude |
| 2026-10-29 | **Single-seed comparisons use a single-seed reference.** E10 and E11 are seed 42 only, so their paired comparisons use E0_full_r2 restricted to seed 42 (`evaluate.py compare --ref_seeds 42`); the 3-seed E0 figures are shown alongside, descriptively only. Comparing one seed against a 3-seed average would favour the reference (seed averaging smooths probabilities and risks). Fixed before any E10/E11 comparison was run | E10/E11 design said 'vs E0_full_r2' without saying which seeds | logged by Claude, for Husnain's note |
