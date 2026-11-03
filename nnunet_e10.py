"""
nnunet_e10.py — E10 nnU-Net baseline helpers (PROTOCOL §8 E10; design in §11). Runs in venv_nnunet311
(numpy, pandas, SimpleITK only; no imports from this repo's torch code).

    python nnunet_e10.py prepare --cache_dir ../cache_48x96x96 --raw $nnUNet_raw
    python nnunet_e10.py splits  --preprocessed $nnUNet_preprocessed
    python nnunet_e10.py collect --results $nnUNet_results --cache_dir ../cache_48x96x96 --run_root ../runs \
                                 --ref_exp E0_full_r2

prepare : Dataset501_STSrev from the SAME cached inputs as SarcomaNet (48x96x96, spacing = resampled spacing),
          SEG patients only (48). Channels are named so that nnU-Net applies per-image z-score normalisation
          (no dataset-wide intensity statistics, hence no cross-fold intensity leakage).
splits  : splits_final.json = frozen splits.json seed 42 folds restricted to SEG patients (nnU-Net fold k = fold k).
collect : final-checkpoint validation predictions (fold_k/validation) -> runs/E10_nnunet/seed42/fold<k>/ in the
          train.py format (Dice vs the cached GT mask; grade/survival columns uninformative, never reported).
          Patient metadata (grade, time, event) is copied from <ref_exp>/seed42/fold<k>/oof_predictions.csv.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

DATASET = "Dataset501_STSrev"
TRAINER = "nnUNetTrainer_250epochs"
PLANS, CONF = "nnUNetPlans", "3d_fullres"
SEED = 42
CHANNELS = {"0": "T2FS_cached", "1": "PET_cached", "2": "CT_cached"}   # not "CT" -> z-score per image


def load_splits(path="splits.json"):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def splits_sha256(sp):
    return hashlib.sha256(json.dumps({k: v for k, v in sp.items() if not k.startswith("_")},
                                     sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_nii(a, spacing_zyx, path):
    im = sitk.GetImageFromArray(a); im.SetSpacing(tuple(float(s) for s in spacing_zyx[::-1]))
    sitk.WriteImage(im, str(path), useCompression=True)


def prepare(cache_dir, raw):
    sp = load_splits(); d = Path(raw) / DATASET
    if d.exists():
        raise SystemExit(f"{d} exists; refusing to overwrite")
    (d / "imagesTr").mkdir(parents=True); (d / "labelsTr").mkdir()
    seg = [p for p in sp["patients"] if sp["has_mask"][p]]
    for p in seg:
        c = Path(cache_dir) / p; meta = json.loads((c / "meta.json").read_text())
        img, m = np.load(c / "image.npy"), np.load(c / "mask.npy").astype(np.uint8)
        if m.sum() == 0:
            raise SystemExit(f"{p}: empty mask in cache but has_mask is True")
        for ch in range(3):
            write_nii(img[ch].astype(np.float32), meta["resampled_spacing_mm"], d / "imagesTr" / f"{p}_{ch:04d}.nii.gz")
        write_nii(m, meta["resampled_spacing_mm"], d / "labelsTr" / f"{p}.nii.gz")
    (d / "dataset.json").write_text(json.dumps({"channel_names": CHANNELS, "labels": {"background": 0, "tumour": 1},
                                                "numTraining": len(seg), "file_ending": ".nii.gz",
                                                "description": "TCIA STS, cached 48x96x96 SarcomaNet inputs, "
                                                               f"splits.json sha {splits_sha256(sp)[:8]}"}, indent=1))
    print(f"wrote {len(seg)} cases to {d}")


def splits(preprocessed):
    sp = load_splits(); f = sp["folds"][str(SEED)]
    seg = [p for p in sp["patients"] if sp["has_mask"][p]]
    folds = [{"train": [p for p in seg if f[p] != k], "val": [p for p in seg if f[p] == k]} for k in range(sp["n_folds"])]
    dest = Path(preprocessed) / DATASET / "splits_final.json"
    if dest.exists():
        if json.loads(dest.read_text()) != folds:
            raise SystemExit(f"{dest} exists and differs from the frozen splits")
        print(f"{dest} already matches the frozen splits"); return
    dest.write_text(json.dumps(folds, indent=1))
    print(f"wrote {dest}: val sizes {[len(x['val']) for x in folds]}")


def dice_iou(p, g):
    s = p.sum() + g.sum()
    if g.sum() == 0:
        return np.nan, np.nan
    inter = np.logical_and(p, g).sum()
    return 2 * inter / s, inter / (s - inter)


def collect(results, cache_dir, run_root, ref_exp):
    sp = load_splits(); base = Path(results) / DATASET / f"{TRAINER}__{PLANS}__{CONF}"
    for k in range(sp["n_folds"]):
        out = Path(run_root) / "E10_nnunet" / f"seed{SEED}" / f"fold{k}"
        if (out / "oof_predictions.csv").exists():
            raise SystemExit(f"{out} exists; refusing to overwrite")
        fd = base / f"fold_{k}"
        if not (fd / "checkpoint_final.pth").exists():
            raise SystemExit(f"{fd}: no checkpoint_final.pth (training incomplete)")
        ref = pd.read_csv(Path(run_root) / ref_exp / f"seed{SEED}" / f"fold{k}" / "oof_predictions.csv")
        out.mkdir(parents=True); (out / "oof_seg").mkdir(); rows = []
        for _, r in ref.iterrows():
            p = r.pid; d = j = np.nan; pv = gv = -1
            if r.has_mask == 1:
                pred = sitk.GetArrayFromImage(sitk.ReadImage(str(fd / "validation" / f"{p}.nii.gz"))).astype(bool)
                gt = np.load(Path(cache_dir) / p / "mask.npy").astype(bool)
                if pred.shape != gt.shape:
                    raise SystemExit(f"{p}: prediction {pred.shape} vs mask {gt.shape}")
                d, j = dice_iou(pred, gt); pv, gv = int(pred.sum()), int(gt.sum())
                np.savez_compressed(out / "oof_seg" / f"{p}.npz", pred=pred.astype(np.uint8))
            rows.append({"exp_id": "E10_nnunet", "seed": SEED, "fold": k, "pid": p, "has_mask": int(r.has_mask),
                         "has_grade": int(r.has_grade), "grade": int(r.grade), "p_low": 1 / 3, "p_int": 1 / 3,
                         "p_high": 1 / 3, "risk": 0.0, "time": r.time, "event": int(r.event), "dice": d, "iou": j,
                         "pred_voxels": pv, "gt_voxels": gv})
        df = pd.DataFrame(rows)
        tr = pd.read_csv(Path(run_root) / ref_exp / f"seed{SEED}" / f"fold{k}" / "train_predictions.csv")
        tr.assign(p_low=1 / 3, p_int=1 / 3, p_high=1 / 3, risk=0.0).to_csv(out / "train_predictions.csv", index=False)
        from revision_utils import git_state           # reads .git directly (no git binary on compute nodes)
        (out / "run_meta.json").write_text(json.dumps({
            "stage": "seg_baseline_nnunet", "trainer": TRAINER, "plans": PLANS, "configuration": CONF,
            "checkpoint": "checkpoint_final.pth (validation run by nnU-Net at the end of training)",
            "seed": SEED, "fold": k, "tasks": ["seg"], "splits_sha256": splits_sha256(sp),
            "git": git_state(), "nnunet_fold_dir": str(fd), "val_ids": list(df.pid)}, indent=1, default=str))
        df.to_csv(out / "oof_predictions.csv", index=False)
        print(f"fold {k}: mean Dice {np.nanmean(df.dice):.3f} over {int(df.has_mask.sum())} SEG patients")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--cache_dir", required=True); p.add_argument("--raw", required=True)
    s = sub.add_parser("splits"); s.add_argument("--preprocessed", required=True)
    c = sub.add_parser("collect"); c.add_argument("--results", required=True); c.add_argument("--cache_dir", required=True)
    c.add_argument("--run_root", required=True); c.add_argument("--ref_exp", default="E0_full_r2")
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare(a.cache_dir, a.raw)
    elif a.cmd == "splits":
        splits(a.preprocessed)
    else:
        collect(a.results, a.cache_dir, a.run_root, a.ref_exp)


if __name__ == "__main__":
    main()
