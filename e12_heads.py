"""
e12_heads.py — E12 pilot, step 2 (PROTOCOL §11, 2026-11-02). Runs in the main venv.

    python e12_heads.py --config configs/E0.json --features ../results/E12_nnunet_features.npz \
        --seg_exp E10_nnunet --exp E12_nnunet_feat

Seed 42, per outer fold k: features of all patients from the fold-k nnU-Net encoder; grade and OS heads fitted on
the fold's training patients only, with exactly the E7 routine (baselines_radiomics.prepare, fit_grade, fit_cox and
the same grids). Segmentation columns (and oof_seg masks, copied) come from the nnU-Net run --seg_exp.
Writes the train.py output format so evaluate.py summary/compare run unchanged. Never overwrites.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from baselines_clinical import fit_cox, fit_grade
from baselines_radiomics import C_GRID_RAD, L1_RATIO, PEN_GRID_RAD, prepare
from data_pipeline import GRADE_MISSING, ClinicalBuilder
from revision_utils import fold_ids, load_splits, write_run_meta

SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--features", required=True)
    ap.add_argument("--seg_exp", default="E10_nnunet"); ap.add_argument("--exp", default="E12_nnunet_feat")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text()); sp = load_splits(cfg["splits"])
    cb = ClinicalBuilder(cfg["clinical_xls"], endpoint=cfg["endpoint"], clinical_version="v2")
    z = np.load(a.features); feats, pids = z["feats"], [str(p) for p in z["pids"]]
    if feats.shape[0] != sp["n_folds"] or sorted(pids) != sorted(sp["patients"]):
        raise SystemExit("feature file does not cover all folds and patients (dry-run file?)")
    root = Path(cfg["run_root"])
    for k in range(sp["n_folds"]):
        out = root / a.exp / f"seed{SEED}" / f"fold{k}"
        if (out / "oof_predictions.csv").exists():
            raise SystemExit(f"{out} exists; refusing to overwrite")
        F = pd.DataFrame(feats[k], columns=[f"f{j}" for j in range(feats.shape[2])]); F.insert(0, "pid", pids)
        F = F.set_index("pid", drop=False)
        tr, va = fold_ids(sp, SEED, k)
        Xtr, Xva, names, imp_tr, imp_va = prepare(F, tr, va)
        ytr = np.array([cb.grade(p) for p in tr]); gm = ytr != GRADE_MISSING
        clf, C = fit_grade(Xtr[gm], ytr[gm], SEED * 100 + k, C_GRID_RAD)
        ttr = np.array([cb.survival(p)[0] for p in tr]); etr = np.array([cb.survival(p)[1] for p in tr])
        risk_fn, pen = fit_cox(Xtr, ttr, etr, SEED * 100 + k, PEN_GRID_RAD, L1_RATIO)
        seg_dir = root / a.seg_exp / f"seed{SEED}" / f"fold{k}"
        seg = pd.read_csv(seg_dir / "oof_predictions.csv").set_index("pid")
        if sorted(seg.index) != sorted(va):
            raise SystemExit(f"{seg_dir}: validation patients differ from the frozen fold")

        def rows(ids, X, with_seg):
            P = np.zeros((len(X), 3)); P[:, clf.classes_] = clf.predict_proba(X); r = risk_fn(X); o = []
            for i, p in enumerate(ids):
                g = cb.grade(p); t, e = cb.survival(p)
                row = {"exp_id": a.exp, "seed": SEED, "fold": k, "pid": p, "has_mask": int(sp["has_mask"][p]),
                       "has_grade": int(g != GRADE_MISSING), "grade": g, "p_low": P[i, 0], "p_int": P[i, 1],
                       "p_high": P[i, 2], "risk": float(r[i]), "time": t, "event": int(e)}
                if with_seg:
                    s = seg.loc[p]
                    row.update({"dice": s.dice, "iou": s.iou, "pred_voxels": int(s.pred_voxels),
                                "gt_voxels": int(s.gt_voxels)})
                o.append(row)
            return pd.DataFrame(o)
        out.mkdir(parents=True, exist_ok=True)
        if (seg_dir / "oof_seg").exists():
            shutil.copytree(seg_dir / "oof_seg", out / "oof_seg")
        write_run_meta(out, cfg, sp, {"stage": "E12_nnunet_features_heads", "features_file": str(a.features),
                                      "seg_from": str(seg_dir), "seed": SEED, "fold": k, "chosen_C": C,
                                      "chosen_penalizer": pen, "l1_ratio": L1_RATIO, "n_features_kept": len(names),
                                      "n_train": len(tr), "val_ids": va, "tasks": ["seg", "cls", "surv"]})
        rows(tr, Xtr, False).to_csv(out / "train_predictions.csv", index=False)
        rows(va, Xva, True).to_csv(out / "oof_predictions.csv", index=False)
        print(f"{a.exp} fold {k}: {len(names)} features kept, C={C}, penalizer={pen}")


if __name__ == "__main__":
    main()
