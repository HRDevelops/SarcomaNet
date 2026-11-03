"""
baselines_radiomics.py — E7 step 2: radiomics models on the frozen splits (PROTOCOL §8 E7, R1-M2; design in §11).

    python baselines_radiomics.py --config configs/E0.json --features ../results/E7_radiomics_features.csv --exp E7_radiomics

Per (seed, fold), everything fitted on the training fold only:
  1. missing patients (empty GT mask, 3) -> training-fold median per feature (flag `imputed`)
  2. standardise with training-fold mean / SD
  3. drop near-constant features, then greedily drop features with |Pearson r| > 0.95 to an earlier kept one
  4. grade: multinomial L2 logistic, C by inner 4-fold CV (log-loss)        [same routine as E4]
  5. OS: elastic-net Cox (l1_ratio 0.5), penalty by inner 4-fold CV (Harrell C) [same routine as E4]
Writes the train.py output format (has_mask = 0) so evaluate.py summary/compare run unchanged.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from baselines_clinical import fit_cox, fit_grade
from data_pipeline import GRADE_MISSING, ClinicalBuilder
from revision_utils import fold_ids, load_splits, write_run_meta

C_GRID_RAD = [0.001, 0.01, 0.1, 1.0]
PEN_GRID_RAD = [0.1, 0.3, 1.0, 3.0]
L1_RATIO = 0.5
CORR_MAX = 0.95


def prepare(F: pd.DataFrame, tr, va):
    cols = [c for c in F.columns if c != "pid"]
    Ftr, Fva = F.reindex(tr)[cols], F.reindex(va)[cols]
    med = Ftr.median()
    imp_tr, imp_va = Ftr.isna().all(1).values, Fva.isna().all(1).values
    Xtr, Xva = Ftr.fillna(med).values, Fva.fillna(med).values
    mu, sd = Xtr.mean(0), Xtr.std(0)
    ok = sd > 1e-8
    Xtr, Xva = (Xtr[:, ok] - mu[ok]) / sd[ok], (Xva[:, ok] - mu[ok]) / sd[ok]
    names = list(np.array(cols)[ok])
    R = np.abs(np.corrcoef(Xtr, rowvar=False)); keep = []
    for j in range(Xtr.shape[1]):
        if all(R[j, k] <= CORR_MAX for k in keep):
            keep.append(j)
    return Xtr[:, keep], Xva[:, keep], [names[j] for j in keep], imp_tr, imp_va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--exp", required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text()); sp = load_splits(cfg["splits"])
    cb = ClinicalBuilder(cfg["clinical_xls"], endpoint=cfg["endpoint"], clinical_version="v2")
    F = pd.read_csv(a.features).set_index("pid", drop=False)

    for s in sp["seeds"]:
        for k in range(sp["n_folds"]):
            out = Path(cfg["run_root"]) / a.exp / f"seed{s}" / f"fold{k}"
            if (out / "oof_predictions.csv").exists():
                raise SystemExit(f"{out} exists; refusing to overwrite")
            tr, va = fold_ids(sp, s, k)
            Xtr, Xva, names, imp_tr, imp_va = prepare(F, tr, va)
            ytr = np.array([cb.grade(p) for p in tr]); gm = ytr != GRADE_MISSING
            clf, C = fit_grade(Xtr[gm], ytr[gm], s * 100 + k, C_GRID_RAD)
            ttr = np.array([cb.survival(p)[0] for p in tr]); etr = np.array([cb.survival(p)[1] for p in tr])
            risk_fn, pen = fit_cox(Xtr, ttr, etr, s * 100 + k, PEN_GRID_RAD, L1_RATIO)

            def rows(pids, X, imp):
                P = np.zeros((len(X), 3)); P[:, clf.classes_] = clf.predict_proba(X); r = risk_fn(X); o = []
                for i, p in enumerate(pids):
                    g = cb.grade(p); t, e = cb.survival(p)
                    o.append({"exp_id": a.exp, "seed": s, "fold": k, "pid": p, "has_mask": 0,
                              "has_grade": int(g != GRADE_MISSING), "grade": g, "p_low": P[i, 0], "p_int": P[i, 1],
                              "p_high": P[i, 2], "risk": float(r[i]), "time": t, "event": int(e), "dice": np.nan,
                              "iou": np.nan, "pred_voxels": -1, "gt_voxels": -1, "imputed": int(imp[i])})
                return pd.DataFrame(o)
            out.mkdir(parents=True, exist_ok=True)
            write_run_meta(out, cfg, sp, {"stage": "baseline_radiomics", "features_file": str(a.features),
                                          "seed": s, "fold": k, "chosen_C": C, "chosen_penalizer": pen,
                                          "l1_ratio": L1_RATIO, "n_features_kept": len(names),
                                          "features_kept": names, "n_train": len(tr), "val_ids": va})
            rows(tr, Xtr, imp_tr).drop(columns=["dice", "iou", "pred_voxels", "gt_voxels"]).to_csv(
                out / "train_predictions.csv", index=False)
            rows(va, Xva, imp_va).to_csv(out / "oof_predictions.csv", index=False)
            print(f"{a.exp} seed {s} fold {k}: {len(names)} features, C={C}, penalizer={pen}")


if __name__ == "__main__":
    main()
