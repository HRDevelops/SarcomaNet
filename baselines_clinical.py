"""
baselines_clinical.py — non-imaging baselines on the frozen splits (PROTOCOL §8: E4, E6).

    python baselines_clinical.py --config configs/E0.json --exp E4_clinical --features clinical
    python baselines_clinical.py --config configs/E0.json --exp E4_subtype  --features subtype

For every (seed, fold) of splits.json:
  * grade: multinomial L2 logistic regression on the training-fold GRADE patients,
    C chosen by inner stratified 4-fold CV (log-loss) inside the training fold only
  * survival (OS): L2-penalised Cox (lifelines), penalizer chosen by inner 4-fold CV
    (Harrell C) inside the training fold only
Features: "clinical" = clinical vector v2 (age, sex, site, one-hot MSKCC; D-9)
          "subtype"  = one-hot MSKCC only
Continuous features are standardised with training-fold statistics.

Writes the same files as train.py (oof_predictions.csv, train_predictions.csv, run_meta.json),
with has_mask = 0, so evaluate.py summarises them unchanged.
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from data_pipeline import GRADE_MISSING, ClinicalBuilder
from evaluate import harrell_c
from revision_utils import fold_ids, load_splits, write_run_meta

warnings.filterwarnings("ignore")
C_GRID   = [0.01, 0.1, 1.0, 10.0]
PEN_GRID = [0.01, 0.1, 1.0, 10.0]


def features(cb, pids, kind):
    X = np.stack([cb.clinical_vector(p, "v2") for p in pids])
    return X[:, 3:] if kind == "subtype" else X


def standardise(Xtr, Xte, cont_cols):
    Xtr, Xte = Xtr.copy(), Xte.copy()
    for c in cont_cols:
        mu, sd = Xtr[:, c].mean(), Xtr[:, c].std() or 1.0
        Xtr[:, c] = (Xtr[:, c] - mu) / sd; Xte[:, c] = (Xte[:, c] - mu) / sd
    return Xtr, Xte


def fit_grade(X, y, seed, c_grid=C_GRID):
    cv = StratifiedKFold(4, shuffle=True, random_state=seed)
    best, best_ll = None, np.inf
    for C in c_grid:
        ll = []
        for a, b in cv.split(X, y):
            m = LogisticRegression(C=C, max_iter=5000).fit(X[a], y[a])
            P = np.clip(m.predict_proba(X[b]), 1e-9, 1)
            cols = list(m.classes_)
            ll.append(-np.mean([np.log(P[i, cols.index(y[b][i])]) if y[b][i] in cols else np.log(1e-9)
                                for i in range(len(b))]))
        if np.mean(ll) < best_ll:
            best, best_ll = C, np.mean(ll)
    return LogisticRegression(C=best, max_iter=5000).fit(X, y), best


def fit_cox(X, t, e, seed, pen_grid=PEN_GRID, l1_ratio=0.0):
    from lifelines import CoxPHFitter
    cols = [f"x{i}" for i in range(X.shape[1])]
    keep = [i for i in range(X.shape[1]) if X[:, i].std() > 0]          # drop constant columns (absent subtypes)
    df = pd.DataFrame(X[:, keep], columns=[cols[i] for i in keep]); df["t"] = t; df["e"] = e
    cv = StratifiedKFold(4, shuffle=True, random_state=seed)
    best, best_c = None, -np.inf
    for pen in pen_grid:
        cs = []
        for a, b in cv.split(df, df.e):
            try:
                m = CoxPHFitter(penalizer=pen, l1_ratio=l1_ratio).fit(df.iloc[a], "t", "e")
                cs.append(harrell_c(df.t.values[b], df.e.values[b], m.predict_log_partial_hazard(df.iloc[b]).values))
            except Exception:   # noqa: BLE001
                cs.append(np.nan)
        if np.nanmean(cs) > best_c:
            best, best_c = pen, np.nanmean(cs)
    if best is None:                       # every inner fit failed: fall back to the strongest penalty
        best = max(pen_grid)
    m = CoxPHFitter(penalizer=best, l1_ratio=l1_ratio).fit(df, "t", "e")
    return (lambda Z: m.predict_log_partial_hazard(pd.DataFrame(Z[:, keep], columns=[cols[i] for i in keep])).values), best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--features", choices=["clinical", "subtype"], required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text()); sp = load_splits(cfg["splits"])
    cb = ClinicalBuilder(cfg["clinical_xls"], endpoint=cfg["endpoint"], clinical_version="v2")
    cont = [0] if a.features == "clinical" else []                     # age is the only continuous column

    for s in sp["seeds"]:
        for k in range(sp["n_folds"]):
            out = Path(cfg["run_root"]) / a.exp / f"seed{s}" / f"fold{k}"
            if (out / "oof_predictions.csv").exists():
                raise SystemExit(f"{out} exists; refusing to overwrite")
            tr, va = fold_ids(sp, s, k)
            Xtr, Xva = standardise(features(cb, tr, a.features), features(cb, va, a.features), cont)
            ytr = np.array([cb.grade(p) for p in tr]); gm = ytr != GRADE_MISSING
            clf, C = fit_grade(Xtr[gm], ytr[gm], s * 100 + k)
            def probs(X):
                P = np.zeros((len(X), 3)); P[:, clf.classes_] = clf.predict_proba(X); return P
            ttr = np.array([cb.survival(p)[0] for p in tr]); etr = np.array([cb.survival(p)[1] for p in tr])
            risk_fn, pen = fit_cox(Xtr, ttr, etr, s * 100 + k)

            def rows(pids, X):
                P, r = probs(X), risk_fn(X); out_rows = []
                for i, p in enumerate(pids):
                    g = cb.grade(p); t, e = cb.survival(p)
                    out_rows.append({"exp_id": a.exp, "seed": s, "fold": k, "pid": p, "has_mask": 0,
                                     "has_grade": int(g != GRADE_MISSING), "grade": g,
                                     "p_low": P[i, 0], "p_int": P[i, 1], "p_high": P[i, 2], "risk": float(r[i]),
                                     "time": t, "event": int(e), "dice": np.nan, "iou": np.nan,
                                     "pred_voxels": -1, "gt_voxels": -1})
                return pd.DataFrame(out_rows)
            out.mkdir(parents=True, exist_ok=True)
            write_run_meta(out, cfg, sp, {"stage": "baseline_clinical", "features": a.features, "seed": s, "fold": k,
                                          "chosen_C": C, "chosen_penalizer": pen, "n_train": len(tr), "val_ids": va})
            rows(tr, Xtr).drop(columns=["dice", "iou", "pred_voxels", "gt_voxels"]).to_csv(out / "train_predictions.csv", index=False)
            rows(va, Xva).to_csv(out / "oof_predictions.csv", index=False)
            print(f"{a.exp} seed {s} fold {k}: C={C} penalizer={pen}")


if __name__ == "__main__":
    main()
