"""
make_inner_splits.py — E11 nested-E sensitivity (PROTOCOL §3): inner 4-fold splits of each outer training fold.

    python make_inner_splits.py --clinical_xls <INFOclinical_STS.xlsx> --outer_seed 42 --out_dir splits_inner

For outer seed 42 and each outer fold k, the outer TRAINING patients are split into 4 inner folds with the same
stratification as splits.json (OS event x grade == High). Each file has the splits.json format (patients, seeds,
folds, has_mask, has_grade), so train.py / pretrain_fold.py run unchanged with "splits": "splits_inner/outer<k>.json";
inner fold j of that file = inner split j. Outer validation patients never appear. Refuses to overwrite.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from data_pipeline import ClinicalBuilder
from revision_utils import fold_ids, load_splits

N_INNER = 4


def build(sp: dict, cb: ClinicalBuilder, outer_seed: int, k: int) -> dict:
    tr, va = fold_ids(sp, outer_seed, k)
    strata = np.array([2 * int(cb.survival(p, "os")[1]) + int(cb.grade(p) == 2) for p in tr])
    skf = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=outer_seed)
    fold_of = {}
    for j, (_, vi) in enumerate(skf.split(tr, strata)):
        for i in vi:
            fold_of[tr[i]] = j
    return {"description": f"E11 inner splits: outer seed {outer_seed}, outer fold {k} training patients only "
                           f"(PROTOCOL §3). Outer validation patients excluded: {va}",
            "n_folds": N_INNER, "seeds": [outer_seed], "patients": tr,
            "stratification": "OS event x (grade == High); missing grade -> not High",
            "has_mask": {p: sp["has_mask"][p] for p in tr}, "has_grade": {p: sp["has_grade"][p] for p in tr},
            "folds": {str(outer_seed): fold_of}, "outer": {"splits_sha256": sp["_sha256"], "seed": outer_seed,
                                                           "fold": k, "val_ids": va}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinical_xls", required=True)
    ap.add_argument("--splits", default="splits.json")
    ap.add_argument("--outer_seed", type=int, default=42)
    ap.add_argument("--out_dir", default="splits_inner")
    a = ap.parse_args()
    sp = load_splits(a.splits); cb = ClinicalBuilder(a.clinical_xls, endpoint="os")
    out = Path(a.out_dir); out.mkdir(exist_ok=True)
    for k in range(sp["n_folds"]):
        f = out / f"outer{k}.json"
        if f.exists():
            raise SystemExit(f"{f} exists; inner splits are frozen")
        d = build(sp, cb, a.outer_seed, k)
        text = json.dumps(d, indent=1, sort_keys=True); f.write_text(text)
        sizes = [sum(v == j for v in d["folds"][str(a.outer_seed)].values()) for j in range(N_INNER)]
        ev = [sum(int(cb.survival(p, "os")[1]) for p, v in d["folds"][str(a.outer_seed)].items() if v == j)
              for j in range(N_INNER)]
        print(f"outer {k}: n={len(d['patients'])} inner sizes {sizes} events {ev} sha {hashlib.sha256(text.encode()).hexdigest()[:8]}")


if __name__ == "__main__":
    main()
