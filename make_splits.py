"""
make_splits.py — build the frozen fold assignment (PROTOCOL §4, D-7, D-8). Run ONCE.

    python make_splits.py --clinical_xls archive/INFOclinical_STS.xlsx --out splits.json

* 51 patients, 5 folds, seeds 42 / 1337 / 2026.
* Stratified on (OS event) x (grade == High). Missing grade counts as "not High"
  for splitting only.
* has_mask = False for the three RT-STRUCT failures (D-3); has_grade = False for
  grade "N/A" (D-2).
* Refuses to overwrite an existing file (the split is frozen once committed).

Every training / evaluation script reads splits.json; nothing else calls StratifiedKFold.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from data_pipeline import GRADE_MISSING, ClinicalBuilder

SEEDS          = [42, 1337, 2026]
N_FOLDS        = 5
MASK_FAILURES  = ["STS_024", "STS_039", "STS_045"]   # PROTOCOL D-3


def build(clinical_xls: Path, data_root: Path = None) -> dict:
    cb  = ClinicalBuilder(clinical_xls, endpoint="os")
    ids = sorted(p for p in cb.index if re.fullmatch(r"STS_\d{3}", p))
    if data_root is not None:
        on_disk = sorted(p.name for p in Path(data_root).iterdir() if p.is_dir())
        if on_disk != ids:
            raise SystemExit(f"patient list mismatch: sheet {len(ids)} vs disk {len(on_disk)}")
    if len(ids) != 51:
        raise SystemExit(f"expected 51 patients, found {len(ids)}")

    grade  = {p: cb.grade(p) for p in ids}
    event  = {p: int(cb.survival(p, "os")[1]) for p in ids}
    strata = np.array([2 * event[p] + int(grade[p] == 2) for p in ids])

    out = {
        "description": "Frozen CV splits for the Array revision (PROTOCOL.md §4).",
        "n_folds": N_FOLDS,
        "seeds": SEEDS,
        "stratification": "OS event x (grade == High); missing grade -> not High",
        "patients": ids,
        "has_mask":  {p: p not in MASK_FAILURES for p in ids},
        "has_grade": {p: grade[p] != GRADE_MISSING for p in ids},
        "folds": {},
    }
    for s in SEEDS:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=s)
        fold_of = {}
        for k, (_, val_idx) in enumerate(skf.split(ids, strata)):
            for i in val_idx:
                fold_of[ids[i]] = k
        out["folds"][str(s)] = fold_of
    return out


def summary(sp: dict) -> str:
    lines = []
    ids = sp["patients"]
    for s, fold_of in sp["folds"].items():
        for k in range(sp["n_folds"]):
            val = [p for p in ids if fold_of[p] == k]
            n_seg = sum(sp["has_mask"][p] for p in val)
            n_gr  = sum(sp["has_grade"][p] for p in val)
            lines.append(f"seed {s} fold {k}: n={len(val)} seg={n_seg} grade={n_gr}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinical_xls", required=True)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--out", default="splits.json")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; splits are frozen. Delete only with approval.")
    sp = build(Path(a.clinical_xls), Path(a.data_root) if a.data_root else None)
    text = json.dumps(sp, indent=1, sort_keys=True)
    out.write_text(text)
    print(summary(sp))
    print("sha256:", hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
