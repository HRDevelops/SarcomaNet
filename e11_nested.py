"""
e11_nested.py — E11 nested-E sensitivity (PROTOCOL §3; design in §11). Two sub-commands:

    python e11_nested.py configs                    # write configs/E11/o<k>_E<E>.json (inner runs)
    python e11_nested.py select --run_root ../runs  # choose E per outer fold, write configs/E11/outer_fold<k>.json

Inner runs: E0 configuration, outer seed 42, for each outer fold k and E in {50, 100, 150, 200}, trained on each
inner split j of splits_inner/outer<k>.json with a CrMAE encoder pretrained on that inner training set only
(encoder_root ../runs/encoders_E11/o<k>). Each E is its own run: the cosine schedule depends on E.

Selection score (fixed before any inner run): on the pooled inner out-of-fold predictions of outer fold k,
    S(E) = ( mean Dice + macro OvR AUC + pooled rank-normalised Harrell C ) / 3
E* = argmax S(E); ties -> the smaller E. The outer model is then retrained on the whole outer training fold for
E* epochs with the existing outer-fold encoder and evaluated once on the outer validation fold (exp E11_nestedE).
"""

import argparse
import json
from pathlib import Path

import numpy as np

import evaluate as ev
from revision_utils import load_splits

E_GRID = [50, 100, 150, 200]
OUTER_SEED = 42
N_OUTER = 5
CFG_DIR = Path("configs/E11")


def inner_exp(k, E):
    return f"E11_inner_o{k}_E{E}"


def write_configs():
    base = json.loads(Path("configs/E0.json").read_text())
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    for k in range(N_OUTER):
        for E in E_GRID:
            c = dict(base)
            c.update({"exp_id": inner_exp(k, E), "splits": f"splits_inner/outer{k}.json", "epochs": E,
                      "encoder_root": f"../runs/encoders_E11/o{k}", "save_final": False,
                      "_comment": f"E11 inner run: E0 config, outer seed {OUTER_SEED} fold {k}, E = {E}. "
                                  "Only exp_id, splits, epochs, encoder_root, save_final differ from configs/E0.json."})
            f = CFG_DIR / f"o{k}_E{E}.json"
            if f.exists():
                raise SystemExit(f"{f} exists")
            f.write_text(json.dumps(c, indent=1))
    print(f"wrote {N_OUTER * len(E_GRID)} configs to {CFG_DIR}")


def score(oof):
    seg = oof[oof.has_mask == 1]; gr = oof[oof.has_grade == 1]
    dice = float(np.nanmean(seg.dice))
    auc = ev.grade_metrics(gr.grade.values.astype(int), gr[ev.PROB_COLS].values)["macro_auc_ovr"]
    r = ev.add_rank_risk(oof)
    c = ev.harrell_c(r.time.values, r.event.values.astype(int), r.risk_rank.values)
    return {"dice": dice, "macro_auc_ovr": float(auc), "c_pooled_rank": float(c), "S": (dice + auc + c) / 3}


def select(run_root):
    base = json.loads(Path("configs/E0.json").read_text())
    out = {"rule": "S = (Dice + macro AUC + pooled rank C)/3 on pooled inner OOF; argmax, ties -> smaller E",
           "outer_seed": OUTER_SEED, "folds": {}}
    for k in range(N_OUTER):
        sp = load_splits(f"splits_inner/outer{k}.json")
        res = {}
        for E in E_GRID:
            oof, _ = ev.load_runs(Path(run_root), inner_exp(k, E), sp)
            res[E] = score(oof)
        best = max(E_GRID, key=lambda E: (round(res[E]["S"], 12), -E))
        out["folds"][str(k)] = {"scores": {str(E): v for E, v in res.items()}, "chosen_E": best}
        c = dict(base)
        c.update({"exp_id": "E11_nestedE", "epochs": best, "save_final": False,
                  "_comment": f"E11 outer refit, fold {k}: E0 config with epochs = {best} chosen by e11_nested.py select "
                              "(inner 4-fold, outer seed 42). Run with --seed 42 --fold %d only." % k})
        f = CFG_DIR / f"outer_fold{k}.json"
        if f.exists():
            raise SystemExit(f"{f} exists")
        f.write_text(json.dumps(c, indent=1))
        print(f"outer fold {k}: " + ", ".join(f"E{E} S={res[E]['S']:.3f}" for E in E_GRID) + f" -> E* = {best}")
    Path(CFG_DIR / "selection.json").write_text(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("configs")
    s = sub.add_parser("select"); s.add_argument("--run_root", required=True)
    a = ap.parse_args()
    write_configs() if a.cmd == "configs" else select(a.run_root)


if __name__ == "__main__":
    main()
