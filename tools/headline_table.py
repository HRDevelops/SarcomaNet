"""Print a markdown headline table from <results>/<exp>_summary.json files (no recomputation).
    python tools/headline_table.py --results ../results --exps E0_full_r2 E1_nopre ...
"""
import argparse
import json
from pathlib import Path


def f(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def ci(d, nd=3):
    if not d or d.get("ci95") is None:
        return "—"
    lo, hi = d["ci95"]
    return f"{f(d.get('value', d.get('mean')), nd)} ({f(lo, nd)}–{f(hi, nd)})"


def row(exp, s):
    seg, gr, sv = s.get("segmentation") or {}, s.get("grading") or {}, s.get("survival") or {}
    km = sv.get("km_logrank") or {}
    return [exp,
            ci(seg.get("dice")) if seg else "—",
            f(seg["hd95"]["mean"], 1) if seg.get("hd95") else "—",
            f(gr["accuracy"]["value"]) if gr else "—",
            ci(gr.get("balanced_accuracy")) if gr else "—",
            ci(gr.get("macro_auc_ovr")) if gr else "—",
            f(gr["auc_high_vs_rest"]["value"]) if gr else "—",
            ci(sv.get("c_pooled_rank_normalised")) if sv else "—",
            f(sv.get("uno_c_tau3y")), f(sv.get("ibs_to_2y")),
            f(km.get("p"), 2) if km else "—"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--exps", nargs="+", required=True)
    a = ap.parse_args()
    hdr = ["Exp", "Dice (95% CI)", "HD95 mm", "Acc", "Bal. acc (95% CI)", "Macro AUC (95% CI)", "AUC G3 vs rest",
           "C pooled rank (95% CI)", "Uno C", "IBS 2y", "KM p"]
    print("| " + " | ".join(hdr) + " |"); print("|" + "---|" * len(hdr))
    for e in a.exps:
        p = Path(a.results) / f"{e}_summary.json"
        if not p.exists():
            print(f"| {e} | missing |"); continue
        print("| " + " | ".join(row(e, json.loads(p.read_text()))) + " |")


if __name__ == "__main__":
    main()
