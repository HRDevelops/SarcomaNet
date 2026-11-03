"""
privacy_attacks.py — E8 (PROTOCOL §7 Privacy, D-13; R1-M6, R2-5, R4.9). CPU only; reads saved run outputs.

    python privacy_attacks.py --run_root ../runs --out ../results --exps E2_baseline E2_apd E0_full_r2 E1_nopre \
        --pair E2_baseline E2_apd --dp_exp E0_full_r2

Threat model: an honest-but-curious party holds the trained encoder (or its embeddings) and one other scan view
of a target patient, and tries to (1) link that view to the patient (re-identification) or (2) tell whether the
patient was in the training set (membership inference).

Attack 1, retrieval: query = embedding of the MRI-only view (PET/CT zeroed), gallery = PET/CT-only embeddings of
all 51 patients; cosine similarity; rank of the true patient. top-1, top-5, MRR (= mAP with one relevant item).
Held-out queries (each patient once per seed) and training queries (each patient in 4 folds per seed) separately.
Chance: top-1 1/51, top-5 5/51, MRR = H_51 / 51.
Attack 2, membership inference: score = model confidence on the true grade (equivalent in rank to the negative
cross-entropy loss); AUC training-fold vs held-out patients, per run.
APD effect: E2_apd vs E2_baseline (they differ only in APD). Paired patient bootstrap on held-out MRR; exact McNemar
on per-patient held-out top-1 hits (hit = top-1 in at least 2 of 3 seeds).
DP reference: grade head (linear, 256 -> 3) on frozen full-view embeddings of --dp_exp, trained with DP-SGD
(Opacus, epsilon in {1, 8}, delta 1e-5) and without noise (reference); the encoder itself is NOT private.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import evaluate as ev

N_BOOT = 2000


# ── attack 1: retrieval ─────────────────────────────────────────────────────
def retrieval_ranks(Q: np.ndarray, G: np.ndarray) -> np.ndarray:
    """1-based rank of gallery item i for query i under cosine similarity (ties count against the attacker)."""
    Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
    Gn = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-12)
    S = Qn @ Gn.T
    true = np.diag(S)[:, None]
    return (S >= true).sum(1)


def chance_mrr(n: int) -> float:
    return float(np.mean(1.0 / np.arange(1, n + 1)))


def retrieval_table(run_root: Path, exp: str) -> pd.DataFrame:
    rows = []
    for d in sorted((run_root / exp).glob("seed*/fold*")):
        z = np.load(d / "embeddings.npz")
        r = retrieval_ranks(z["mri_only"], z["petct_only"])
        s, k = int(d.parent.name[4:]), int(d.name[4:])
        for pid, tr, rk in zip(z["pids"], z["is_train"], r):
            rows.append({"exp": exp, "seed": s, "fold": k, "pid": str(pid), "is_train": bool(tr), "rank": int(rk),
                         "n_gallery": len(r)})
    if not rows:
        raise SystemExit(f"no embeddings under {run_root / exp}")
    return pd.DataFrame(rows)


def per_patient(T: pd.DataFrame, train: bool) -> pd.DataFrame:
    t = T[T.is_train == train].assign(top1=lambda x: x["rank"] <= 1, top5=lambda x: x["rank"] <= 5,
                                      rr=lambda x: 1.0 / x["rank"])
    return t.groupby("pid").agg(top1=("top1", "mean"), top5=("top5", "mean"), rr=("rr", "mean"),
                                top1_majority=("top1", lambda v: v.mean() >= 0.5), n=("rank", "size")).reset_index()


def summarise_retrieval(T: pd.DataFrame) -> dict:
    n = int(T.n_gallery.iloc[0]); out = {"n_gallery": n, "chance": {"top1": 1 / n, "top5": 5 / n, "mrr": chance_mrr(n)}}
    for name, train in (("held_out", False), ("training", True)):
        P = per_patient(T, train); res = {"n_patients": len(P)}
        for m, col in (("top1", "top1"), ("top5", "top5"), ("mrr", "rr")):
            v = P[col].values
            lo, hi = ev.bootstrap_ci(lambda i: v[i].mean(), len(v), N_BOOT)
            res[m] = {"value": float(v.mean()), "ci95": [lo, hi]}
        out[name] = res
    return out


# ── attack 2: membership inference ──────────────────────────────────────────
def mia_auc(train_pred: pd.DataFrame, oof: pd.DataFrame) -> float:
    cols = ev.PROB_COLS

    def conf(df):
        df = df[df.has_grade == 1]
        return df[cols].values[np.arange(len(df)), df.grade.values.astype(int)]
    a, b = conf(train_pred), conf(oof)
    y = np.r_[np.ones(len(a)), np.zeros(len(b))]
    return float(roc_auc_score(y, np.r_[a, b]))


def mia_table(run_root: Path, exp: str) -> dict:
    aucs = []
    for d in sorted((run_root / exp).glob("seed*/fold*")):
        aucs.append(mia_auc(pd.read_csv(d / "train_predictions.csv"), pd.read_csv(d / "oof_predictions.csv")))
    a = np.array(aucs)
    return {"n_runs": len(a), "auc_mean": float(a.mean()), "auc_sd": float(a.std(ddof=1)),
            "auc_min": float(a.min()), "auc_max": float(a.max()), "runs_above_0.5": int((a > 0.5).sum()),
            "chance": 0.5}


# ── APD paired comparison ───────────────────────────────────────────────────
def paired_apd(T_ref: pd.DataFrame, T_apd: pd.DataFrame) -> dict:
    r = per_patient(T_ref, False).merge(per_patient(T_apd, False), on="pid", suffixes=("_ref", "_apd"))
    a, b = r.rr_apd.values, r.rr_ref.values
    d = ev.paired_bootstrap(lambda i: a[i].mean(), lambda i: b[i].mean(), len(r), N_BOOT)
    ref_only = int((r.top1_majority_ref & ~r.top1_majority_apd).sum())
    apd_only = int((r.top1_majority_apd & ~r.top1_majority_ref).sum())
    return {"held_out_mrr_delta_apd_minus_ref": d,
            "mcnemar_top1": {"ref_only_hit": ref_only, "apd_only_hit": apd_only,
                             "p": ev.mcnemar_exact(ref_only, apd_only)}}


# ── DP-SGD grade head on frozen embeddings ──────────────────────────────────
def train_head(X, y, eps, seed, epochs=100, batch=16, lr=0.1, clip=1.0, delta=1e-5):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    torch.manual_seed(seed); np.random.seed(seed)
    head = nn.Linear(X.shape[1], 3)
    opt = torch.optim.SGD(head.parameters(), lr=lr)
    dl = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)),
                    batch_size=batch, shuffle=True)
    spent = None
    if eps is not None:
        from opacus import PrivacyEngine
        pe = PrivacyEngine(accountant="rdp")
        head, opt, dl = pe.make_private_with_epsilon(module=head, optimizer=opt, data_loader=dl, epochs=epochs,
                                                     target_epsilon=eps, target_delta=delta, max_grad_norm=clip)
    lossf = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for xb, yb in dl:
            if len(xb) == 0:
                continue
            opt.zero_grad(); lossf(head(xb), yb).backward(); opt.step()
    if eps is not None:
        spent = float(pe.get_epsilon(delta))
    return head, spent


def dp_head(run_root: Path, exp: str, eps_list=(None, 8.0, 1.0)) -> tuple:
    import torch
    rows, spent = [], {}
    for d in sorted((run_root / exp).glob("seed*/fold*")):
        z = np.load(d / "embeddings.npz"); oof = pd.read_csv(d / "oof_predictions.csv").set_index("pid")
        trp = pd.read_csv(d / "train_predictions.csv").set_index("pid")
        lab = pd.concat([oof[["grade", "has_grade"]], trp[["grade", "has_grade"]]])
        pids = [str(p) for p in z["pids"]]
        g = lab.reindex(pids); has = g.has_grade.values == 1; tr = z["is_train"]
        X = z["full"]; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; X = (X - mu) / sd
        fit = tr & has; s, k = int(d.parent.name[4:]), int(d.name[4:])
        for eps in eps_list:
            head, e_spent = train_head(X[fit], g.grade.values[fit].astype(int), eps, seed=s * 100 + k)
            spent.setdefault(str(eps), []).append(e_spent)
            with torch.no_grad():
                P = torch.softmax(head(torch.tensor(X, dtype=torch.float32)), 1).numpy()
            for i in np.where(~tr & has)[0]:
                rows.append({"eps": "none" if eps is None else eps, "seed": s, "fold": k, "pid": pids[i],
                             "grade": int(g.grade.values[i]), "p_low": P[i, 0], "p_int": P[i, 1], "p_high": P[i, 2]})
    R = pd.DataFrame(rows); res = {}
    for eps, t in R.groupby("eps"):
        A = t.groupby("pid").agg({"grade": "first", **{c: "mean" for c in ev.PROB_COLS}}).reset_index()
        y, P = A.grade.values, A[ev.PROB_COLS].values
        m = ev.grade_metrics(y, P)
        out = {"n": int(len(y)), "accuracy": m["accuracy"]}
        for key in ("balanced_accuracy", "macro_auc_ovr"):
            fn = (lambda i, key=key: ev.grade_metrics(y[i], P[i])[key])
            lo, hi = ev.bootstrap_ci(fn, len(y), N_BOOT)
            out[key] = {"value": m[key], "ci95": [lo, hi]}
        e = [v for v in spent.get(str(None if eps == "none" else eps), []) if v is not None]
        out["epsilon_spent_max"] = max(e) if e else None
        res[str(eps)] = out
    return res, R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exps", nargs="+", required=True)
    ap.add_argument("--pair", nargs=2, metavar=("REF_NO_APD", "APD"), required=True)
    ap.add_argument("--dp_exp", required=True)
    ap.add_argument("--tag", default="E8_privacy")
    a = ap.parse_args()
    out = Path(a.out); dest = out / f"{a.tag}.json"
    if dest.exists():
        raise SystemExit(f"{dest} exists; refusing to overwrite")
    run_root = Path(a.run_root); res = {"retrieval": {}, "membership_inference": {}}
    tables = {}
    for e in sorted(set(a.exps) | set(a.pair)):
        tables[e] = retrieval_table(run_root, e)
        res["retrieval"][e] = summarise_retrieval(tables[e])
        res["membership_inference"][e] = mia_table(run_root, e)
        print(e, json.dumps({"held_out_top1": res["retrieval"][e]["held_out"]["top1"]["value"],
                             "mia_auc": res["membership_inference"][e]["auc_mean"]}))
    res["apd_paired"] = {"ref": a.pair[0], "apd": a.pair[1], **paired_apd(tables[a.pair[0]], tables[a.pair[1]])}
    res["dp_head"], R = dp_head(run_root, a.dp_exp)
    res["dp_head"]["_exp"] = a.dp_exp
    pd.concat(tables.values()).to_csv(out / f"{a.tag}_retrieval_ranks.csv", index=False)
    R.to_csv(out / f"{a.tag}_dp_head_predictions.csv", index=False)
    dest.write_text(json.dumps(res, indent=1, default=ev._json_default))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
