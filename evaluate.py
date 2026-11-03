"""
evaluate.py — every reported number comes from here (PROTOCOL.md §6–§7, frozen 2026-10-29).

Inputs: <run_root>/<exp_id>/seed<s>/fold<k>/{oof_predictions.csv, train_predictions.csv, oof_seg/*.npz}
        plus the cache (GT masks, meta.json spacing) for surface metrics.

    python evaluate.py summary --run_root ../runs --exp E0_full --cache_dir ../cache_48x96x96 --out results
    python evaluate.py compare --run_root ../runs --ref E0_full --exps E2_baseline E2_crmae ... --out results

Rules implemented:
  * unit of analysis = patient; folds and seeds are not independent samples
  * seeds aggregated per patient: class probabilities averaged; risk -> rank within
    its (seed, fold), then averaged (primary, D-12); raw risk also averaged (secondary)
  * segmentation metrics only for has_mask == 1 (SEG set), grading only for has_grade == 1
  * 95% CIs: patient bootstrap, 2000 resamples, percentile; stratified by event for survival
  * comparisons: paired patient bootstrap of the difference, two-sided p, Holm across a family
  * Breslow baseline hazard and the KM threshold come from each fold's TRAINING patients
  * KM groups: per seed, high risk if risk > median training-fold risk; the reported
    grouping is the majority over seeds (operationalisation of PROTOCOL §7, recorded here)
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage, stats
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

N_BOOT_DEFAULT = 2000
HORIZONS_DAYS  = (365.0, 730.0)
UNO_TAU_DAYS   = 1095.0
CLASSES        = ("low", "int", "high")
PROB_COLS      = ["p_low", "p_int", "p_high"]


# ════════════════════════════════════════════════════════════════════════════
# Segmentation
# ════════════════════════════════════════════════════════════════════════════

def dice_iou(pred: np.ndarray, gt: np.ndarray):
    p, g = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(p, g).sum(); s = p.sum() + g.sum(); u = np.logical_or(p, g).sum()
    return (2 * inter / s if s else np.nan), (inter / u if u else np.nan)


def _surface(m: np.ndarray) -> np.ndarray:
    return m & ~ndimage.binary_erosion(m, structure=ndimage.generate_binary_structure(3, 1), border_value=0)


def surface_metrics(pred: np.ndarray, gt: np.ndarray, spacing) -> tuple:
    """(HD95, ASSD) in the units of `spacing` (mm). NaN if either mask is empty."""
    p, g = pred.astype(bool), gt.astype(bool)
    if not p.any() or not g.any():
        return np.nan, np.nan
    sp, sg = _surface(p), _surface(g)
    dt_g = ndimage.distance_transform_edt(~sg, sampling=spacing)
    dt_p = ndimage.distance_transform_edt(~sp, sampling=spacing)
    d_pg, d_gp = dt_g[sp], dt_p[sg]
    both = np.concatenate([d_pg, d_gp])
    return float(np.percentile(both, 95)), float(both.mean())


# ════════════════════════════════════════════════════════════════════════════
# Grading
# ════════════════════════════════════════════════════════════════════════════

def grade_metrics(y: np.ndarray, P: np.ndarray) -> dict:
    y = np.asarray(y, int); P = np.asarray(P, float); yhat = P.argmax(1)
    out = {"n": int(len(y)), "accuracy": float((yhat == y).mean()),
           "balanced_accuracy": float(balanced_accuracy_score(y, yhat))}
    out["macro_auc_ovr"] = (float(roc_auc_score(y, P, multi_class="ovr", average="macro", labels=[0, 1, 2]))
                            if len(np.unique(y)) == 3 else np.nan)
    hi = (y == 2).astype(int)
    out["auc_high_vs_rest"] = float(roc_auc_score(hi, P[:, 2])) if 0 < hi.sum() < len(hi) else np.nan
    cm = confusion_matrix(y, yhat, labels=[0, 1, 2])
    for c, name in enumerate(CLASSES):
        tp = cm[c, c]; fn = cm[c].sum() - tp; fp = cm[:, c].sum() - tp; tn = cm.sum() - tp - fn - fp
        prec = tp / (tp + fp) if tp + fp else np.nan
        rec  = tp / (tp + fn) if tp + fn else np.nan
        out[f"precision_{name}"]   = float(prec)
        out[f"recall_{name}"]      = float(rec)
        out[f"specificity_{name}"] = float(tn / (tn + fp)) if tn + fp else np.nan
        out[f"f1_{name}"] = float(2 * prec * rec / (prec + rec)) if (prec + rec) and not np.isnan(prec + rec) else np.nan
    out["confusion_matrix"] = cm.tolist()
    return out


def majority_accuracy(y) -> float:
    y = np.asarray(y, int)
    return float(np.bincount(y, minlength=3).max() / len(y))


# ════════════════════════════════════════════════════════════════════════════
# Survival
# ════════════════════════════════════════════════════════════════════════════

def harrell_c(time, event, risk) -> float:
    """Harrell's C. Comparable pair: event_i and (t_j > t_i, or t_j == t_i with j censored);
    tied event times are not comparable; risk ties count 0.5 (matches lifelines/sksurv)."""
    t = np.asarray(time, float); e = np.asarray(event, bool); r = np.asarray(risk, float)
    num = den = 0.0
    for i in np.where(e)[0]:
        j = (t > t[i]) | ((t == t[i]) & ~e)
        den += j.sum(); num += (r[i] > r[j]).sum() + 0.5 * (r[i] == r[j]).sum()
    return float(num / den) if den else np.nan


def _surv_struct(time, event):
    from sksurv.util import Surv
    return Surv.from_arrays(event=np.asarray(event, bool), time=np.asarray(time, float))


def uno_c(time, event, risk, tau=UNO_TAU_DAYS) -> float:
    from sksurv.metrics import concordance_index_ipcw
    s = _surv_struct(time, event)
    try:
        return float(concordance_index_ipcw(s, s, np.asarray(risk, float), tau=tau)[0])
    except ValueError:
        return np.nan


def td_auc(time, event, risk, horizons=HORIZONS_DAYS) -> dict:
    from sksurv.metrics import cumulative_dynamic_auc
    s = _surv_struct(time, event)
    h = [x for x in horizons if np.asarray(time).min() < x < np.asarray(time).max()]
    if not h:
        return {}
    auc, _ = cumulative_dynamic_auc(s, s, np.asarray(risk, float), np.array(h))
    return {f"td_auc_{int(x)}d": float(a) for x, a in zip(h, auc)}


def breslow_baseline(train_time, train_event, train_risk):
    """Breslow cumulative baseline hazard H0(t) as a step function (times, H0)."""
    t = np.asarray(train_time, float); e = np.asarray(train_event, bool); r = np.exp(np.asarray(train_risk, float))
    ut = np.unique(t[e])
    H, acc = [], 0.0
    for u in ut:
        acc += e[t == u].sum() / r[t >= u].sum()
        H.append(acc)
    return ut, np.array(H)


def survival_curves(times_grid, base_t, base_H, risk):
    idx = np.searchsorted(base_t, times_grid, side="right") - 1
    H = np.where(idx >= 0, base_H[np.clip(idx, 0, None)], 0.0)
    return np.exp(-np.outer(np.exp(np.asarray(risk, float)), H))


def survival_from_training(times_grid, train_time, train_event, train_risk, test_risk):
    """
    Breslow survival curves for test patients, computed in log space so that the
    result is invariant to a common shift of all risk scores and never overflows:
        H_i(t) = sum_{event times u <= t} d_u * exp(r_i - logsumexp_{j in R(u)} r_j)
    (identical to survival_curves(breslow_baseline(...)) whenever that does not overflow).
    """
    from scipy.special import logsumexp
    t = np.asarray(train_time, float); e = np.asarray(train_event, bool); r = np.asarray(train_risk, float)
    ut = np.unique(t[e])
    log_d = np.log([e[t == u].sum() for u in ut])
    lse   = np.array([logsumexp(r[t >= u]) for u in ut])
    rt = np.asarray(test_risk, float)
    logterm = log_d[None, :] + rt[:, None] - lse[None, :]                 # (n_test, n_event_times)
    inc = np.exp(np.minimum(logterm, 700.0))
    cumH = np.cumsum(inc, axis=1)
    idx = np.searchsorted(ut, np.asarray(times_grid, float), side="right") - 1
    H = np.where(idx[None, :] >= 0, cumH[:, np.clip(idx, 0, None)], 0.0)
    return np.exp(-H)


# ════════════════════════════════════════════════════════════════════════════
# Statistics
# ════════════════════════════════════════════════════════════════════════════

def bootstrap_ci(fn, n: int, n_boot=N_BOOT_DEFAULT, strata=None, seed=0):
    """Percentile CI of fn(idx) over patient resamples (optionally stratified)."""
    rng = np.random.default_rng(seed)
    groups = [np.arange(n)] if strata is None else [np.where(strata == s)[0] for s in np.unique(strata)]
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        v = fn(idx)
        if v is not None and not np.isnan(v):
            vals.append(v)
    if len(vals) < 0.5 * n_boot:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_bootstrap(fn_a, fn_b, n: int, n_boot=N_BOOT_DEFAULT, strata=None, seed=0) -> dict:
    """Δ = fn_a − fn_b on the same patients; percentile CI and two-sided bootstrap p."""
    rng = np.random.default_rng(seed)
    all_idx = np.arange(n)
    groups = [all_idx] if strata is None else [np.where(strata == s)[0] for s in np.unique(strata)]
    obs = fn_a(all_idx) - fn_b(all_idx)
    d = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        v = fn_a(idx) - fn_b(idx)
        if not np.isnan(v):
            d.append(v)
    d = np.array(d)
    if len(d) < 0.5 * n_boot:
        return {"delta": float(obs), "ci_low": np.nan, "ci_high": np.nan, "p": np.nan}
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return {"delta": float(obs), "ci_low": float(np.percentile(d, 2.5)),
            "ci_high": float(np.percentile(d, 97.5)), "p": float(min(1.0, max(p, 1.0 / len(d))))}


def holm(pvals: dict) -> dict:
    keys = [k for k, v in pvals.items() if not np.isnan(v)]
    order = sorted(keys, key=lambda k: pvals[k]); m = len(order); adj = {}; running = 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        adj[k] = running
    return {k: adj.get(k, np.nan) for k in pvals}


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p from discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    return float(min(1.0, 2 * stats.binom.cdf(min(b, c), n, 0.5)))


def wilcoxon_folds(a, b) -> dict:
    """Descriptive only: n = 5 folds can never give two-sided p < 0.0625."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    try:
        r = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        return {"statistic": float(r.statistic), "p": float(r.pvalue), "n_folds": int(len(a)),
                "note": "descriptive; minimum attainable two-sided p with n=5 is 0.0625"}
    except ValueError as e:
        return {"error": str(e), "n_folds": int(len(a))}


# ════════════════════════════════════════════════════════════════════════════
# Loading and aggregation
# ════════════════════════════════════════════════════════════════════════════

def load_runs(run_root: Path, exp: str, splits: dict = None):
    base = Path(run_root) / exp
    oof, tr = [], []
    for d in sorted(base.glob("seed*/fold*")):
        f = d / "oof_predictions.csv"
        if not f.exists():
            raise SystemExit(f"incomplete run: {d} has no oof_predictions.csv")
        o = pd.read_csv(f); o["run_dir"] = str(d); oof.append(o)
        t = pd.read_csv(d / "train_predictions.csv"); t["seed"] = o.seed.iloc[0]; t["fold"] = o.fold.iloc[0]; tr.append(t)
    if not oof:
        raise SystemExit(f"no runs under {base}")
    oof = pd.concat(oof, ignore_index=True); tr = pd.concat(tr, ignore_index=True)
    for s, g in oof.groupby("seed"):
        if g.pid.duplicated().any():
            raise SystemExit(f"seed {s}: patient predicted twice")
        if splits is not None and set(g.pid) != set(splits["patients"]):
            raise SystemExit(f"seed {s}: {len(set(g.pid))}/{len(splits['patients'])} patients predicted")
    return oof, tr


def add_rank_risk(oof: pd.DataFrame) -> pd.DataFrame:
    oof = oof.copy()
    oof["risk_rank"] = oof.groupby(["seed", "fold"]).risk.transform(
        lambda r: (r.rank(method="average") - 1) / max(len(r) - 1, 1))
    return oof


def aggregate_seeds(oof: pd.DataFrame) -> pd.DataFrame:
    oof = add_rank_risk(oof)
    agg = {c: "mean" for c in PROB_COLS + ["risk", "risk_rank", "dice", "iou"] + [c for c in ("hd95", "assd") if c in oof]}
    agg.update({c: "first" for c in ("has_mask", "has_grade", "grade", "time", "event")})
    out = oof.groupby("pid").agg(agg).reset_index()
    out["n_seeds"] = oof.groupby("pid").seed.nunique().values
    return out


def add_surface_metrics(oof: pd.DataFrame, cache_dir: Path) -> pd.DataFrame:
    oof = oof.copy(); hd, asd = [], []
    for _, r in oof.iterrows():
        if r.has_mask != 1:
            hd.append(np.nan); asd.append(np.nan); continue
        pred = np.load(Path(r.run_dir) / "oof_seg" / f"{r.pid}.npz")["pred"]
        gt   = np.load(Path(cache_dir) / r.pid / "mask.npy")
        sp   = json.loads((Path(cache_dir) / r.pid / "meta.json").read_text())["resampled_spacing_mm"]
        h, a = surface_metrics(pred, gt, sp); hd.append(h); asd.append(a)
    oof["hd95"] = hd; oof["assd"] = asd
    return oof


def km_groups(oof: pd.DataFrame, tr: pd.DataFrame) -> pd.Series:
    thr = tr.groupby(["seed", "fold"]).risk.median().rename("thr").reset_index()
    o = oof.merge(thr, on=["seed", "fold"])
    o["high"] = (o.risk > o.thr).astype(int)
    return o.groupby("pid").high.mean().gt(0.5).astype(int)


def ibs_per_seed(oof: pd.DataFrame, tr: pd.DataFrame) -> float:
    """Integrated Brier score over [first observed time, 2 y], Breslow baseline from training folds, mean over seeds."""
    from sksurv.metrics import integrated_brier_score
    vals = []
    for s, g in oof.groupby("seed"):
        sv = _surv_struct(g.time, g.event)
        grid = np.linspace(g.time.min(), min(HORIZONS_DAYS[-1], g.time.max() - 1), 50)   # sksurv: inside follow-up
        S = np.zeros((len(g), len(grid)))
        for k, gk in g.groupby("fold"):
            tk = tr[(tr.seed == s) & (tr.fold == k)]
            S[g.index.get_indexer(gk.index)] = survival_from_training(grid, tk.time, tk.event, tk.risk, gk.risk)
        try:
            vals.append(float(integrated_brier_score(sv, sv, S, grid)))
        except ValueError:
            pass
    return float(np.mean(vals)) if vals else np.nan


# ════════════════════════════════════════════════════════════════════════════
# Summary for one experiment
# ════════════════════════════════════════════════════════════════════════════

def summarise(oof, tr, n_boot=N_BOOT_DEFAULT) -> dict:
    P = aggregate_seeds(oof)
    res = {"n_patients": int(len(P)), "n_seeds": int(oof.seed.nunique())}

    seg = P[P.has_mask == 1].reset_index(drop=True)
    res["segmentation"] = {"n": int(len(seg))}
    for m in ("dice", "iou", "hd95", "assd"):
        if m not in seg or len(seg) == 0:      # e.g. non-imaging baselines
            continue
        if m not in seg:
            continue
        v = seg[m].values; ok = ~np.isnan(v)
        res["segmentation"][m] = {"mean": float(np.nanmean(v)), "median": float(np.nanmedian(v)),
                                  "iqr": [float(np.nanpercentile(v, 25)), float(np.nanpercentile(v, 75))],
                                  "n_valid": int(ok.sum()),
                                  "ci95": bootstrap_ci(lambda i: np.nanmean(v[i]), len(v), n_boot)}

    gr = P[P.has_grade == 1].reset_index(drop=True)
    y, Pm = gr.grade.values.astype(int), gr[PROB_COLS].values
    gm = grade_metrics(y, Pm)
    for k in ("accuracy", "balanced_accuracy", "macro_auc_ovr", "auc_high_vs_rest") + tuple(
            f"{m}_{c}" for m in ("precision", "recall", "specificity", "f1") for c in CLASSES):
        gm[k] = {"value": gm[k], "ci95": bootstrap_ci(lambda i, k=k: grade_metrics(y[i], Pm[i])[k], len(y), n_boot)}
    gm["majority_class_accuracy"] = majority_accuracy(y)
    res["grading"] = gm

    t, e = P.time.values, P.event.values.astype(int)
    sv = {"n": int(len(P)), "events": int(e.sum())}
    for name, col in (("c_pooled_rank_normalised", "risk_rank"), ("c_pooled_raw", "risk")):
        r = P[col].values
        sv[name] = {"value": harrell_c(t, e, r),
                    "ci95": bootstrap_ci(lambda i, r=r: harrell_c(t[i], e[i], r[i]), len(t), n_boot, strata=e)}
    sv["uno_c_tau3y"] = uno_c(t, e, P.risk_rank.values)
    sv.update(td_auc(t, e, P.risk_rank.values))
    sv["ibs_to_2y"] = ibs_per_seed(oof.reset_index(drop=True), tr)
    per_fold = []
    for (s, k), g in oof.groupby(["seed", "fold"]):
        per_fold.append({"seed": int(s), "fold": int(k), "events": int(g.event.sum()),
                         "c_index": harrell_c(g.time, g.event, g.risk)})
    sv["per_fold"] = per_fold
    try:
        from lifelines.statistics import logrank_test
        grp = km_groups(oof, tr).reindex(P.pid).values
        lr = logrank_test(t[grp == 1], t[grp == 0], e[grp == 1], e[grp == 0])
        sv["km_logrank"] = {"n_high": int(grp.sum()), "n_low": int((1 - grp).sum()),
                            "events_high": int(e[grp == 1].sum()), "events_low": int(e[grp == 0].sum()),
                            "p": float(lr.p_value)}
    except Exception as ex:   # noqa: BLE001
        sv["km_logrank"] = {"error": str(ex)}
    res["survival"] = sv

    per_seed = []
    for s, g in oof.groupby("seed"):
        gg = g[g.has_grade == 1]
        per_seed.append({"seed": int(s), "dice": float(g.loc[g.has_mask == 1, "dice"].mean()),
                         "accuracy": float((gg[PROB_COLS].values.argmax(1) == gg.grade.values).mean()),
                         "c_pooled_rank": harrell_c(g.time, g.event, add_rank_risk(g).risk_rank)})
    res["per_seed"] = per_seed
    return res, P


COMPARE_METRICS = ("dice", "balanced_accuracy", "macro_auc_ovr", "c_pooled_rank_normalised")


def _macro_auc(y, P):
    if len(np.unique(y)) < 3:
        return np.nan
    try:
        return float(roc_auc_score(y, P, multi_class="ovr", average="macro", labels=[0, 1, 2]))
    except ValueError:
        return np.nan


def compare(ref_P: pd.DataFrame, other_P: pd.DataFrame, n_boot=N_BOOT_DEFAULT, metrics=COMPARE_METRICS) -> dict:
    """Paired patient bootstrap, other − ref, for the primary metric of each task (PROTOCOL §7).
    `metrics` restricts the tests to the tasks a variant was trained for (e.g. single-task E9 models)."""
    out = {}
    if "dice" in metrics:
        s = ref_P[ref_P.has_mask == 1][["pid", "dice"]].merge(other_P[["pid", "dice"]], on="pid", suffixes=("_r", "_o"))
        a, b = s.dice_o.values, s.dice_r.values
        out["dice"] = paired_bootstrap(lambda i: np.nanmean(a[i]), lambda i: np.nanmean(b[i]), len(s), n_boot)
    if {"balanced_accuracy", "macro_auc_ovr"} & set(metrics):
        g = ref_P[ref_P.has_grade == 1][["pid", "grade"] + PROB_COLS].merge(
            other_P[["pid"] + PROB_COLS], on="pid", suffixes=("_r", "_o"))
        y = g.grade.values.astype(int)
        Pr, Po = g[[c + "_r" for c in PROB_COLS]].values, g[[c + "_o" for c in PROB_COLS]].values
        if "balanced_accuracy" in metrics:
            out["balanced_accuracy"] = paired_bootstrap(lambda i: balanced_accuracy_score(y[i], Po[i].argmax(1)),
                                                        lambda i: balanced_accuracy_score(y[i], Pr[i].argmax(1)),
                                                        len(y), n_boot)
            cr, co = (Pr.argmax(1) == y), (Po.argmax(1) == y)
            out["mcnemar_grade"] = {"ref_only_correct": int((cr & ~co).sum()), "other_only_correct": int((co & ~cr).sum()),
                                    "p": mcnemar_exact(int((cr & ~co).sum()), int((co & ~cr).sum()))}
        if "macro_auc_ovr" in metrics:
            out["macro_auc_ovr"] = paired_bootstrap(lambda i: _macro_auc(y[i], Po[i]), lambda i: _macro_auc(y[i], Pr[i]),
                                                    len(y), n_boot)
    if "c_pooled_rank_normalised" in metrics:
        v = ref_P[["pid", "time", "event", "risk_rank"]].merge(other_P[["pid", "risk_rank"]], on="pid", suffixes=("_r", "_o"))
        t, e = v.time.values, v.event.values.astype(int)
        out["c_pooled_rank_normalised"] = paired_bootstrap(lambda i: harrell_c(t[i], e[i], v.risk_rank_o.values[i]),
                                                           lambda i: harrell_c(t[i], e[i], v.risk_rank_r.values[i]),
                                                           len(v), n_boot, strata=e)
    return out


def _json_default(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return None if np.isnan(o) else float(o)
    if isinstance(o, tuple): return list(o)
    return str(o)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("summary"); a1.add_argument("--exp", required=True)
    a2 = sub.add_parser("compare"); a2.add_argument("--ref", required=True); a2.add_argument("--exps", nargs="+", required=True)
    a2.add_argument("--tag", required=True, help="family name; Holm is applied across --exps within this family")
    a2.add_argument("--metrics", nargs="+", default=list(COMPARE_METRICS), choices=list(COMPARE_METRICS))
    a2.add_argument("--ref_seeds", nargs="+", type=int, default=None,
                    help="restrict the reference to these seeds (like-for-like vs single-seed experiments)")
    for p in (a1, a2):
        p.add_argument("--run_root", required=True); p.add_argument("--cache_dir", required=True)
        p.add_argument("--splits", default="splits.json"); p.add_argument("--out", default="results")
        p.add_argument("--n_boot", type=int, default=N_BOOT_DEFAULT)
    a = ap.parse_args()
    sp = json.loads(Path(a.splits).read_text())
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    def load(exp):
        oof, tr = load_runs(Path(a.run_root), exp, sp)
        return add_surface_metrics(oof, Path(a.cache_dir)), tr

    if a.cmd == "summary":
        oof, tr = load(a.exp)
        res, P = summarise(oof, tr, a.n_boot)
        (out / f"{a.exp}_summary.json").write_text(json.dumps(res, indent=1, default=_json_default))
        P.to_csv(out / f"{a.exp}_per_patient.csv", index=False)
        oof.to_csv(out / f"{a.exp}_oof_all.csv", index=False)
        print(json.dumps({k: res[k] for k in ("n_patients", "n_seeds")}, default=_json_default))
    else:
        dest = out / f"compare_{a.tag}.json"
        if dest.exists():
            raise SystemExit(f"{dest} exists; refusing to overwrite")
        ref_oof, _ = load(a.ref)
        if a.ref_seeds:
            ref_oof = ref_oof[ref_oof.seed.isin(a.ref_seeds)]
            if set(ref_oof.seed.unique()) != set(a.ref_seeds):
                raise SystemExit(f"{a.ref}: seeds {sorted(ref_oof.seed.unique())} found, {a.ref_seeds} requested")
        ref_P = aggregate_seeds(ref_oof)
        res = {"_family": {"tag": a.tag, "ref": a.ref, "ref_seeds": a.ref_seeds or "all", "exps": a.exps,
                           "metrics": a.metrics, "n_boot": a.n_boot,
                           "direction": "delta = exp - ref", "holm": "across exps, per metric"}}
        for e in a.exps:
            o, _ = load(e)
            res[e] = compare(ref_P, aggregate_seeds(o), a.n_boot, a.metrics)
        for metric in a.metrics:
            adj = holm({e: res[e][metric]["p"] for e in a.exps})
            for e in a.exps:
                res[e][metric]["p_holm"] = adj.get(e, np.nan)
        dest.write_text(json.dumps(res, indent=1, default=_json_default))
        print(f"wrote {dest}")


if __name__ == "__main__":
    main()
