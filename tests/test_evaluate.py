"""Synthetic tests with known answers for evaluate.py (PLAN Phase 2.3)."""
import numpy as np
import pandas as pd
import pytest

import evaluate as ev


# ── segmentation ─────────────────────────────────────────────────────────────
def test_dice_iou_known():
    a = np.zeros((10, 10, 10), bool); b = np.zeros_like(a)
    a[0:4, 0:5, 0:5] = 1           # 100 voxels
    b[2:6, 0:5, 0:5] = 1           # 100 voxels, overlap 50
    d, j = ev.dice_iou(a, b)
    assert d == pytest.approx(0.5) and j == pytest.approx(50 / 150)


def test_dice_empty_reference_is_nan_not_one():
    z = np.zeros((4, 4, 4), bool)
    d, j = ev.dice_iou(z, z)
    assert np.isnan(d) and np.isnan(j)


def test_surface_metrics_shifted_cube():
    a = np.zeros((30, 30, 30), bool); b = np.zeros_like(a)
    a[5:15, 5:15, 5:15] = 1; b[5:15, 5:15, 8:18] = 1        # shift 3 voxels along x
    hd, asd = ev.surface_metrics(a, a, (1, 1, 1))
    assert hd == 0 and asd == 0
    hd, _ = ev.surface_metrics(a, b, (1, 1, 1))
    assert hd == pytest.approx(3.0)
    hd2, _ = ev.surface_metrics(a, b, (1, 1, 2.0))           # anisotropic spacing doubles x distances
    assert hd2 == pytest.approx(6.0)
    assert np.isnan(ev.surface_metrics(a, np.zeros_like(a), (1, 1, 1))[0])


# ── grading ──────────────────────────────────────────────────────────────────
def test_grade_metrics_perfect_and_majority():
    y = np.array([0, 1, 2, 2, 1, 0, 2])
    P = np.eye(3)[y] * 0.8 + 0.2 / 3
    m = ev.grade_metrics(y, P)
    assert m["accuracy"] == 1 and m["balanced_accuracy"] == 1 and m["macro_auc_ovr"] == 1
    assert m["f1_low"] == 1 and m["specificity_high"] == 1
    P2 = np.tile([0, 0, 1.0], (7, 1))                         # always "high"
    m2 = ev.grade_metrics(y, P2)
    assert m2["accuracy"] == pytest.approx(3 / 7) and m2["balanced_accuracy"] == pytest.approx(1 / 3)
    assert ev.majority_accuracy(y) == pytest.approx(3 / 7)


# ── survival ─────────────────────────────────────────────────────────────────
def test_harrell_c_known():
    t = np.array([1., 2, 3, 4, 5]); e = np.array([1, 1, 1, 1, 0])
    assert ev.harrell_c(t, e, -t) == 1.0
    assert ev.harrell_c(t, e, t) == 0.0
    assert ev.harrell_c(t, e, np.zeros(5)) == 0.5
    # censored subject at time 5 can only be the later member of a pair
    assert ev.harrell_c(np.array([5., 1]), np.array([0, 1]), np.array([9., 0])) == 0.0


def test_harrell_matches_lifelines():
    from lifelines.utils import concordance_index
    rng = np.random.default_rng(3)
    t = rng.integers(1, 30, 60).astype(float); e = rng.uniform(size=60) < 0.4; r = rng.normal(size=60)
    # lifelines: higher score = longer survival, so pass -risk
    assert ev.harrell_c(t, e, r) == pytest.approx(concordance_index(t, -r, e), abs=1e-9)


def test_breslow_survival_monotone_and_ordered():
    t = np.array([1., 2, 3, 4, 6, 8]); e = np.array([1, 0, 1, 1, 0, 1]); r = np.zeros(6)
    bt, bH = ev.breslow_baseline(t, e, r)
    assert np.all(np.diff(bH) > 0) and bH[0] == pytest.approx(1 / 6)
    S = ev.survival_curves(np.array([0.5, 1, 5, 10]), bt, bH, np.array([0.0, 1.0]))
    assert S[0, 0] == 1.0 and np.all(np.diff(S, axis=1) <= 0) and np.all(S[1] <= S[0])


# ── statistics ───────────────────────────────────────────────────────────────
def test_mcnemar_reviewer_example():
    assert ev.mcnemar_exact(3, 0) == pytest.approx(0.25)     # R4: 3/51 vs 0/51 -> p = 0.25
    assert ev.mcnemar_exact(0, 0) == 1.0


def test_holm():
    adj = ev.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": np.nan})
    assert adj["a"] == pytest.approx(0.03) and adj["c"] == pytest.approx(0.06) and adj["b"] == pytest.approx(0.06)
    assert np.isnan(adj["d"])


def test_bootstrap_ci_covers_mean():
    rng = np.random.default_rng(0); x = rng.normal(5, 1, 200)
    lo, hi = ev.bootstrap_ci(lambda i: x[i].mean(), len(x), n_boot=1000)
    assert lo < 5 < hi and hi - lo < 0.5


def test_paired_bootstrap_identical_is_null():
    x = np.random.default_rng(1).normal(size=50)
    r = ev.paired_bootstrap(lambda i: x[i].mean(), lambda i: x[i].mean(), 50, n_boot=300)
    assert r["delta"] == 0 and r["p"] == 1.0


def test_wilcoxon_floor_n5():
    r = ev.wilcoxon_folds([1, 2, 3, 4, 5], [0, 0, 0, 0, 0])
    assert r["p"] == pytest.approx(0.0625)


# ── aggregation ──────────────────────────────────────────────────────────────
def test_rank_normalisation_within_fold():
    oof = pd.DataFrame({"seed": [1, 1, 1, 1], "fold": [0, 0, 1, 1], "pid": list("abcd"),
                        "risk": [10.0, 20.0, -5.0, -1.0]})
    r = ev.add_rank_risk(oof).risk_rank.tolist()
    assert r == [0.0, 1.0, 0.0, 1.0]                        # scale offset between folds removed


def test_aggregate_seeds_averages_probs():
    rows = []
    for s, p in ((1, [0.2, 0.3, 0.5]), (2, [0.4, 0.3, 0.3])):
        rows.append({"seed": s, "fold": 0, "pid": "a", "p_low": p[0], "p_int": p[1], "p_high": p[2],
                     "risk": 1.0, "dice": 0.5, "iou": 0.3, "has_mask": 1, "has_grade": 1, "grade": 2,
                     "time": 100.0, "event": 1})
        rows.append({**rows[-1], "pid": "b", "risk": 0.0})
    P = ev.aggregate_seeds(pd.DataFrame(rows)).set_index("pid")
    assert P.loc["a", "p_low"] == pytest.approx(0.3) and P.loc["a", "n_seeds"] == 2
    assert P.loc["a", "risk_rank"] == 1.0 and P.loc["b", "risk_rank"] == 0.0


def test_survival_from_training_matches_and_is_shift_invariant():
    rng = np.random.default_rng(5)
    tt = rng.integers(10, 400, 40).astype(float); te = rng.uniform(size=40) < 0.35; tr = rng.normal(size=40)
    xr = rng.normal(size=8); grid = np.linspace(20, 380, 25)
    bt, bH = ev.breslow_baseline(tt, te, tr)
    ref = ev.survival_curves(grid, bt, bH, xr)
    got = ev.survival_from_training(grid, tt, te, tr, xr)
    assert np.allclose(ref, got, atol=1e-10)
    big = ev.survival_from_training(grid, tt, te, tr + 5000.0, xr + 5000.0)   # risk-scale drift
    assert np.all(np.isfinite(big)) and np.allclose(big, got, atol=1e-10)


def _toy_P(n=30, seed=0, perfect=False):
    rng = np.random.default_rng(seed)
    y = np.tile([0, 1, 2], n // 3)
    P = np.eye(3)[y] * 0.85 + 0.05 if perfect else rng.dirichlet([1, 1, 1], n)
    return pd.DataFrame({"pid": [f"p{i}" for i in range(n)], "has_mask": 1, "has_grade": 1, "grade": y,
                         "p_low": P[:, 0], "p_int": P[:, 1], "p_high": P[:, 2],
                         "dice": rng.uniform(0.5, 0.9, n), "time": rng.uniform(1, 10, n),
                         "event": (np.arange(n) % 3 == 0).astype(int), "risk_rank": rng.uniform(0, 1, n)})


def test_compare_metric_subset_and_auc_delta():
    ref, oth = _toy_P(seed=1), _toy_P(seed=2, perfect=True)
    out = ev.compare(ref, oth, n_boot=200, metrics=("macro_auc_ovr",))
    assert set(out) == {"macro_auc_ovr"}
    assert out["macro_auc_ovr"]["delta"] == pytest.approx(1.0 - ev._macro_auc(ref.grade.values, ref[ev.PROB_COLS].values))
    full = ev.compare(ref, ref, n_boot=200)
    assert set(full) == {"dice", "balanced_accuracy", "mcnemar_grade", "macro_auc_ovr", "c_pooled_rank_normalised"}
    assert all(full[m]["delta"] == 0 for m in ev.COMPARE_METRICS)
