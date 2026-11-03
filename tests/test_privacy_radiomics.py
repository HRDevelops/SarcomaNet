"""Known-answer tests for privacy_attacks.py (E8) and baselines_radiomics.prepare (E7)."""
import numpy as np
import pandas as pd
import pytest

import privacy_attacks as pa
from baselines_radiomics import prepare


def test_retrieval_identical_views_rank_one():
    rng = np.random.default_rng(0); E = rng.normal(size=(20, 8))
    assert (pa.retrieval_ranks(E, E) == 1).all()


def test_retrieval_known_rank_and_ties_count_against_attacker():
    Q = np.array([[1.0, 0.0], [0.0, 1.0]])
    G = np.array([[0.0, 1.0], [0.0, 1.0]])       # query 0's true match is orthogonal; query 1 ties with item 0
    assert list(pa.retrieval_ranks(Q, G)) == [2, 2]


def test_chance_mrr():
    assert pa.chance_mrr(1) == 1.0
    assert pa.chance_mrr(4) == pytest.approx((1 + 1 / 2 + 1 / 3 + 1 / 4) / 4)


def test_mia_auc_perfect_and_null():
    def df(conf_true):
        n = len(conf_true); P = np.full((n, 3), 0.0); P[:, 2] = conf_true; P[:, 0] = 1 - np.asarray(conf_true)
        return pd.DataFrame({"has_grade": 1, "grade": 2, "p_low": P[:, 0], "p_int": P[:, 1], "p_high": P[:, 2]})
    assert pa.mia_auc(df([0.9, 0.8]), df([0.2, 0.1])) == 1.0
    assert pa.mia_auc(df([0.5, 0.5]), df([0.5, 0.5])) == 0.5


def test_radiomics_prepare_imputes_and_drops_correlated():
    rng = np.random.default_rng(1); a = rng.normal(size=10)
    F = pd.DataFrame({"pid": [f"p{i}" for i in range(10)], "f1": a, "f2": 2 * a + 1e-6,   # f2 duplicates f1
                      "f3": rng.normal(size=10), "const": 1.0}).set_index("pid", drop=False)
    F.loc["p9", ["f1", "f2", "f3", "const"]] = np.nan                                       # no mask -> no features
    tr, va = [f"p{i}" for i in range(7)], ["p7", "p8", "p9"]
    Xtr, Xva, names, imp_tr, imp_va = prepare(F, tr, va)
    assert names == ["f1", "f3"]
    assert list(imp_va) == [False, False, True] and not imp_tr.any()
    assert np.allclose(Xtr.mean(0), 0) and not np.isnan(Xva).any()
