"""Tests for E10 (padding wrapper, nnU-Net forced splits) and E11 (inner splits, selection score)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from revision_utils import fold_ids, load_splits

ROOT = Path(__file__).resolve().parents[1]


def test_pad_to_multiple_restores_shape_and_content():
    from seg_baselines import PadToMultiple
    seen = {}

    class Ident(torch.nn.Module):
        def forward(self, x):
            seen["shape"] = tuple(x.shape[2:]); return x
    x = torch.randn(2, 3, 48, 96, 96)
    y = PadToMultiple(Ident(), 32)(x)
    assert seen["shape"] == (64, 96, 96) and y.shape == x.shape and torch.equal(y, x)


@pytest.mark.parametrize("k", range(5))
def test_inner_splits_partition_outer_training_fold(k):
    sp = load_splits(ROOT / "splits.json"); tr, va = fold_ids(sp, 42, k)
    inner = load_splits(ROOT / "splits_inner" / f"outer{k}.json")
    assert sorted(inner["patients"]) == sorted(tr)
    assert not set(inner["patients"]) & set(va)
    assert inner["outer"]["splits_sha256"] == sp["_sha256"]
    seen = []
    for j in range(4):
        itr, iva = fold_ids(inner, 42, j)
        assert not set(itr) & set(iva) and len(iva) >= 10
        seen += iva
    assert sorted(seen) == sorted(tr)


def test_e11_score_known_values():
    import e11_nested
    df = pd.DataFrame({"seed": 42, "fold": [0, 0, 1, 1], "pid": list("abcd"), "has_mask": 1, "has_grade": 1,
                       "grade": [0, 1, 2, 2], "p_low": [1, 0, 0, 0], "p_int": [0, 1, 0, 0], "p_high": [0, 0, 1, 1],
                       "dice": [0.6, 0.8, 0.7, 0.9], "risk": [3.0, 1.0, 2.0, 1.0], "time": [1, 5, 2, 6],
                       "event": [1, 0, 1, 0]})
    s = e11_nested.score(df)
    assert s["dice"] == pytest.approx(0.75) and s["macro_auc_ovr"] == pytest.approx(1.0)
    assert s["c_pooled_rank"] == pytest.approx(0.9)   # 5 comparable pairs, one rank tie across folds
    assert s["S"] == pytest.approx((0.75 + 1 + 0.9) / 3)


def test_e10_and_e11_configs_differ_from_E0_only_where_stated():
    e0 = json.loads((ROOT / "configs/E0.json").read_text())
    for f in sorted((ROOT / "configs").glob("E10_*.json")) + sorted((ROOT / "configs/E11").glob("o*_E*.json")):
        c = json.loads(f.read_text())
        diff = {k for k in set(c) | set(e0) if c.get(k) != e0.get(k)} - {"_comment"}
        allowed = ({"exp_id", "seg_net", "tasks", "pretrain_mode", "use_apd", "use_multiscale", "use_clinical",
                    "save_final"} if f.name.startswith("E10") else {"exp_id", "splits", "epochs", "encoder_root",
                                                                   "save_final"})
        assert diff <= allowed, (f.name, diff - allowed)
