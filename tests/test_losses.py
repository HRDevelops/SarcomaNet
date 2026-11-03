"""Unit tests for the revised losses (run: python -m pytest tests -q)."""
import math

import numpy as np
import torch

from SarcomaNet import MultiTaskLoss, RiskMemoryBank, SarcomaNet, cox_breslow_nll


def brute_breslow(r, t, e):
    """Reference: -(1/D) sum_{i: e_i=1} [r_i - log sum_{j: t_j >= t_i} exp r_j]."""
    tot, d = 0.0, 0
    for i in range(len(r)):
        if e[i]:
            tot += r[i] - math.log(sum(math.exp(r[j]) for j in range(len(r)) if t[j] >= t[i]))
            d += 1
    return -tot / d


def test_cox_matches_bruteforce_with_ties_and_censoring():
    rng = np.random.default_rng(0)
    r = rng.normal(size=12); t = np.array([5, 3, 3, 9, 1, 3, 7, 7, 2, 8, 4, 6.0])
    e = np.array([1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1])
    got, skipped = cox_breslow_nll(torch.tensor(r), torch.tensor(t), torch.tensor(e))
    assert not skipped
    assert abs(got.item() - brute_breslow(r, t, e)) < 1e-5


def test_cox_bank_equals_full_set_value():
    rng = np.random.default_rng(1)
    r = rng.normal(size=10); t = rng.uniform(1, 100, 10); e = (rng.uniform(size=10) < 0.4).astype(float)
    e[0] = 1
    b = [0, 1, 2]; o = [3, 4, 5, 6, 7, 8, 9]
    got, _ = cox_breslow_nll(torch.tensor(r[b]), torch.tensor(t[b]), torch.tensor(e[b]),
                             torch.tensor(r[o]), torch.tensor(t[o]))
    # events only from the batch, risk sets from everybody
    ref = -np.mean([r[i] - math.log(sum(math.exp(r[j]) for j in range(10) if t[j] >= t[i]))
                    for i in b if e[i]])
    assert abs(got.item() - ref) < 1e-5


def test_cox_no_event_is_zero_and_differentiable():
    r = torch.randn(4, requires_grad=True)
    loss, skipped = cox_breslow_nll(r, torch.tensor([1., 2, 3, 4]), torch.zeros(4))
    assert skipped and loss.item() == 0.0
    loss.backward()
    assert torch.all(r.grad == 0)


def test_cox_gradient_only_through_batch():
    r = torch.randn(3, requires_grad=True); bank = torch.randn(5, requires_grad=True)
    loss, _ = cox_breslow_nll(r, torch.tensor([1., 5, 9]), torch.tensor([1., 0, 1]),
                              bank, torch.tensor([2., 3, 4, 6, 8]))
    loss.backward()
    assert r.grad is not None and bank.grad is None


def test_memory_bank_excludes_batch():
    mb = RiskMemoryBank(["a", "b", "c"], [1.0, 2.0, 3.0])
    mb.update(["b"], torch.tensor([0.7]))
    rr, tt = mb.others(["a"], "cpu")
    assert rr.tolist() == [0.699999988079071, 0.0] and tt.tolist() == [2.0, 3.0]


def _targets(B, has_mask, has_grade, events):
    return {"seg_mask": torch.zeros(B, 16, 16, 16, dtype=torch.long),
            "has_mask": torch.tensor(has_mask, dtype=torch.float32),
            "has_grade": torch.tensor(has_grade, dtype=torch.float32),
            "cls_label": torch.zeros(B, dtype=torch.long),
            "surv_time": torch.tensor([1., 2, 3, 4][:B]),
            "surv_event": torch.tensor(events, dtype=torch.float32)}


def test_masked_task_losses_ignore_excluded_samples():
    torch.manual_seed(0)
    preds = {"seg": torch.randn(4, 2, 16, 16, 16, requires_grad=True),
             "cls": torch.randn(4, 3, requires_grad=True), "surv": torch.randn(4, requires_grad=True)}
    lf = MultiTaskLoss(learnable_weights=True)
    _, b1 = lf(preds, _targets(4, [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 1, 0]))
    # changing the excluded samples' targets must not change seg/cls losses
    t2 = _targets(4, [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 1, 0])
    t2["seg_mask"][2:] = 1; t2["cls_label"][1] = 2; t2["cls_label"][3] = 2
    _, b2 = lf(preds, t2)
    assert torch.isclose(b1["seg"], b2["seg"]) and torch.isclose(b1["cls"], b2["cls"])
    assert b1["n_seg"] == 2 and b1["n_cls"] == 2


def test_no_eligible_samples_gives_zero_task_loss():
    preds = {"seg": torch.randn(2, 2, 16, 16, 16, requires_grad=True),
             "cls": torch.randn(2, 3, requires_grad=True), "surv": torch.randn(2, requires_grad=True)}
    total, b = MultiTaskLoss()(preds, _targets(2, [0, 0], [0, 0], [0, 0]))
    assert b["seg"].item() == 0 and b["cls"].item() == 0 and b["surv"].item() == 0
    total.backward()


def test_model_forward_shapes_paper_spec():
    m = SarcomaNet(in_channels=3, n_seg_classes=2, n_cls_classes=3, clinical_dim=10, base_ch=4)
    out = m(torch.randn(2, 3, 16, 32, 32), torch.randn(2, 10))
    assert out["seg"].shape == (2, 2, 16, 32, 32)
    assert out["cls"].shape == (2, 3) and out["surv"].shape == (2,)
    # paper-spec deep heads (256 -> 64)
    first = lambda seq: next(l for l in seq if isinstance(l, torch.nn.Linear))
    assert first(m.cls_head.fc).out_features == 256 and first(m.surv_head.fc).out_features == 256


def test_single_task_ignores_other_losses():
    torch.manual_seed(1)
    preds = {"seg": torch.randn(2, 2, 16, 16, 16, requires_grad=True),
             "cls": torch.randn(2, 3, requires_grad=True), "surv": torch.randn(2, requires_grad=True)}
    t = _targets(2, [1, 1], [1, 1], [1, 1])
    total, b = MultiTaskLoss(tasks=("seg",))(preds, t)
    assert torch.isclose(total, b["seg"].float())   # exp(-0)*L + 0 for the seg task only
    total.backward()
    assert preds["cls"].grad is None or torch.all(preds["cls"].grad == 0)
