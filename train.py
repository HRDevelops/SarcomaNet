"""
train.py — one (experiment, seed, fold) of the array-revision protocol (PROTOCOL.md, frozen 2026-10-29).

    python train.py --config configs/E0.json --seed 42 --fold 0

Protocol rules implemented here (BUGS.md B1, B2, B3, B6, B8, B13, B17):
  * fixed budget: exactly cfg["epochs"] epochs, no early stopping
  * ONE checkpoint = the final epoch; all tasks are read from it
  * the validation fold is used only for logging curves and for the final OOF predictions
  * folds come from splits.json; pretraining encoder = this (seed, fold)'s within-fold encoder
  * APD discriminator label space = this fold's training patients only
  * seg loss/metrics only for has_mask, cls only for has_grade; OS endpoint (D-1)

Outputs: <run_root>/<exp_id>/seed<s>/fold<k>/
    run_meta.json, history.csv, oof_predictions.csv, oof_seg/<pid>.npz,
    train_predictions.csv (final model on its own training patients; baseline hazard / KM threshold only),
    embeddings.npz (GAP(z) of all 51 patients under 3 input views, for E8),
    final.pth (only if cfg["save_final"])
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from data_pipeline import CLINICAL_DIMS, GRADE_MISSING, ClinicalBuilder, SarcomaDataset
from pretrain_fold import encoder_path
from revision_utils import fold_ids, load_splits, seed_everything, write_run_meta
from SarcomaNet import MultiTaskLoss, ReIDDiscriminator, RiskMemoryBank, SarcomaNet, concordance_index

ENCODER_KEYS = {"stem", "enc1", "enc2", "enc3", "bottleneck"}


def cosine_warmup_scheduler(opt, warmup, total, min_lr, base_lr):
    def _lr(epoch: int) -> float:
        if epoch < warmup:
            return (epoch + 1) / max(warmup, 1)
        progress = (epoch - warmup) / max(total - warmup, 1)
        return min_lr / base_lr + (1.0 - min_lr / base_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(opt, _lr)


def load_encoder(model: SarcomaNet, path: Path) -> int:
    pre = torch.load(str(path), map_location="cpu", weights_only=False)
    md  = model.state_dict()
    matched = {k: v for k, v in pre.items()
               if k.split(".")[0] in ENCODER_KEYS and k in md and md[k].shape == v.shape}
    n_enc = sum(1 for k in md if k.split(".")[0] in ENCODER_KEYS)
    if len(matched) != n_enc:
        raise RuntimeError(f"encoder transfer incomplete: {len(matched)}/{n_enc} from {path}")
    md.update(matched)
    model.load_state_dict(md)
    return len(matched)


def worker_init(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s); random.seed(s)


def dice_iou(p: np.ndarray, g: np.ndarray):
    """No smoothing. Only called for patients with a reference mask (BUGS.md B7)."""
    inter = np.logical_and(p, g).sum(); ps, gs = p.sum(), g.sum()
    dice = 2 * inter / (ps + gs) if ps + gs > 0 else float("nan")
    union = np.logical_or(p, g).sum()
    iou = inter / union if union > 0 else float("nan")
    return float(dice), float(iou)


def to_dev(batch, device):
    keys = ("seg_mask", "has_mask", "has_grade", "cls_label", "surv_time", "surv_event")
    return {k: batch[k].to(device, non_blocking=True) for k in keys}


@torch.no_grad()
def predict(model, loader, device, use_amp, clinical_on):
    model.eval()
    rows, segs = [], {}
    for b in loader:
        x = b["image"].to(device); c = b["clinical"].to(device) if clinical_on else None
        with autocast(device_type=device.type, enabled=use_amp):
            out = model(x, c)
        prob = torch.softmax(out["cls"].float(), 1).cpu().numpy()
        risk = out["surv"].float().cpu().numpy()
        pred = out["seg"].argmax(1).cpu().numpy().astype(np.uint8)
        gt   = b["seg_mask"].numpy().astype(bool)
        for i, pid in enumerate(b["patient_id"]):
            hm = bool(b["has_mask"][i] > 0.5)
            d, j = dice_iou(pred[i].astype(bool), gt[i]) if hm else (float("nan"), float("nan"))
            rows.append({"pid": pid, "has_mask": int(hm), "has_grade": int(b["has_grade"][i] > 0.5),
                         "grade": int(b["cls_label"][i]) if b["has_grade"][i] > 0.5 else GRADE_MISSING,
                         "p_low": prob[i, 0], "p_int": prob[i, 1], "p_high": prob[i, 2],
                         "risk": float(risk[i]), "time": float(b["surv_time"][i]),
                         "event": int(b["surv_event"][i]), "dice": d, "iou": j,
                         "pred_voxels": int(pred[i].sum()), "gt_voxels": int(gt[i].sum()) if hm else -1})
            segs[pid] = pred[i]
    return pd.DataFrame(rows), segs


def log_metrics(df: pd.DataFrame) -> dict:
    """Validation metrics for LOGGING ONLY (never used to select anything)."""
    g = df[df.has_grade == 1]
    acc = float((g[["p_low", "p_int", "p_high"]].values.argmax(1) == g.grade.values).mean()) if len(g) else float("nan")
    return {"val_dice": float(df.loc[df.has_mask == 1, "dice"].mean()),
            "val_acc": acc,
            "val_cindex": float(concordance_index(df.risk.values, df.time.values, df.event.values))}


@torch.no_grad()
def embeddings(model, ds, device, use_amp, clinical_on):
    """GAP(z) under three input views: full, MRI only, PET+CT only (privacy attack E8)."""
    model.eval()
    pids, out = [], {"full": [], "mri_only": [], "petct_only": []}
    for i in range(len(ds)):
        s = ds[i]; x = s["image"].unsqueeze(0).to(device)
        c = s["clinical"].unsqueeze(0).to(device) if clinical_on else None
        views = {"full": x, "mri_only": x * torch.tensor([1., 0, 0], device=device).view(1, 3, 1, 1, 1),
                 "petct_only": x * torch.tensor([0., 1, 1], device=device).view(1, 3, 1, 1, 1)}
        for k, v in views.items():
            with autocast(device_type=device.type, enabled=use_amp):
                z = model(v, c)["z"]
            out[k].append(z.float().mean(dim=(2, 3, 4)).squeeze(0).cpu().numpy())
        pids.append(s["patient_id"])
    return pids, {k: np.stack(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=None, help="override: dry runs ONLY")
    ap.add_argument("--max_batches", type=int, default=None, help="dry runs ONLY")
    a = ap.parse_args()

    cfg = json.loads(Path(a.config).read_text())
    dry = bool(a.epochs or a.max_batches)
    epochs = a.epochs or cfg["epochs"]
    sp = load_splits(cfg["splits"])
    out = Path(cfg["run_root"]) / (cfg["exp_id"] + ("_DRYRUN" if dry else "")) / f"seed{a.seed}" / f"fold{a.fold}"
    if (out / "oof_predictions.csv").exists():
        raise SystemExit(f"{out} already finished; refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)

    seed_everything(a.seed * 100 + a.fold)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.get("amp", True) and device.type == "cuda"
    train_ids, val_ids = fold_ids(sp, a.seed, a.fold)
    shape = tuple(cfg["target_shape"])
    clinical_on = cfg.get("use_clinical", True)
    cdim = CLINICAL_DIMS[cfg["clinical_version"]] if clinical_on else 0
    tasks = tuple(cfg.get("tasks", ["seg", "cls", "surv"]))

    def make_ds(ids, aug):
        return SarcomaDataset(cfg.get("data_root", "."), cfg["clinical_xls"], ids, augment=aug,
                              target_shape=shape, cache_dir=cfg["cache_dir"], has_mask=sp["has_mask"],
                              endpoint=cfg["endpoint"], clinical_version=cfg["clinical_version"])
    nw = cfg.get("num_workers", 4)
    g = torch.Generator(); g.manual_seed(a.seed * 100 + a.fold)
    train_dl = DataLoader(make_ds(train_ids, True), batch_size=cfg["batch_size"], shuffle=True, num_workers=nw,
                          pin_memory=device.type == "cuda", worker_init_fn=worker_init, generator=g)
    val_dl   = DataLoader(make_ds(val_ids, False), batch_size=1, shuffle=False, num_workers=nw)

    model = SarcomaNet(in_channels=3, n_seg_classes=2, n_cls_classes=3, clinical_dim=cdim,
                       base_ch=cfg["base_ch"], dropout=cfg["dropout"],
                       use_multiscale=cfg.get("use_multiscale", True)).to(device)

    n_enc = 0
    if cfg.get("pretrain_mode", "crmae") != "none":
        n_enc = load_encoder(model, encoder_path(cfg["encoder_root"], cfg["pretrain_mode"], a.seed, a.fold, dry))

    cb = ClinicalBuilder(cfg["clinical_xls"], endpoint=cfg["endpoint"], clinical_version=cfg["clinical_version"])
    tr_grades = [cb.grade(p) for p in train_ids if cb.grade(p) != GRADE_MISSING]
    counts = np.maximum(np.bincount(tr_grades, minlength=3).astype(float), 1.0)
    cls_w = torch.tensor(counts.sum() / (3 * counts), dtype=torch.float32, device=device)

    loss_fn = MultiTaskLoss(w_seg=cfg["w_seg"], w_cls=cfg["w_cls"], w_surv=cfg["w_surv"],
                            learnable_weights=cfg["learnable_weights"], cls_weights=cls_w,
                            label_smoothing=cfg["label_smoothing"], tasks=tasks).to(device)
    bank = RiskMemoryBank(train_ids, [cb.survival(p)[0] for p in train_ids]) if cfg.get("cox_memory_bank", True) else None

    reid, pid_idx = None, None
    lam = cfg.get("lambda_reid", 0.0) if cfg.get("use_apd", True) else 0.0
    if lam > 0:
        pid_idx = {p: i for i, p in enumerate(train_ids)}       # training patients only (B9/P3)
        reid = ReIDDiscriminator(in_ch=cfg["base_ch"] * 16, n_patients=len(train_ids),
                                 dropout=cfg["dropout"]).to(device)

    params = list(model.parameters()) + list(loss_fn.parameters()) + (list(reid.parameters()) if reid else [])
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = cosine_warmup_scheduler(opt, cfg["warmup_epochs"], epochs, cfg["min_lr"], cfg["lr"])
    scaler = GradScaler(device="cuda", enabled=use_amp)

    write_run_meta(out, cfg, sp, {"stage": "train", "seed": a.seed, "fold": a.fold, "epochs_run": epochs,
                                  "dry_run": dry, "n_train": len(train_ids), "val_ids": val_ids,
                                  "encoder_tensors_loaded": n_enc, "clinical_dim": cdim, "tasks": tasks,
                                  "class_weights": cls_w.tolist()})
    hist = []
    for ep in range(1, epochs + 1):
        model.train(); loss_fn.train(); loss_fn.reset_cox_counters()
        if reid: reid.train()
        alpha = 2.0 / (1.0 + math.exp(-10.0 * ep / epochs)) - 1.0
        t0 = time.time(); acc = {"loss": 0., "seg": 0., "cls": 0., "surv": 0., "reid": 0.}; nb = 0
        for bi, b in enumerate(train_dl):
            if a.max_batches and bi >= a.max_batches:
                break
            x = b["image"].to(device, non_blocking=True)
            c = b["clinical"].to(device) if clinical_on else None
            tgt = to_dev(b, device); pids = list(b["patient_id"])
            rb, tb = bank.others(pids, device) if bank else (None, None)
            opt.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=use_amp):
                outp = model(x, c)
                loss, br = loss_fn(outp, tgt, rb, tb)
                if reid is not None:
                    idx = torch.tensor([pid_idx[p] for p in pids], device=device)
                    l_reid = torch.nn.functional.cross_entropy(reid(outp["z"], alpha=alpha), idx)
                    loss = loss + lam * l_reid
                    acc["reid"] += l_reid.item()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update()
            if bank: bank.update(pids, outp["surv"])
            acc["loss"] += loss.item(); acc["seg"] += br["seg"].item(); acc["cls"] += br["cls"].item()
            acc["surv"] += br["surv"].item(); nb += 1
        sched.step()
        vdf, _ = predict(model, val_dl, device, use_amp, clinical_on)
        row = {"epoch": ep, "lr": opt.param_groups[0]["lr"], "alpha": alpha, "seconds": round(time.time() - t0, 2),
               **{f"train_{k}": v / max(nb, 1) for k, v in acc.items()},
               "cox_skip_pct": 100.0 * loss_fn.n_cox_skipped / max(loss_fn.n_cox_total, 1),
               **{f"log_var_{t}": float(v) for t, v in zip(("seg", "cls", "surv"), loss_fn.log_var.detach().cpu())},
               **log_metrics(vdf)}
        hist.append(row)
        pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
        print(f"  [{cfg['exp_id']} s{a.seed} f{a.fold}] ep {ep:03d}/{epochs} loss={row['train_loss']:.4f} "
              f"dice={row['val_dice']:.3f} acc={row['val_acc']:.3f} C={row['val_cindex']:.3f} "
              f"cox_skip={row['cox_skip_pct']:.0f}% ({row['seconds']}s)  [logging only]", flush=True)

    # ── final epoch = the one and only evaluated model ──────────────────────────
    vdf, segs = predict(model, val_dl, device, use_amp, clinical_on)
    vdf.insert(0, "fold", a.fold); vdf.insert(0, "seed", a.seed); vdf.insert(0, "exp_id", cfg["exp_id"])
    (out / "oof_seg").mkdir(exist_ok=True)
    for pid, m in segs.items():
        np.savez_compressed(out / "oof_seg" / f"{pid}.npz", pred=m)
    # training-fold predictions of the same final model (no augmentation): used ONLY
    # for training-fold quantities (Breslow baseline hazard, KM risk threshold; PROTOCOL §7)
    tdf, _ = predict(model, DataLoader(make_ds(train_ids, False), batch_size=1, shuffle=False,
                                       num_workers=nw), device, use_amp, clinical_on)
    tdf.drop(columns=["dice", "iou", "pred_voxels", "gt_voxels"]).to_csv(out / "train_predictions.csv", index=False)
    all_ds = make_ds(sp["patients"], False)
    pids, emb = embeddings(model, all_ds, device, use_amp, clinical_on)
    np.savez_compressed(out / "embeddings.npz", pids=np.array(pids),
                        is_train=np.array([p in set(train_ids) for p in pids]), **emb)
    if cfg.get("save_final", False):
        torch.save({"model": model.state_dict(), "loss_fn": loss_fn.state_dict(),
                    "reid": reid.state_dict() if reid else None, "cfg": cfg, "epoch": epochs}, out / "final.pth")
    vdf.to_csv(out / "oof_predictions.csv", index=False)   # written last = run complete
    print(f"  done -> {out}")


if __name__ == "__main__":
    main()
