"""
seg_baselines.py — E10 MONAI segmentation baselines (PROTOCOL §8 E10; R1-M8, R4.3; design in §11).

    python seg_baselines.py --config configs/E10_swinunetr.json --seed 42 --fold 0

Same everything as E0 except the network: cached 48x96x96 inputs (T2FS, PET, CT), frozen splits, the fold's
SEG training patients only, the same augmentation, loss (0.5 Dice + 0.5 focal, SarcomaNet.DiceLoss/FocalLoss),
AdamW lr 3e-4 / wd 1e-5, 5 warm-up epochs + cosine to 1e-6, AMP, grad clip 1.0, batch 8, E = 150, and the
final-epoch model only. Networks (cfg["seg_net"]): MONAI UNet, SegResNet, SwinUNETR (input zero-padded in depth to a
multiple of 32 for SwinUNETR, output cropped back).
Writes train.py's output format; grade/survival columns are filled with uninformative values (p = 1/3, risk = 0)
and must not be reported (run_meta "tasks": ["seg"]).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from data_pipeline import GRADE_MISSING, SarcomaDataset
from revision_utils import fold_ids, load_splits, seed_everything, write_run_meta
from SarcomaNet import DiceLoss, FocalLoss
from train import cosine_warmup_scheduler, dice_iou, worker_init


class PadToMultiple(nn.Module):
    def __init__(self, net, mult=32):
        super().__init__(); self.net, self.mult = net, mult

    def forward(self, x):
        D, H, W = x.shape[2:]
        pd_ = [(-s) % self.mult for s in (D, H, W)]
        pads = []
        for p in reversed(pd_):                       # F.pad order: W, H, D
            pads += [p // 2, p - p // 2]
        y = self.net(F.pad(x, pads))
        return y[:, :, pd_[0] // 2: pd_[0] // 2 + D, pd_[1] // 2: pd_[1] // 2 + H, pd_[2] // 2: pd_[2] // 2 + W]


def build_net(name):
    from monai.networks import nets
    if name == "unet":
        return nets.UNet(spatial_dims=3, in_channels=3, out_channels=2, channels=(16, 32, 64, 128, 256),
                         strides=(2, 2, 2, 2), num_res_units=2)
    if name == "segresnet":
        return nets.SegResNet(spatial_dims=3, in_channels=3, out_channels=2, init_filters=16,
                              blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1))
    if name == "swinunetr":
        return PadToMultiple(nets.SwinUNETR(in_channels=3, out_channels=2, feature_size=24, spatial_dims=3), 32)
    raise ValueError(name)


@torch.no_grad()
def predict(model, loader, device, use_amp):
    model.eval(); rows, segs = [], {}
    for b in loader:
        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(b["image"].to(device))
        pred = logits.float().argmax(1).cpu().numpy().astype(np.uint8)
        gt = b["seg_mask"].numpy().astype(bool)
        for i, pid in enumerate(b["patient_id"]):
            hm = bool(b["has_mask"][i] > 0.5)
            d, j = dice_iou(pred[i].astype(bool), gt[i]) if hm else (float("nan"), float("nan"))
            rows.append({"pid": pid, "has_mask": int(hm), "has_grade": int(b["has_grade"][i] > 0.5),
                         "grade": int(b["cls_label"][i]) if b["has_grade"][i] > 0.5 else GRADE_MISSING,
                         "p_low": 1 / 3, "p_int": 1 / 3, "p_high": 1 / 3, "risk": 0.0,
                         "time": float(b["surv_time"][i]), "event": int(b["surv_event"][i]), "dice": d, "iou": j,
                         "pred_voxels": int(pred[i].sum()), "gt_voxels": int(gt[i].sum()) if hm else -1})
            segs[pid] = pred[i]
    return pd.DataFrame(rows), segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=None, help="override: dry runs ONLY")
    ap.add_argument("--max_batches", type=int, default=None, help="dry runs ONLY")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text()); sp = load_splits(cfg["splits"])
    dry = bool(a.epochs or a.max_batches); epochs = a.epochs or cfg["epochs"]
    out = Path(cfg["run_root"]) / (cfg["exp_id"] + ("_DRYRUN" if dry else "")) / f"seed{a.seed}" / f"fold{a.fold}"
    if (out / "oof_predictions.csv").exists():
        raise SystemExit(f"{out} already finished; refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)

    seed_everything(a.seed * 100 + a.fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.get("amp", True) and device.type == "cuda"
    train_ids, val_ids = fold_ids(sp, a.seed, a.fold)
    seg_train = [p for p in train_ids if sp["has_mask"][p]]
    shape = tuple(cfg["target_shape"])

    def make_ds(ids, aug):
        return SarcomaDataset(cfg.get("data_root", "."), cfg["clinical_xls"], ids, augment=aug, target_shape=shape,
                              cache_dir=cfg["cache_dir"], has_mask=sp["has_mask"], endpoint=cfg["endpoint"],
                              clinical_version=cfg["clinical_version"])
    nw = cfg.get("num_workers", 4)
    g = torch.Generator(); g.manual_seed(a.seed * 100 + a.fold)
    train_dl = DataLoader(make_ds(seg_train, True), batch_size=cfg["batch_size"], shuffle=True, num_workers=nw,
                          pin_memory=device.type == "cuda", worker_init_fn=worker_init, generator=g)
    val_dl = DataLoader(make_ds(val_ids, False), batch_size=1, shuffle=False, num_workers=nw)

    model = build_net(cfg["seg_net"]).to(device)
    dice_l, focal_l = DiceLoss(ignore_bg=True), FocalLoss(gamma=2.0)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = cosine_warmup_scheduler(opt, cfg["warmup_epochs"], epochs, cfg["min_lr"], cfg["lr"])
    scaler = GradScaler(device="cuda", enabled=use_amp)
    n_params = sum(p.numel() for p in model.parameters())
    write_run_meta(out, cfg, sp, {"stage": "seg_baseline", "seg_net": cfg["seg_net"], "seed": a.seed, "fold": a.fold,
                                  "epochs_run": epochs, "dry_run": dry, "n_train_seg": len(seg_train),
                                  "val_ids": val_ids, "tasks": ["seg"], "n_params": n_params})
    hist = []
    for ep in range(1, epochs + 1):
        model.train(); t0 = time.time(); tot, nb = 0.0, 0
        for bi, b in enumerate(train_dl):
            if a.max_batches and bi >= a.max_batches:
                break
            x, y = b["image"].to(device), b["seg_mask"].to(device)
            with autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
            logits = logits.float()
            loss = 0.5 * dice_l(logits, y) + 0.5 * focal_l(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tot += loss.item(); nb += 1
        sched.step()
        vdf, _ = predict(model, val_dl, device, use_amp)
        hist.append({"epoch": ep, "lr": opt.param_groups[0]["lr"], "seconds": round(time.time() - t0, 2),
                     "train_seg": tot / max(nb, 1), "val_dice": float(np.nanmean(vdf.dice))})   # logging only
        pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
        print(f"  [{cfg['exp_id']} s{a.seed} f{a.fold}] ep {ep:03d}/{epochs} loss={hist[-1]['train_seg']:.4f} "
              f"dice={hist[-1]['val_dice']:.3f} ({hist[-1]['seconds']}s)  [logging only]", flush=True)

    vdf, segs = predict(model, val_dl, device, use_amp)                    # final epoch only
    vdf.insert(0, "fold", a.fold); vdf.insert(0, "seed", a.seed); vdf.insert(0, "exp_id", cfg["exp_id"])
    (out / "oof_seg").mkdir(exist_ok=True)
    for pid, m in segs.items():
        np.savez_compressed(out / "oof_seg" / f"{pid}.npz", pred=m)
    tdf, _ = predict(model, DataLoader(make_ds(train_ids, False), batch_size=1, shuffle=False, num_workers=nw),
                     device, use_amp)
    tdf.drop(columns=["dice", "iou", "pred_voxels", "gt_voxels"]).to_csv(out / "train_predictions.csv", index=False)
    vdf.to_csv(out / "oof_predictions.csv", index=False)                    # written last = run complete
    print(f"done {out}: final-epoch val Dice {np.nanmean(vdf.dice):.3f}")


if __name__ == "__main__":
    main()
