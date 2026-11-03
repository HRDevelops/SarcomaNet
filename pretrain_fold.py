"""
pretrain_fold.py — within-fold self-supervised pretraining (PROTOCOL §5; BUGS.md B3).

    python pretrain_fold.py --config configs/E0.json --seed 42 --fold 0 --mode crmae

Trains on the TRAINING patients of (seed, fold) only, from splits.json.
  --mode crmae   : per sample, pick one modality m ~ U{0,1,2}; zero 50% of its
                   4x4x4 patches; reconstruct modality m on masked voxels (paper Alg. 1)
  --mode spatial : standard spatial-patch MAE (E1b): zero the same 50% of patches in
                   ALL channels; reconstruct all 3 channels on masked voxels
The encoder of the LAST epoch is saved (no selection on any loss; §3).

Output: <out_root>/<mode>/seed<s>/fold<k>/{encoder.pth, history.csv, run_meta.json}
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from pretrain import CrMAEModel, PretrainDataset, make_patch_mask
from revision_utils import fold_ids, load_splits, seed_everything, write_run_meta


def encoder_path(out_root, mode, seed, fold, dry: bool = False) -> Path:
    """Dry-run encoders live under <mode>_DRYRUN so they can never be used by real runs."""
    return Path(out_root) / (mode + ("_DRYRUN" if dry else "")) / f"seed{seed}" / f"fold{fold}" / "encoder.pth"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--mode", choices=["crmae", "spatial"], default="crmae")
    ap.add_argument("--epochs", type=int, default=None, help="override (dry runs only)")
    ap.add_argument("--max_batches", type=int, default=None, help="dry runs only")
    a = ap.parse_args()

    cfg = json.loads(Path(a.config).read_text())
    sp  = load_splits(cfg["splits"])
    epochs = a.epochs or cfg["pretrain_epochs"]
    dry = bool(a.epochs or a.max_batches)
    out = encoder_path(cfg["encoder_root"], a.mode, a.seed, a.fold, dry).parent
    if (out / "encoder.pth").exists():
        raise SystemExit(f"{out}/encoder.pth exists; refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)

    seed_everything(a.seed * 100 + a.fold)
    train_ids, val_ids = fold_ids(sp, a.seed, a.fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    ds = PretrainDataset(cfg["cache_dir"], train_ids)
    if len(ds) != len(train_ids):
        raise SystemExit(f"cache incomplete: {len(ds)}/{len(train_ids)} training patients")
    dl = DataLoader(ds, batch_size=cfg["pretrain_batch_size"], shuffle=True,
                    num_workers=cfg.get("num_workers", 4), pin_memory=use_amp, drop_last=False)

    model = CrMAEModel(in_channels=3, base_ch=cfg["base_ch"],
                       out_channels=1 if a.mode == "crmae" else 3).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=cfg["pretrain_lr"], weight_decay=cfg["pretrain_weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler(device="cuda", enabled=use_amp)

    write_run_meta(out, cfg, sp, {"stage": "pretrain", "mode": a.mode, "seed": a.seed, "fold": a.fold,
                                  "epochs": epochs, "n_train": len(train_ids),
                                  "val_ids_excluded": val_ids, "dry_run": dry})
    hist = []
    for ep in range(1, epochs + 1):
        model.train(); t0 = time.time(); tot = 0.0; nb = 0
        for bi, x in enumerate(dl):
            if a.max_batches and bi >= a.max_batches:
                break
            x = x.to(device, non_blocking=True)
            B, C, D, H, W = x.shape
            masks = torch.stack([make_patch_mask((D, H, W), cfg["mask_ratio"], cfg["patch_size"], device)
                                 for _ in range(B)]).unsqueeze(1)                  # (B,1,D,H,W)
            xm = x.clone()
            if a.mode == "crmae":
                ch = torch.randint(0, C, (B,), device=device)                    # per-sample modality
                sel = torch.nn.functional.one_hot(ch, C).float().view(B, C, 1, 1, 1)
                xm = xm * (1 - sel * masks)
                target = (x * sel).sum(1, keepdim=True)
            else:
                xm = xm * (1 - masks)
                target = x
            with autocast(device_type=device.type, enabled=use_amp):
                rec = model(xm)
                if rec.shape[2:] != (D, H, W):
                    rec = torch.nn.functional.interpolate(rec, size=(D, H, W), mode="trilinear", align_corners=False)
            m = masks.expand_as(target)
            loss = (((rec.float() - target) ** 2) * m).sum() / m.sum().clamp(min=1.0)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tot += loss.item(); nb += 1
        sched.step()
        hist.append({"epoch": ep, "recon_loss": tot / max(nb, 1), "lr": opt.param_groups[0]["lr"],
                     "seconds": round(time.time() - t0, 2)})
        print(f"  [{a.mode} s{a.seed} f{a.fold}] ep {ep:03d}/{epochs} loss={hist[-1]['recon_loss']:.5f} "
              f"({hist[-1]['seconds']}s)", flush=True)
        pd.DataFrame(hist).to_csv(out / "history.csv", index=False)

    torch.save(model.encoder_state_dict(), out / "encoder.pth")   # last epoch
    print(f"  saved {out / 'encoder.pth'}")


if __name__ == "__main__":
    main()
