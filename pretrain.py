"""
pretrain.py — Cross-Modal Masked Autoencoding (CrMAE) Self-Supervised Pretraining

Novel Contribution:
    Given 3-channel multi-modal volumes (T2FS MRI, FDG-PET, CT), randomly masks
    one entire modality channel at the patch level and forces the encoder to
    reconstruct it using only the remaining two channels. This teaches the
    encoder to discover cross-modal anatomical correspondences without any
    labels.

    Unlike standard MAE (same-modality reconstruction), CrMAE exploits the
    genuinely multi-modal nature of the inputs — the complementary biophysical
    contrasts in T2FS MRI (water/oedema), FDG-PET (metabolic activity), and
    CT (electron density / bone) force the encoder to learn tumour morphology
    that is invariant to modality-specific intensity artefacts.

Usage:
    python pretrain.py                        # default 40 epochs
    python pretrain.py --epochs 60            # more pretraining
    python pretrain.py --config config.json   # use config file

Output:
    outputs/pretrained_encoder.pth  — encoder weights for transfer to train.py
"""

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader

from SarcomaNet import ResBlock3D, EncoderBlock, Bottleneck


# ─── Dataset ──────────────────────────────────────────────────────────────────

class PretrainDataset(Dataset):
    """
    Loads preprocessed image.npy files from the cache directory.
    Each volume is (3, D, H, W) with channels = [T2FS MRI, FDG-PET, CT].
    No segmentation masks or clinical labels required.
    """

    def __init__(self, cache_dir: str, patient_ids: list):
        self.samples = []
        for pid in patient_ids:
            p = Path(cache_dir) / pid / "image.npy"
            if p.exists():
                self.samples.append((pid, str(p)))
        print(f"  PretrainDataset: {len(self.samples)}/{len(patient_ids)} patients cached")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pid, path = self.samples[idx]
        image = np.load(path).astype(np.float32)   # (3, D, H, W) — [T2FS, PET, CT]

        # Lightweight augmentation (no labels, so all transforms are safe)
        if random.random() < 0.5:
            image = image[:, ::-1].copy()          # flip depth
        if random.random() < 0.5:
            image = image[:, :, ::-1].copy()       # flip height
        if random.random() < 0.5:
            image = image[:, :, :, ::-1].copy()    # flip width
        # Random brightness jitter per channel
        for c in range(image.shape[0]):
            image[c] = (image[c] * random.uniform(0.85, 1.15)).clip(0, 1)

        return torch.from_numpy(image)


# ─── Model ────────────────────────────────────────────────────────────────────

class CrMAEModel(nn.Module):
    """
    Cross-Modal Masked Autoencoder.

    Encoder: identical architecture to SarcomaNet (stem → enc1 → enc2 → enc3 → bottleneck).
    Decoder: lightweight 3× ConvTranspose3d to reconstruct a single masked channel.

    The encoder weights can be directly transferred to SarcomaNet for fine-tuning,
    giving the supervised model a head-start with patient-agnostic representations
    learned from unlabelled multi-modal volumes.
    """

    def __init__(self, in_channels: int = 3, base_ch: int = 8, out_channels: int = 1):
        super().__init__()
        ch = [base_ch * (2 ** i) for i in range(5)]   # e.g. [8,16,32,64,128] for base_ch=8

        # ── Encoder (mirrors SarcomaNet exactly) ───────────────────────────────
        self.stem       = ResBlock3D(in_channels, ch[0])
        self.enc1       = EncoderBlock(ch[0], ch[1])
        self.enc2       = EncoderBlock(ch[1], ch[2])
        self.enc3       = EncoderBlock(ch[2], ch[3])
        self.bottleneck = Bottleneck(ch[3], ch[4])

        # ── Reconstruction decoder (no skip connections — reconstruction only) ─
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(ch[4], ch[3], kernel_size=2, stride=2),
            nn.BatchNorm3d(ch[3]), nn.GELU(),
            nn.ConvTranspose3d(ch[3], ch[2], kernel_size=2, stride=2),
            nn.BatchNorm3d(ch[2]), nn.GELU(),
            nn.ConvTranspose3d(ch[2], ch[1], kernel_size=2, stride=2),
            nn.BatchNorm3d(ch[1]), nn.GELU(),
            nn.Conv3d(ch[1], out_channels, kernel_size=1),   # 1 = CrMAE (masked modality); 3 = spatial MAE
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.stem(x)
        x1, _ = self.enc1(s0)
        x2, _ = self.enc2(x1)
        x3, _ = self.enc3(x2)
        return self.bottleneck(x3)

    def forward(self, x_masked: torch.Tensor) -> torch.Tensor:
        z    = self.encode(x_masked)
        recon = self.decoder(z)
        return recon

    def encoder_state_dict(self) -> dict:
        """Return encoder-only weights for transfer to SarcomaNet."""
        encoder_keys = {"stem", "enc1", "enc2", "enc3", "bottleneck"}
        return {
            k: v for k, v in self.state_dict().items()
            if k.split(".")[0] in encoder_keys
        }


# ─── Patch Masking ────────────────────────────────────────────────────────────

def make_patch_mask(
    shape:      tuple,
    mask_ratio: float = 0.5,
    patch_size: int   = 4,
    device:     torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Create a random 3D patch-level binary mask.
    Returns shape (D, H, W) — 1 = masked (reconstruct here), 0 = visible.
    """
    D, H, W = shape
    pd = max(1, D // patch_size)
    ph = max(1, H // patch_size)
    pw = max(1, W // patch_size)
    n_patches = pd * ph * pw
    n_mask    = max(1, int(n_patches * mask_ratio))

    patch_mask = torch.zeros(n_patches)
    patch_mask[torch.randperm(n_patches)[:n_mask]] = 1.0
    patch_mask = patch_mask.reshape(1, 1, pd, ph, pw)

    # Upsample to voxel resolution
    mask = F.interpolate(patch_mask, size=(D, H, W), mode="nearest").squeeze()
    return mask.to(device)


# ─── Training ─────────────────────────────────────────────────────────────────

def run_pretrain(cfg: dict, epochs: int, lr: float, mask_ratio: float, patch_size: int):
    torch.set_num_threads(os.cpu_count() or 4)

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect all patient IDs
    patient_ids = sorted(p.name for p in Path(cfg["data_root"]).iterdir() if p.is_dir())
    ds = PretrainDataset("cache", patient_ids)

    if len(ds) == 0:
        print("[ERROR] No cached volumes found. Run preprocess.py first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nw = int(cfg.get("num_workers", 0))
    if device.type == "cuda" and nw <= 0:
        max_workers = 2 if os.name == "nt" else 4
        nw = min(max_workers, os.cpu_count() or 1)

    dl = DataLoader(
        ds,
        batch_size = cfg.get("batch_size", 4),
        shuffle    = True,
        num_workers= nw,
        pin_memory = (device.type == "cuda"),
        drop_last  = False,
    )

    model = CrMAEModel(
        in_channels = cfg["in_channels"],
        base_ch     = cfg["base_ch"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  CrMAE — {n_params:,} parameters")
    print(f"  Pretraining: {epochs} epochs | lr={lr} | mask_ratio={mask_ratio} | patch_size={patch_size}")
    print(f"  {'Epoch':>6}  {'Recon Loss':>12}  {'LR':>10}")
    print(f"  {'-'*34}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    use_amp = device.type == "cuda"
    scaler = GradScaler(device="cuda", enabled=use_amp)

    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for images in dl:
            images = images.to(device)                     # (B, 3, D, H, W) — [T2FS, PET, CT]
            B, C, D, H, W = images.shape

            # ── Choose which modality to mask ────────────────────────────────
            mask_ch = random.randint(0, C - 1)

            # ── Create spatial patch mask ─────────────────────────────────────
            mask = make_patch_mask(
                (D, H, W), mask_ratio=mask_ratio,
                patch_size=patch_size, device=device,
            )  # (D, H, W)
            mask = mask.unsqueeze(0).unsqueeze(0).expand(B, 1, D, H, W)  # (B,1,D,H,W)

            # ── Apply mask: zero the chosen channel at masked patches ─────────
            x_masked = images.clone()
            x_masked[:, mask_ch:mask_ch+1] = (
                x_masked[:, mask_ch:mask_ch+1] * (1.0 - mask)
            )

            # ── Forward → reconstruct masked channel ─────────────────────────
            with autocast(device_type=device.type, enabled=use_amp):
                recon = model(x_masked)                        # (B, 1, D', H', W')
                if recon.shape[2:] != (D, H, W):
                    recon = F.interpolate(
                        recon, size=(D, H, W), mode="trilinear", align_corners=False
                    )

                target = images[:, mask_ch:mask_ch+1]         # (B, 1, D, H, W)

                # Loss: MSE on masked voxels only (unmasked voxels trivially correct)
                loss = ((recon - target) ** 2 * mask).sum() / (mask.sum() + 1e-8)

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            total_loss += loss.item()
            n_batches  += 1

        sched.step()
        avg_loss = total_loss / max(n_batches, 1)
        lr_now   = opt.param_groups[0]["lr"]

        print(f"  {epoch:>6}  {avg_loss:>12.6f}  {lr_now:>10.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                model.encoder_state_dict(),
                str(out_dir / "pretrained_encoder.pth"),
            )

    print(f"\n  Best recon loss : {best_loss:.6f}")
    print(f"  Saved           : {(out_dir / 'pretrained_encoder.pth').resolve()}")


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    raise SystemExit("pretrain.py main() pretrains on ALL patients (leakage, BUGS.md B3). "
                     "Use pretrain_fold.py --seed S --fold K instead.")
    p = argparse.ArgumentParser(
        description="CrMAE — Cross-Modal Masked Autoencoding pretraining"
    )
    p.add_argument("--config",     default="config.json",
                   help="Path to config.json")
    p.add_argument("--epochs",     type=int,   default=None,
                   help="Pretraining epochs (overrides config pretrain_epochs)")
    p.add_argument("--lr",         type=float, default=None,
                   help="Learning rate (overrides config pretrain_lr)")
    p.add_argument("--mask_ratio", type=float, default=None,
                   help="Fraction of patches to mask (overrides config mask_ratio)")
    p.add_argument("--patch_size", type=int,   default=None,
                   help="3D patch size for masking (overrides config patch_size)")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    epochs     = args.epochs     or cfg.get("pretrain_epochs", 40)
    lr         = args.lr         or cfg.get("pretrain_lr",     1e-3)
    mask_ratio = args.mask_ratio or cfg.get("mask_ratio",      0.5)
    patch_size = args.patch_size or cfg.get("patch_size",      4)

    run_pretrain(cfg, epochs, lr, mask_ratio, patch_size)


if __name__ == "__main__":
    main()
