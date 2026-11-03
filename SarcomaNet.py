"""
SarcomaNet — Tri-Head 3D ResU-Net  (array-revision)

Model = the submitted paper's model (reference/paper_run/SarcomaNet.py): deep heads
(cls 256-64, surv 256-64), learnable uncertainty weighting without prior. Changes are
limited to the losses (PROTOCOL §2; BUGS.md B6, B8, B12): masked seg/cls losses,
Breslow Cox over batch + memory bank, float32 Cox. The old smoke test was removed
(tests/ covers the losses).

  Head 1 : Segmentation   — voxel-wise tumour mask (Dice + Focal)
  Head 2 : Classification — FNCLCC grade (weighted CE + focal)
  Head 3 : Survival       — Cox proportional hazard log-risk score

Architecture
  Shared encoder  : 3-level 3D ResU-Net (base_ch → 8× base_ch)
  Decoder         : U-Net skip-connection decoder (seg path only)
  Cls + Surv heads: from bottleneck features + optional clinical tabular input
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Building Blocks ──────────────────────────────────────────────────────────

class ResBlock3D(nn.Module):
    """3D residual block (optional stride for downsampling)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.skip  = (
            nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )
            if in_ch != out_ch or stride != 1
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.skip(x))


class EncoderBlock(nn.Module):
    """Two residual blocks at full resolution, then MaxPool3d(2)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            ResBlock3D(in_ch, out_ch),
            ResBlock3D(out_ch, out_ch),
        )
        self.pool = nn.MaxPool3d(2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        skip = self.block(x)
        return self.pool(skip), skip   # (downsampled, skip_for_decoder)


class DecoderBlock(nn.Module):
    """ConvTranspose3d upsample → concat skip → two residual blocks."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up    = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.block = nn.Sequential(
            ResBlock3D(out_ch + skip_ch, out_ch),
            ResBlock3D(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Guard against odd-dimension rounding
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class Bottleneck(nn.Module):
    """Three deep residual blocks at the lowest resolution."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            ResBlock3D(in_ch, out_ch),
            ResBlock3D(out_ch, out_ch),
            ResBlock3D(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ─── Adversarial Patient De-identification (APD) ──────────────────────────────

class GradientReversalFn(torch.autograd.Function):
    """
    Gradient Reversal Layer (Ganin et al., ICML 2016).

    Forward pass: identity.
    Backward pass: multiply gradient by -alpha, forcing the encoder to
    produce features that confuse the discriminator (adversarial objective).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.alpha * grad_output, None


class ReIDDiscriminator(nn.Module):
    """
    Adversarial Patient De-identification Discriminator.

    Given bottleneck features z (B, C, D, H, W), predicts which patient the
    scan belongs to (n_patients-way classification).

    During training, gradient reversal forces the encoder to produce
    patient-agnostic representations — empirical privacy without any
    external cryptographic primitives.

    Increasing alpha from 0→1 over training provides a smooth curriculum:
    the encoder is first free to learn task-relevant features, then
    progressively penalised for encoding patient identity.

    Parameters
    ----------
    in_ch      : bottleneck channel count (base_ch * 16)
    n_patients : number of distinct patients (51 for STS)
    dropout    : dropout in MLP
    """

    def __init__(self, in_ch: int, n_patients: int, dropout: float = 0.3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fc  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_patients),
        )
        # Weight initialisation
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """
        Parameters
        ----------
        z     : bottleneck tensor (B, C, D, H, W)
        alpha : gradient reversal strength [0, 1]
        """
        z_rev = GradientReversalFn.apply(z, alpha)
        return self.fc(self.gap(z_rev))


# ─── Task Heads ───────────────────────────────────────────────────────────────

class SegHead(nn.Module):
    """1×1×1 conv → per-voxel class logits."""

    def __init__(self, in_ch: int, n_classes: int):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ClsHead(nn.Module):
    """
    Multi-Scale Classification Head (Fix 3).

    Accepts a pre-computed flat feature vector that concatenates GAP outputs
    from ALL encoder levels (s1, s2, s3, z) plus clinical features.
    Each encoder level captures grade-relevant texture at a different scale:
      enc1 → fine cell-level texture  (mitotic rate)
      enc2 → local pattern structure  (necrosis zones)
      enc3 → regional tumour shape    (capsule invasion)
      z    → global semantic encoding (overall grade)
    """

    def __init__(self, feat_dim: int, n_classes: int, dropout: float = 0.4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, 64),       nn.ReLU(inplace=True),
            nn.Linear(64, n_classes),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.fc(feat)


class SurvHead(nn.Module):
    """Global average pool → [optional clinical concat] → scalar log-risk (Cox PH)."""

    def __init__(self, in_ch: int, clinical_dim: int = 0, dropout: float = 0.4):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool3d(1)
        feat_dim = in_ch + clinical_dim
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, 64),       nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, clinical: Optional[torch.Tensor] = None) -> torch.Tensor:
        feat = self.gap(x).flatten(1)
        if clinical is not None:
            feat = torch.cat([feat, clinical], dim=1)
        return self.fc(feat).squeeze(1)   # (B,)


# ─── Main Model ───────────────────────────────────────────────────────────────

class SarcomaNet(nn.Module):
    """
    Tri-Head 3D ResU-Net for Soft Tissue Sarcoma.

    Spatial flow (D×H×W input, 3 encoder levels):
        stem  → s0 (D)
        enc1  → s1 (D),   x1 (D/2)
        enc2  → s2 (D/2), x2 (D/4)
        enc3  → s3 (D/4), x3 (D/8)
        bottleneck → z (D/8)
        dec3 (z + s3) → d3 (D/4)
        dec2 (d3+ s2) → d2 (D/2)
        dec1 (d2+ s1) → d1 (D)
        seg_head(d1)  → (B, n_seg, D, H, W)
        cls_head(z)   → (B, n_cls)
        surv_head(z)  → (B,)

    Parameters
    ----------
    in_channels   : imaging channels, e.g. 3 for [T2FS, PET, CT]
    n_seg_classes : 2 = background + tumour
    n_cls_classes : number of subtype / grade classes
    clinical_dim  : size of tabular clinical feature vector (0 to disable)
    base_ch       : base feature width (doubled per level)
    dropout       : dropout rate in cls/surv heads
    """

    def __init__(
        self,
        in_channels:    int   = 3,
        n_seg_classes:  int   = 2,
        n_cls_classes:  int   = 4,
        clinical_dim:   int   = 8,
        base_ch:        int   = 32,
        dropout:        float = 0.4,
        use_multiscale: bool  = True,   # ablation flag: multi-scale vs single-scale cls
    ):
        super().__init__()
        ch = [base_ch * (2 ** i) for i in range(5)]   # 32, 64, 128, 256, 512
        self.use_multiscale = use_multiscale

        # Encoder
        self.stem       = ResBlock3D(in_channels, ch[0])
        self.enc1       = EncoderBlock(ch[0], ch[1])
        self.enc2       = EncoderBlock(ch[1], ch[2])
        self.enc3       = EncoderBlock(ch[2], ch[3])

        # Bottleneck
        self.bottleneck = Bottleneck(ch[3], ch[4])

        # Decoder (segmentation path)
        self.dec3       = DecoderBlock(ch[4], ch[3], ch[3])
        self.dec2       = DecoderBlock(ch[3], ch[2], ch[2])
        self.dec1       = DecoderBlock(ch[2], ch[1], ch[1])

        # Heads
        self.seg_head   = SegHead(ch[1], n_seg_classes)
        self.cls_gap    = nn.AdaptiveAvgPool3d(1)
        # Multi-scale: GAP(s1+s2+s3+z)+clinical  |  Single-scale: GAP(z)+clinical
        ms_dim = (ch[1] + ch[2] + ch[3] + ch[4] + clinical_dim
                  if use_multiscale else ch[4] + clinical_dim)
        self.cls_head   = ClsHead(ms_dim, n_cls_classes, dropout)
        self.surv_head  = SurvHead(ch[4], clinical_dim, dropout)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x:        torch.Tensor,
        clinical: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        _, _, D, H, W = x.shape

        # Encode
        s0       = self.stem(x)
        x1, s1   = self.enc1(s0)
        x2, s2   = self.enc2(x1)
        x3, s3   = self.enc3(x2)
        z        = self.bottleneck(x3)

        # Decode (seg path)
        d3       = self.dec3(z,  s3)
        d2       = self.dec2(d3, s2)
        d1       = self.dec1(d2, s1)

        # Seg output — interpolate to input size for odd-dimension safety
        seg_logits = F.interpolate(
            self.seg_head(d1), size=(D, H, W), mode="trilinear", align_corners=False
        )

        # Classification feature: multi-scale or single-scale (ablation flag)
        if self.use_multiscale:
            ms_feat = torch.cat([
                self.cls_gap(s1).flatten(1),
                self.cls_gap(s2).flatten(1),
                self.cls_gap(s3).flatten(1),
                self.cls_gap(z).flatten(1),
            ], dim=1)
        else:
            ms_feat = self.cls_gap(z).flatten(1)
        if clinical is not None:
            ms_feat = torch.cat([ms_feat, clinical], dim=1)

        return {
            "seg":  seg_logits,
            "cls":  self.cls_head(ms_feat),
            "surv": self.surv_head(z, clinical),
            "z":    z,
        }


# ─── Loss Functions ───────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """Soft multi-class Dice loss (macro-average, background optional)."""

    def __init__(self, smooth: float = 1.0, ignore_bg: bool = True):
        super().__init__()
        self.smooth    = smooth
        self.ignore_bg = ignore_bg

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs   = torch.softmax(logits, dim=1)
        n_cls   = probs.shape[1]
        one_hot = F.one_hot(targets.long(), n_cls).permute(0, 4, 1, 2, 3).float()
        start   = 1 if self.ignore_bg else 0
        scores  = []
        for c in range(start, n_cls):
            p, g  = probs[:, c], one_hot[:, c]
            inter = (p * g).sum()
            union = p.sum() + g.sum()
            scores.append(1.0 - (2.0 * inter + self.smooth) / (union + self.smooth))
        return torch.stack(scores).mean()


class FocalLoss(nn.Module):
    """Focal loss for class-imbalanced voxel classification."""

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets.long(), weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def cox_breslow_nll(
    risk_batch:  torch.Tensor,
    time_batch:  torch.Tensor,
    event_batch: torch.Tensor,
    risk_bank:   Optional[torch.Tensor] = None,
    time_bank:   Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, bool]:
    """
    Negative Cox partial log-likelihood, Breslow handling of tied event times
    (array-revision, PROTOCOL D-10, BUGS.md B8).

    The risk set of each event in the batch is formed from the batch itself
    (with gradient) PLUS a memory bank of detached risk scores of the other
    training patients (their latest forward pass). With bank = None it reduces
    to the batch-local Cox loss of the submitted paper (without the < 2 events
    shortcut).

    Breslow: every subject j with t_j >= t_i is in R(t_i), including other
    events tied at t_i; each event contributes r_i - log sum_{j in R(t_i)} exp(r_j).

    Returns (loss, skipped). skipped = True when the batch has no events
    (loss is an exact zero that still participates in autograd).
    Computed in float32 regardless of autocast.
    """
    r  = risk_batch.float().reshape(-1)
    t  = time_batch.float().reshape(-1)
    e  = event_batch.float().reshape(-1)
    if e.sum() == 0:
        return r.sum() * 0.0, True

    if risk_bank is not None and risk_bank.numel() > 0:
        r_all = torch.cat([r, risk_bank.float().reshape(-1).detach()])
        t_all = torch.cat([t, time_bank.float().reshape(-1)])
    else:
        r_all, t_all = r, t

    at_risk = (t_all.unsqueeze(0) >= t.unsqueeze(1))                 # (B, N): j in R(t_i)
    neg_inf = torch.finfo(r_all.dtype).min
    log_den = torch.logsumexp(torch.where(at_risk, r_all.unsqueeze(0), torch.full_like(at_risk, neg_inf, dtype=r_all.dtype)), dim=1)
    nll = -((r - log_den) * e).sum() / e.sum()
    return nll, False


class RiskMemoryBank:
    """Latest detached risk score and survival time per training patient."""

    def __init__(self, pids, times, init_risk: float = 0.0):
        self.idx   = {p: i for i, p in enumerate(pids)}
        self.times = torch.tensor(times, dtype=torch.float32)
        self.risk  = torch.full((len(pids),), float(init_risk), dtype=torch.float32)

    def others(self, batch_pids, device):
        keep = torch.ones(len(self.idx), dtype=torch.bool)
        for p in batch_pids:
            keep[self.idx[p]] = False
        return self.risk[keep].to(device), self.times[keep].to(device)

    def update(self, batch_pids, risk_batch):
        r = risk_batch.detach().float().cpu().reshape(-1)
        for p, v in zip(batch_pids, r):
            self.risk[self.idx[p]] = v


class MultiTaskLoss(nn.Module):
    """
    Seg + cls + survival loss with homoscedastic uncertainty weighting
    (Kendall et al., 2018): sum_k exp(-s_k) L_k + s_k, s_k = log sigma_k^2, init 0.

    array-revision (PROTOCOL §2, D-2, D-3, D-10; BUGS.md B6, B8, B12):
      * seg loss only over samples with has_mask = 1
      * cls loss only over samples with has_grade = 1
      * survival: Breslow Cox over batch + memory bank (if supplied)
      * fixed weights are used only when learnable_weights = False (paper spec:
        learnable, so w_* are NOT used; the paper text is corrected accordingly)
      * a task with no eligible sample in the batch contributes 0 and its s_k
        term is still added (keeps the objective's form constant)
    """

    def __init__(
        self,
        w_seg:  float = 1.0,
        w_cls:  float = 1.0,
        w_surv: float = 0.5,
        learnable_weights:  bool                    = True,
        cls_weights:        Optional[torch.Tensor]  = None,
        label_smoothing:    float                   = 0.0,
        log_var_prior:      float                   = 0.0,
        tasks:              Tuple[str, ...]         = ("seg", "cls", "surv"),
    ):
        super().__init__()
        bad = set(tasks) - {"seg", "cls", "surv"}
        if bad or not tasks:
            raise ValueError(f"bad tasks {tasks}")
        self.register_buffer("active", torch.tensor([t in tasks for t in ("seg", "cls", "surv")], dtype=torch.float32))
        self.learnable     = learnable_weights
        self.log_var_prior = log_var_prior
        self.log_var       = nn.Parameter(torch.zeros(3), requires_grad=learnable_weights)
        self.register_buffer("weights", torch.tensor([w_seg, w_cls, w_surv]))

        self.dice_loss      = DiceLoss(ignore_bg=True)
        self.focal_seg_loss = FocalLoss(gamma=2.0)
        self.cls_ce_loss    = nn.CrossEntropyLoss(weight=cls_weights, label_smoothing=label_smoothing)
        self.cls_focal_loss = FocalLoss(gamma=2.0, weight=cls_weights)
        self.n_cox_skipped = 0
        self.n_cox_total   = 0

    def reset_cox_counters(self):
        self.n_cox_skipped = 0
        self.n_cox_total   = 0

    def forward(
        self,
        preds:   Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        risk_bank: Optional[torch.Tensor] = None,
        time_bank: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        seg_logits = preds["seg"]
        zero = seg_logits.sum() * 0.0

        m = targets["has_mask"] > 0.5
        if m.any():
            l_seg = (0.5 * self.dice_loss(seg_logits[m], targets["seg_mask"][m])
                     + 0.5 * self.focal_seg_loss(seg_logits[m], targets["seg_mask"][m]))
        else:
            l_seg = zero

        g = targets["has_grade"] > 0.5
        if g.any():
            lab = targets["cls_label"][g].long()
            l_cls = (0.5 * self.cls_ce_loss(preds["cls"][g], lab)
                     + 0.5 * self.cls_focal_loss(preds["cls"][g], lab))
        else:
            l_cls = zero

        l_surv, skipped = cox_breslow_nll(preds["surv"], targets["surv_time"], targets["surv_event"],
                                          risk_bank, time_bank)
        self.n_cox_total   += 1
        self.n_cox_skipped += int(skipped)

        losses = torch.stack([l_seg.float(), l_cls.float(), l_surv.float()])
        if self.learnable:
            s = self.log_var
            total = (self.active * (torch.exp(-s) * losses + s)).sum()
            if self.log_var_prior > 0:
                total = total + self.log_var_prior * (self.active * s ** 2).sum()
        else:
            total = (self.active * self.weights * losses).sum()

        breakdown = {"seg": l_seg.detach(), "cls": l_cls.detach(), "surv": l_surv.detach(),
                     "n_seg": int(m.sum()), "n_cls": int(g.sum())}
        return total, breakdown


# ─── Evaluation Metrics ───────────────────────────────────────────────────────

def compute_dice(
    pred: np.ndarray,
    gt:   np.ndarray,
    smooth: float = 1.0,
) -> float:
    """Binary Dice coefficient."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = (pred & gt).sum()
    return float((2 * inter + smooth) / (pred.sum() + gt.sum() + smooth))


def concordance_index(
    risk:   np.ndarray,
    times:  np.ndarray,
    events: np.ndarray,
) -> float:
    """Harrell's C-index for survival model evaluation."""
    concordant = discordant = 0
    n = len(times)
    for i in range(n):
        if events[i] == 0:
            continue
        for j in range(n):
            if i == j:
                continue
            if times[i] < times[j]:
                if risk[i] > risk[j]:
                    concordant += 1
                elif risk[i] < risk[j]:
                    discordant += 1
    total = concordant + discordant
    return concordant / total if total > 0 else 0.5
