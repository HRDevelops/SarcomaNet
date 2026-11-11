"""
make_figures.py — manuscript figures from the E0 runs (revision, Sarcoma_v2).

    python tools/make_figures.py --runs <dir containing E0_full_r2/seed*/fold*> --cache <cache dir> \
        --out <manuscript dir> --seg_pids STS_008 STS_007 STS_026

Writes (never overwrites):
  Fig_09_training_v2.pdf  training dynamics, mean +- SD over 15 runs (5 folds x 3 seeds)
  Fig_10_segmentation_v2.pdf  seed-42 final-epoch predictions for the given patients (T2FS, the mask reference)
  Fig_11_km_v2.pdf  Kaplan-Meier OS curves, high vs low predicted risk (evaluate.km_groups)
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate as ev  # noqa: E402

C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"     # validated categorical slots 1-3 (dataviz reference palette)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "lines.linewidth": 1.5,
                     "pdf.fonttype": 42})


def use_v3_style():
    """Match the redrawn diagram figures (figures_src/figstyle_v3.tex): Helvetica-like font, same palette."""
    global C1, C2, C3, INK, MUTED, GRID
    import os
    from matplotlib import font_manager as fm
    # TrueType copies (tools/otf2ttf.py) embed cleanly in PDF; the system .otf files are the fallback
    ttf_dir = Path(os.environ.get("SARCOMA_FONT_DIR", "/nonexistent"))
    for f in ("Regular", "Bold", "Italic"):
        p = ttf_dir / f"NimbusSans-{f}.ttf"
        if not p.exists():
            p = Path(f"/usr/share/fonts/urw-base35/NimbusSans-{f}.otf")
        if p.exists():
            fm.fontManager.addfont(str(p))
    if ttf_dir.exists():   # make sure the system .otf copies of the same family are not picked instead
        fm.fontManager.ttflist = [f for f in fm.fontManager.ttflist
                                  if not (f.name == "Nimbus Sans" and f.fname.endswith(".otf"))]
    C1, C2, C3 = "#4C6FB0", "#C9762A", "#4E8C3E"          # encoder blue, head orange, decoder green
    INK, MUTED, GRID = "#000000", "#6A6A6A", "#E6E6E6"
    plt.rcParams.update({"font.family": ["Nimbus Sans", "DejaVu Sans"], "font.size": 8,
                         "mathtext.fontset": "custom", "mathtext.rm": "Nimbus Sans",
                         "mathtext.it": "Nimbus Sans:italic", "mathtext.bf": "Nimbus Sans:bold",
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "grid.color": GRID, "axes.linewidth": 0.8})


def safe_out(path):
    if path.exists():
        raise SystemExit(f"{path} exists; refusing to overwrite")
    return path


def band(ax, h, col, color, label=None):
    g = h.groupby("epoch")[col]; m, s = g.mean(), g.std()
    ax.fill_between(m.index, m - s, m + s, color=color, alpha=0.18, linewidth=0)
    ax.plot(m.index, m.values, color=color, label=label)
    return m


def training_figure(runs, out, name="Fig_09_training_v2.pdf"):
    h = pd.concat([pd.read_csv(f).assign(run=str(f.parent)) for f in sorted(runs.glob("seed*/fold*/history.csv"))])
    n = h.run.nunique()
    fig, axs = plt.subplots(2, 4, figsize=(7.2, 3.6), constrained_layout=True)
    panels = [("train_seg", "Train loss: seg."), ("train_cls", "Train loss: grade"),
              ("train_surv", "Train loss: Cox"), (None, "Log-variances $s_k$"),
              ("val_dice", "Val. Dice"), ("val_acc", "Val. accuracy"),
              ("val_cindex", "Val. C-index (fold)"), ("train_reid", "APD discr. loss")]
    for ax, (col, title), tag in zip(axs.flat, panels, "abcdefgh"):
        ax.set_title(f"({tag}) {title}", fontsize=8, loc="left", color=INK)
        if col is None:
            for c, color, lab in (("log_var_seg", C1, "seg"), ("log_var_cls", C2, "grade"), ("log_var_surv", C3, "surv")):
                m = band(ax, h, c, color)
                ax.annotate(lab, (m.index[-1], m.values[-1]), xytext=(3, 0), textcoords="offset points",
                            va="center", fontsize=7, color=INK)
            ax.set_xlim(0, 170)
        else:
            band(ax, h, col, C1)
        if col in ("val_acc", "val_cindex"):
            ax.axhline(0.5 if col == "val_cindex" else 28 / 48, color=MUTED, linewidth=0.8, linestyle="--")
        ax.set_xlabel("Epoch")
    fig.suptitle(f"Mean $\\pm$ SD over {n} runs (5 folds $\\times$ 3 seeds); validation curves were not used for "
                 "model selection", fontsize=7, color=MUTED)
    fig.savefig(safe_out(out / name))
    plt.close(fig)
    print(f"training figure: {n} runs, {h.epoch.max()} epochs")


def seg_figure(runs, cache, pids, out):
    fig, axs = plt.subplots(1, len(pids), figsize=(7.2, 2.7), constrained_layout=True)
    for ax, pid in zip(np.atleast_1d(axs), pids):
        img = np.load(cache / pid / "image.npy")[0]                       # T2FS (mask reference series)
        gt = np.load(cache / pid / "mask.npy").astype(bool)
        f = sorted(runs.glob(f"seed42/fold*/oof_seg/{pid}.npz"))[0]
        pr = np.load(f)["pred"].astype(bool)
        dice = 2 * (pr & gt).sum() / (pr.sum() + gt.sum())
        z = int(gt.sum((1, 2)).argmax())
        ax.imshow(img[z], cmap="gray", vmin=0, vmax=1)
        ax.contour(gt[z], levels=[0.5], colors=[C3], linewidths=1.2)
        ax.contour(pr[z], levels=[0.5], colors=[C2], linewidths=1.2)
        ax.set_title(f"{pid}: Dice {dice:.2f} (seed 42)", fontsize=8, color=INK)
        ax.axis("off")
    h = [plt.Line2D([], [], color=C3, label="Reference"), plt.Line2D([], [], color=C2, label="Prediction")]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, fontsize=7)
    fig.savefig(safe_out(out / "Fig_10_segmentation_v2.pdf"))
    plt.close(fig)


def seg_grid_figure(runs, cache, pids, out, ncols=4, name="Fig_10_segmentation_v3.pdf"):
    """More segmentation examples: patients given in rank order of seed-averaged Dice (best to worst)."""
    nrows = int(np.ceil(len(pids) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(7.2, 1.95 * nrows + 0.3), constrained_layout=True)
    for ax in axs.flat:
        ax.axis("off")
    for ax, pid in zip(axs.flat, pids):
        img = np.load(cache / pid / "image.npy")[0]                       # T2FS (mask reference series)
        gt = np.load(cache / pid / "mask.npy").astype(bool)
        pr = np.load(sorted(runs.glob(f"seed42/fold*/oof_seg/{pid}.npz"))[0])["pred"].astype(bool)
        dice = 2 * (pr & gt).sum() / (pr.sum() + gt.sum())
        z = int(gt.sum((1, 2)).argmax())
        ax.imshow(img[z], cmap="gray", vmin=0, vmax=1)
        ax.contour(gt[z], levels=[0.5], colors=[C3], linewidths=1.0)
        if pr[z].any():
            ax.contour(pr[z], levels=[0.5], colors=[C2], linewidths=1.0)
        ax.set_title(f"{pid}  Dice {dice:.2f}", fontsize=7.5, color=INK)
    h = [plt.Line2D([], [], color=C3, label="Reference"), plt.Line2D([], [], color=C2, label="Prediction")]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.01))
    fig.savefig(safe_out(out / name), bbox_inches="tight")
    plt.close(fig)


def seg_error_figure(runs, cache, pids, out, name="Fig_10_segmentation_v5.pdf", per_row=2):
    """High-contrast segmentation figure: per patient T2FS | reference | prediction (+Dice) | error map
    (TP green, FP red, FN blue), cropped around the tumour. Seed-42 final-epoch out-of-fold predictions."""
    from matplotlib.patches import Patch
    GT_C, PR_C, TP_C, FP_C, FN_C = "#2ECC71", "#E74C3C", "#2ECC71", "#E74C3C", "#3B6FD8"
    n = len(pids); rows = int(np.ceil(n / per_row))
    ratios = []
    for k in range(per_row):
        ratios += [1, 1, 1, 1] + ([0.18] if k < per_row - 1 else [])
    fig, axs = plt.subplots(rows, len(ratios), figsize=(7.2, rows * 0.95 + 0.55),
                            gridspec_kw={"width_ratios": ratios, "wspace": 0.04, "hspace": 0.22})
    axs = np.atleast_2d(axs)
    for ax in axs.flat:
        ax.axis("off")

    def rgba(mask, color, alpha):
        c = matplotlib.colors.to_rgba(color)
        a = np.zeros(mask.shape + (4,)); a[mask] = (*c[:3], alpha)
        return a

    for idx, pid in enumerate(pids):
        r, k = divmod(idx, per_row)
        c0 = k * 5
        img = np.load(cache / pid / "image.npy")[0]                       # T2FS (mask reference series)
        gt = np.load(cache / pid / "mask.npy").astype(bool)
        pr = np.load(sorted(runs.glob(f"seed42/fold*/oof_seg/{pid}.npz"))[0])["pred"].astype(bool)
        dice = 2 * (pr & gt).sum() / (pr.sum() + gt.sum())
        z = int(gt.sum((1, 2)).argmax())
        im, g, p = img[z], gt[z], pr[z]
        ys, xs = np.where(g | p)
        cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
        half = max(ys.max() - ys.min(), xs.max() - xs.min()) / 2 * 1.3 + 4
        half = int(min(max(half, 22), min(im.shape) / 2))
        y0 = int(np.clip(round(cy - half), 0, im.shape[0] - 2 * half)); x0 = int(np.clip(round(cx - half), 0, im.shape[1] - 2 * half))
        sl = (slice(y0, y0 + 2 * half), slice(x0, x0 + 2 * half))
        im, g, p = im[sl], g[sl], p[sl]
        panels = axs[r, c0:c0 + 4]
        for ax in panels:
            ax.imshow(im, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        panels[1].imshow(rgba(g, GT_C, 0.40), interpolation="nearest")
        panels[1].contour(g, levels=[0.5], colors=[GT_C], linewidths=0.9)
        panels[2].imshow(rgba(p, PR_C, 0.40), interpolation="nearest")
        if p.any():
            panels[2].contour(p, levels=[0.5], colors=[PR_C], linewidths=0.9)
        err = rgba(g & p, TP_C, 0.70) + rgba(p & ~g, FP_C, 0.80) + rgba(g & ~p, FN_C, 0.75)
        panels[3].imshow(err, interpolation="nearest")
        badge = "#2E9E5B" if dice >= 0.65 else ("#E08A1E" if dice >= 0.40 else "#C0392B")
        panels[2].text(0.97, 0.04, f"Dice {dice:.2f}", transform=panels[2].transAxes, ha="right", va="bottom",
                       fontsize=5.5, color="white", fontweight="bold",
                       bbox={"boxstyle": "round,pad=0.2", "fc": badge, "ec": "none"})
        panels[0].text(0.03, 0.96, pid.replace("_", " "), transform=panels[0].transAxes, ha="left", va="top",
                       fontsize=5.5, color="white", fontweight="bold")
        if r == 0:
            for ax, t in zip(panels, ("T2FS", "Reference", "Prediction", "Error map")):
                ax.set_title(t, fontsize=7, color=INK, pad=3)
    h = [Patch(facecolor=TP_C, label="True positive"), Patch(facecolor=FP_C, label="False positive"),
         Patch(facecolor=FN_C, label="False negative"),
         plt.Line2D([], [], color=GT_C, label="Reference contour"), plt.Line2D([], [], color=PR_C, label="Prediction contour")]
    fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.01, right=0.99, top=1 - 0.28 / (rows * 0.95 + 0.55), bottom=0.38 / (rows * 0.95 + 0.55))
    fig.savefig(safe_out(out / name))
    plt.close(fig)


def km_figure(runs, out, name="Fig_11_km_v2.pdf"):
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    oof = pd.concat([pd.read_csv(f) for f in sorted(runs.glob("seed*/fold*/oof_predictions.csv"))])
    tr = []
    for f in sorted(runs.glob("seed*/fold*/train_predictions.csv")):
        o = pd.read_csv(f.parent / "oof_predictions.csv")
        tr.append(pd.read_csv(f).assign(seed=o.seed.iloc[0], fold=o.fold.iloc[0]))
    grp = ev.km_groups(oof, pd.concat(tr))
    pat = oof.groupby("pid")[["time", "event"]].first().join(grp.rename("high"))
    t, e, g = pat.time.values / 365.25, pat.event.values.astype(int), pat.high.values
    p = logrank_test(t[g == 1], t[g == 0], e[g == 1], e[g == 0]).p_value
    fig, ax = plt.subplots(figsize=(3.5, 2.8), constrained_layout=True)
    for val, color, grp_name in ((1, C2, "High predicted risk"), (0, C1, "Low predicted risk")):
        k = KaplanMeierFitter().fit(t[g == val], e[g == val],
                                    label=f"{grp_name} (n = {(g == val).sum()}, {e[g == val].sum()} deaths)")
        k.plot_survival_function(ax=ax, color=color, ci_show=True, ci_alpha=0.12, show_censors=True,
                                 censor_styles={"ms": 4, "marker": "|"})
    ax.set_xlabel("Years from diagnosis"); ax.set_ylabel("Overall survival")
    ax.set_ylim(0, 1.02); ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    ax.set_title(f"Log-rank p = {p:.2f}", fontsize=8, loc="left", color=INK)
    fig.savefig(safe_out(out / name))
    plt.close(fig)
    print(f"KM: high {int(g.sum())} / low {int((1 - g).sum())}, log-rank p = {p:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True); ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--seg_pids", nargs="+", required=True)
    ap.add_argument("--style", choices=["v2", "v3"], default="v2",
                    help="v3: diagram-matched style; writes Fig_09_training_v3, Fig_10_segmentation_v4 (grid), Fig_11_km_v3")
    ap.add_argument("--seg_error_only", action="store_true",
                    help="with --style v3: only the high-contrast segmentation figure (Fig_10_segmentation_v5.pdf)")
    a = ap.parse_args()
    runs, out = Path(a.runs), Path(a.out)
    if a.style == "v3" and a.seg_error_only:
        use_v3_style()
        seg_error_figure(runs, Path(a.cache), a.seg_pids, out)
        return
    if a.style == "v3":
        use_v3_style()
        training_figure(runs, out, "Fig_09_training_v3.pdf")
        seg_grid_figure(runs, Path(a.cache), a.seg_pids, out, name="Fig_10_segmentation_v4.pdf")
        km_figure(runs, out, "Fig_11_km_v3.pdf")
        return
    training_figure(runs, out)
    seg_figure(runs, Path(a.cache), a.seg_pids, out)
    km_figure(runs, out)


if __name__ == "__main__":
    main()
