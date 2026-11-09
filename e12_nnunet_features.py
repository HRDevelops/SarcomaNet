"""
e12_nnunet_features.py — E12 pilot, step 1 (PROTOCOL §11, 2026-11-02). Runs in venv_nnunet311.

    python e12_nnunet_features.py --results $nnUNet_results --raw $nnUNet_raw --cache_dir ../cache_48x96x96 \
        --extra_dir ../nnunet/E12_extra --out ../results/E12_nnunet_features.npz [--dry]

For each outer fold k (seed 42), the E10 nnU-Net fold-k model (checkpoint_final.pth) encodes all 51 patients after
nnU-Net's own preprocessing; feature = global average of the bottleneck (last encoder stage). SEG patients use the
Dataset501 raw images; the 3 non-SEG patients are written from the cache to --extra_dir with the same routine.
Output: feats (5, 51, C), pids, plus a JSON sidecar with provenance. Never overwrites.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from nnunet_e10 import CONF, DATASET, PLANS, SEED, TRAINER, load_splits, write_nii


def image_files(pid, raw, extra, cache_dir, seg):
    if seg:
        return [str(Path(raw) / DATASET / "imagesTr" / f"{pid}_{c:04d}.nii.gz") for c in range(3)]
    out = []
    for c in range(3):
        f = Path(extra) / f"{pid}_{c:04d}.nii.gz"
        if not f.exists():
            img = np.load(Path(cache_dir) / pid / "image.npy")
            meta = json.loads((Path(cache_dir) / pid / "meta.json").read_text())
            f.parent.mkdir(parents=True, exist_ok=True)
            write_nii(img[c].astype(np.float32), meta["resampled_spacing_mm"], f)
        out.append(str(f))
    return out


def main():
    ap = argparse.ArgumentParser()
    for k in ("results", "raw", "cache_dir", "extra_dir", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--dry", action="store_true", help="fold 0, 3 patients, *_DRYRUN output")
    a = ap.parse_args()
    out = Path(a.out.replace(".npz", "_DRYRUN.npz") if a.dry else a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    sp = load_splits(); pids = list(sp["patients"]); folds = range(sp["n_folds"])
    if a.dry:
        pids, folds = pids[:2] + [p for p in pids if not sp["has_mask"][p]][:1], range(1)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = Path(a.results) / DATASET / f"{TRAINER}__{PLANS}__{CONF}"
    feats, shapes = [], {}
    for k in folds:
        pr = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False, device=dev, verbose=False)
        pr.initialize_from_trained_model_folder(str(model_dir), use_folds=(k,), checkpoint_name="checkpoint_final.pth")
        net = pr.network
        net.load_state_dict(pr.list_of_parameters[0])
        net.to(dev).eval()
        prep = pr.configuration_manager.preprocessor_class(verbose=False)
        Fk = []
        for p in pids:
            files = image_files(p, a.raw, a.extra_dir, a.cache_dir, bool(sp["has_mask"][p]))
            data, _, _ = prep.run_case(files, None, pr.plans_manager, pr.configuration_manager, pr.dataset_json)
            shapes.setdefault(p, list(data.shape))
            with torch.no_grad():
                skips = net.encoder(torch.from_numpy(np.asarray(data, dtype=np.float32))[None].to(dev))
            Fk.append(skips[-1].float().mean(dim=(2, 3, 4))[0].cpu().numpy())
        feats.append(np.stack(Fk)); print(f"fold {k}: {len(Fk)} patients, feature dim {Fk[0].shape[0]}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, feats=np.stack(feats), pids=np.array(pids))
    from revision_utils import git_state
    Path(str(out).replace(".npz", ".json")).write_text(json.dumps({
        "stage": "E12_nnunet_features", "dry_run": a.dry, "seed": SEED, "trainer": TRAINER, "plans": PLANS,
        "configuration": CONF, "checkpoint": "checkpoint_final.pth", "feature": "GAP of last encoder stage",
        "n_folds": len(list(folds)), "n_patients": len(pids), "preprocessed_shapes": shapes,
        "git": git_state()}, indent=1, default=str))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
