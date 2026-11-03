"""
radiomics_extract.py — E7 step 1: PyRadiomics features from the cached model inputs (PROTOCOL §8, E7; §11).

Runs in the separate CPU venv `venv_radiomics` (pyradiomics 3.1.0 needs numpy < 2; requirements_radiomics.lock).

    python radiomics_extract.py --cache_dir ../cache_48x96x96 --out ../results/E7_radiomics_features.csv

Inputs are the SAME volumes the network sees (cache image.npy channels T2FS, PET; GT mask.npy; voxel spacing =
meta.json resampled_spacing_mm), so the comparison with the deep model is like for like. Consequence, stated in
the paper: PET is normalised per patient (÷ 99th percentile), so PET features are not absolute SUV.
Feature extraction is per patient and unsupervised (nothing is fitted), so it cannot leak across folds.
Patients with an empty mask get no row; the model step imputes them from training-fold medians.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor, logger

logger.setLevel(40)  # errors only

SETTINGS = {"binCount": 32, "label": 1, "minimumROIDimensions": 2, "correctMask": False,
            "normalize": False, "resampledPixelSpacing": None, "geometryTolerance": 1e-5}
CLASSES = ["firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]
CHANNELS = {"t2fs": 0, "pet": 1}


def to_sitk(a, spacing_zyx):
    im = sitk.GetImageFromArray(a)                       # numpy (z, y, x) -> sitk (x, y, z)
    im.SetSpacing(tuple(float(s) for s in spacing_zyx[::-1]))
    return im


def make_extractor(shape: bool):
    ex = featureextractor.RadiomicsFeatureExtractor(**SETTINGS)
    ex.disableAllFeatures(); ex.disableAllImageTypes(); ex.enableImageTypeByName("Original")
    for c in CLASSES:
        ex.enableFeatureClassByName(c)
    if shape:
        ex.enableFeatureClassByName("shape")
    return ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    ex_shape, ex = make_extractor(True), make_extractor(False)
    rows, skipped = [], []
    for d in sorted(p for p in Path(a.cache_dir).iterdir() if p.is_dir()):
        meta = json.loads((d / "meta.json").read_text())
        mask = np.load(d / "mask.npy").astype(np.uint8)
        if mask.sum() == 0:
            skipped.append(d.name); continue
        img = np.load(d / "image.npy").astype(np.float32)
        sp = meta["resampled_spacing_mm"]
        m = to_sitk(mask, sp)
        row = {"pid": d.name}
        for name, ch in CHANNELS.items():
            feats = (ex_shape if name == "t2fs" else ex).execute(to_sitk(img[ch], sp), m)
            for k, v in feats.items():
                if k.startswith("original_"):
                    key = k.replace("original_", f"{name}_") if not k.startswith("original_shape") \
                        else k.replace("original_", "")
                    row[key] = float(v)
        rows.append(row)
        print(d.name, len(row) - 1, "features")
    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    out.with_suffix(".json").write_text(json.dumps({"settings": SETTINGS, "classes": CLASSES, "channels": CHANNELS,
                                                    "n_patients": len(df), "n_features": df.shape[1] - 1,
                                                    "skipped_empty_mask": skipped}, indent=1))
    print(f"wrote {out}: {len(df)} patients x {df.shape[1] - 1} features; skipped {skipped}")


if __name__ == "__main__":
    main()
