"""
preprocess.py — One-time DICOM -> NumPy cache builder (array-revision).

    python preprocess.py --data_root <.../Soft-tissue-Sarcoma> --cache_dir cache_48x96x96

Per patient writes <cache_dir>/<pid>/:
    image.npy  (3, D, H, W) float32  [T2FS, PET, CT], each independently normalised
               and resized to the target grid (NOT co-registered; PROTOCOL D-11)
    mask.npy   (D, H, W) uint8       RT-STRUCT rasterised on the reference series,
               then resized to the target grid
    meta.json  which series fed each channel, which RT-STRUCT / reference series
               built the mask (D-11 sensitivity), physical voxel spacing of the
               resampled mask grid in mm (for HD95/ASSD), mask voxel count
and <cache_dir>/cache_manifest.json (summary, shape, script version).

Changes vs. the May 9 version (BUGS.md B16, B11):
  * native target shape (default 48x96x96), no 32x64x64 cache + on-the-fly upsampling
  * paths and shape are arguments; an existing non-empty cache dir is never overwritten
  * per-patient metadata recorded; method of loading/normalising/rasterising unchanged
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from tqdm import tqdm

from data_pipeline import (
    _sorted_slices,
    discover_modalities,
    load_dicom_series,
    normalise_ct,
    normalise_mri,
    normalise_pet,
    parse_rtstruct,
    resize_volume,
)

warnings.filterwarnings("ignore")
VERSION = "array-revision-1"


def physical_extent_mm(folder: Path, n_slices: int, ps) -> tuple:
    """(z, y, x) extent in mm of a DICOM series volume of shape (n_slices, rows, cols)."""
    slices = _sorted_slices(folder)
    rows, cols = int(slices[0].Rows), int(slices[0].Columns)
    try:
        z = [float(s.ImagePositionPatient[2]) for s in slices]
        dz = float(np.median(np.abs(np.diff(sorted(z))))) if len(z) > 1 else float(
            getattr(slices[0], "SliceThickness", 1.0))
    except AttributeError:
        dz = float(getattr(slices[0], "SliceThickness", 1.0))
    return (n_slices * dz, rows * float(ps[0]), cols * float(ps[1]))


def preprocess_patient(pat_dir: Path, shape: tuple) -> dict:
    paths = discover_modalities(pat_dir)
    meta  = {"pid": pat_dir.name, "version": VERSION, "target_shape": list(shape),
             "series": {k: (str(v.relative_to(pat_dir)) if v is not None else None)
                        for k, v in paths.items()}}

    def load_ch(folder, norm_fn, key):
        if folder is None:
            meta[f"{key}_present"] = False
            return np.zeros(shape, dtype=np.float32)
        try:
            vol, _, _ = load_dicom_series(folder)
            meta[f"{key}_present"] = True
            return resize_volume(norm_fn(vol), shape).astype(np.float32)
        except Exception as e:
            meta[f"{key}_present"] = False
            meta[f"{key}_error"]   = str(e)
            return np.zeros(shape, dtype=np.float32)

    image = np.stack([
        load_ch(paths["mri_t2fs"], normalise_mri, "t2fs"),
        load_ch(paths["pet"],      normalise_pet, "pet"),
        load_ch(paths["ct"],       normalise_ct,  "ct"),
    ], axis=0)

    mask = np.zeros(shape, dtype=np.uint8)
    rt_key  = next((k for k in ("rtstruct_t2fs", "rtstruct_pet", "rtstruct_any") if paths.get(k)), None)
    ref_key = next((k for k in ("mri_t2fs", "mri_any", "pet", "ct") if paths.get(k)), None)
    meta["mask_rtstruct_source"] = rt_key
    meta["mask_reference"]       = ref_key
    if rt_key and ref_key:
        try:
            raw_vol, ps, _ = load_dicom_series(paths[ref_key])
            raw_mask = parse_rtstruct(paths[rt_key], raw_vol.shape, _sorted_slices(paths[ref_key]))
            mask = (resize_volume(raw_mask.astype(np.float32), shape) > 0.5).astype(np.uint8)
            ext = physical_extent_mm(paths[ref_key], raw_vol.shape[0], ps)
            meta["reference_shape"]        = list(raw_vol.shape)
            meta["reference_extent_mm"]    = [round(e, 3) for e in ext]
            meta["resampled_spacing_mm"]   = [round(e / t, 4) for e, t in zip(ext, shape)]
        except Exception as e:
            meta["mask_error"] = str(e)
    meta["mask_voxels"] = int(mask.sum())
    return {"image": image, "mask": mask, "meta": meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--shape", type=int, nargs=3, default=[48, 96, 96])
    ap.add_argument("--only", nargs="*", default=None, help="subset of patient IDs (testing)")
    a = ap.parse_args()

    data_root, cache_dir, shape = Path(a.data_root), Path(a.cache_dir), tuple(a.shape)
    if cache_dir.exists() and any(cache_dir.iterdir()):
        sys.exit(f"{cache_dir} exists and is not empty; refusing to overwrite. Use a new --cache_dir.")
    cache_dir.mkdir(parents=True, exist_ok=True)

    pids = sorted(p.name for p in data_root.iterdir() if p.is_dir())
    if a.only:
        pids = [p for p in pids if p in set(a.only)]
    print(f"Preprocessing {len(pids)} patients -> {cache_dir}/  shape={shape}")

    manifest = {"version": VERSION, "shape": list(shape), "data_root": str(data_root), "patients": {}}
    for pid in tqdm(pids, desc="Patients"):
        out = cache_dir / pid
        out.mkdir()
        try:
            r = preprocess_patient(data_root / pid, shape)
        except Exception as e:
            manifest["patients"][pid] = {"error": str(e)}
            tqdm.write(f"  {pid}  ERROR: {e}")
            continue
        np.save(str(out / "image.npy"), r["image"])
        np.save(str(out / "mask.npy"),  r["mask"])
        (out / "meta.json").write_text(json.dumps(r["meta"], indent=1), encoding="utf-8")
        m = r["meta"]
        manifest["patients"][pid] = {k: m.get(k) for k in (
            "t2fs_present", "pet_present", "ct_present", "mask_reference",
            "mask_rtstruct_source", "mask_voxels", "resampled_spacing_mm")}
        tqdm.write(f"  {pid}  T2FS={m.get('t2fs_present')} PET={m.get('pet_present')} "
                   f"CT={m.get('ct_present')} mask={m['mask_voxels']} ref={m.get('mask_reference')}")

    (cache_dir / "cache_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    ps = manifest["patients"].values()
    print(f"\nDone: {cache_dir.resolve()}")
    for k in ("t2fs_present", "pet_present", "ct_present"):
        print(f"  {k}: {sum(bool(p.get(k)) for p in ps)}/{len(pids)}")
    print(f"  non-empty masks: {sum((p.get('mask_voxels') or 0) > 0 for p in ps)}/{len(pids)}")


if __name__ == "__main__":
    main()
