"""
data_pipeline.py — TCIA Soft Tissue Sarcoma data loader

Verified against the actual INFOclinical_STS.xlsx schema:
  Col 0  Patient ID
  Col 1  Age
  Col 2  Sex
  Col 3  Histological type
  Col 4  MSKCC type  (7 classes → n_cls_classes=7)
  Col 5  Site of primary STS
  Col 6  Grade  (High / Intermediate / Low → 3 classes)
  Col 11 Time … diagnosis to outcome (days)
  Col 12 Status (NED, AWD, D)
  Col 13 Time … diagnosis to last follow-up (days)

Clinical feature vector (array-revision, PROTOCOL D-9):
    v2 (default, dim 10): [age/100, sex_bin, site_bin, one-hot MSKCC (7)]
    v1 (sensitivity, dim 6): [age/100, sex_bin, mskcc_idx/6, site_bin, mri2pet/100, 0]

Labels (array-revision):
    grade  : 0 Low / 1 Intermediate / 2 High; -1 = missing ("N/A"), excluded (D-2, bug B4)
    OS     : event = Status "D"; time = diagnosis -> last follow-up for every patient (D-1, bug B5)
    LMFS   : (sensitivity) event = lung metastasis; time = diagnosis -> outcome if event,
             else diagnosis -> last follow-up
"""

import re
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple


import numpy as np
import pandas as pd
import pydicom
import scipy.ndimage as ndi
import torch
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")


# ─── Label Maps ───────────────────────────────────────────────────────────────

GRADE_MAP = {"low": 0, "intermediate": 1, "high": 2}

MSKCC_MAP = {
    "mfh":                       0,
    "liposarcoma":               1,
    "leiomyosarcoma":            2,
    "synovial sarcoma":          3,
    "extraskeletal bone sarcoma":4,
    "fibrosarcoma":              5,
    "other":                     6,
}
N_MSKCC_CLASSES = 7

# Broad anatomical site → binary  (lower-limb=1, other=0)
LOWER_LIMB_KEYWORDS = ("thigh", "leg", "knee", "groin", "inguinal",
                       "femur", "popliteal", "hip", "buttock")


# ─── DICOM Utilities ──────────────────────────────────────────────────────────

def load_dicom_series(folder: Path) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Load a DICOM series folder → (volume, pixel_spacing, slice_thickness).
    volume : (D, H, W) float32, with RescaleSlope/Intercept applied.
    """
    dcm_files = sorted(Path(folder).glob("*.dcm"))
    if not dcm_files:
        raise FileNotFoundError(f"No .dcm files in {folder}")

    slices = [pydicom.dcmread(str(f)) for f in dcm_files]
    try:
        slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    except AttributeError:
        slices.sort(key=lambda s: float(getattr(s, "SliceLocation", 0)))

    ps  = np.array([float(x) for x in slices[0].PixelSpacing], dtype=np.float32)
    th  = float(getattr(slices[0], "SliceThickness", 1.0))

    volume = np.stack([s.pixel_array.astype(np.float32) for s in slices])

    for i, s in enumerate(slices):
        slope     = float(getattr(s, "RescaleSlope",     1.0))
        intercept = float(getattr(s, "RescaleIntercept", 0.0))
        volume[i] = volume[i] * slope + intercept

    return volume, ps, th


def _sorted_slices(folder: Path) -> List:
    dcm_files = sorted(Path(folder).glob("*.dcm"))
    slices    = [pydicom.dcmread(str(f)) for f in dcm_files]
    try:
        slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    except AttributeError:
        pass
    return slices


def parse_rtstruct(
    rtstruct_path:    Path,
    ref_volume_shape: Tuple[int, int, int],
    ref_slices:       List,
    roi_pattern:      str = r"(?i)(tumor|gtv|lesion|sarcoma|sts|mass)",
) -> np.ndarray:
    """Rasterise the first matching RT-struct ROI into a binary mask (D,H,W)."""
    from PIL import Image, ImageDraw

    rs   = pydicom.dcmread(str(rtstruct_path))
    mask = np.zeros(ref_volume_shape, dtype=np.uint8)

    # Select target ROI
    target_roi = None
    for roi in rs.StructureSetROISequence:
        if re.search(roi_pattern, roi.ROIName):
            target_roi = roi.ROINumber
            break
    if target_roi is None:
        target_roi = rs.StructureSetROISequence[0].ROINumber

    # z-pos → slice index map
    z_to_idx: Dict[float, int] = {}
    for idx, sl in enumerate(ref_slices):
        try:
            z_to_idx[round(float(sl.ImagePositionPatient[2]), 2)] = idx
        except AttributeError:
            pass

    D, H, W = ref_volume_shape

    for roi_contour in rs.ROIContourSequence:
        if roi_contour.ReferencedROINumber != target_roi:
            continue
        if not hasattr(roi_contour, "ContourSequence"):
            continue

        for contour in roi_contour.ContourSequence:
            pts = np.array(contour.ContourData, dtype=np.float64).reshape(-1, 3)
            z   = round(float(pts[0, 2]), 2)
            if z not in z_to_idx:
                continue

            sl  = ref_slices[z_to_idx[z]]
            ipp = np.array([float(v) for v in sl.ImagePositionPatient])
            ps  = np.array([float(v) for v in sl.PixelSpacing])

            col = np.clip(((pts[:, 0] - ipp[0]) / ps[1]).astype(int), 0, W - 1)
            row = np.clip(((pts[:, 1] - ipp[1]) / ps[0]).astype(int), 0, H - 1)

            img  = Image.new("L", (W, H), 0)
            poly = list(zip(col.tolist(), row.tolist()))
            if len(poly) >= 3:
                ImageDraw.Draw(img).polygon(poly, fill=1)
            mask[z_to_idx[z]] |= np.array(img, dtype=np.uint8)

    return mask


# ─── Normalisation ────────────────────────────────────────────────────────────

def normalise_ct(v: np.ndarray, clip: Tuple = (-200, 300)) -> np.ndarray:
    lo, hi = clip
    return (np.clip(v, lo, hi) - lo) / (hi - lo + 1e-8)


def normalise_mri(v: np.ndarray) -> np.ndarray:
    p1, p99 = np.percentile(v, [1, 99])
    return (np.clip(v, p1, p99) - p1) / (p99 - p1 + 1e-8)


def normalise_pet(v: np.ndarray) -> np.ndarray:
    p99 = np.percentile(v, 99)
    return np.clip(v / (p99 + 1e-8), 0.0, 1.0)


def resize_volume(v: np.ndarray, target: Tuple[int, int, int], order: int = 1) -> np.ndarray:
    factors = [t / s for t, s in zip(target, v.shape)]
    return ndi.zoom(v, factors, order=order)


# ─── Patient Discovery ────────────────────────────────────────────────────────

# def discover_modalities(patient_dir: Path) -> Dict[str, Optional[Path]]:
#     """Walk patient folder → dict of key modality paths."""
#     paths: Dict[str, Optional[Path]] = {
#         "mri_t2fs": None, "mri_t1": None,
#         "pet": None, "ct": None,
#         "rtstruct_t2fs": None, "rtstruct_pet": None,
#     }

#     for series_dir in sorted(Path(patient_dir).rglob("*")):
#         if not series_dir.is_dir():
#             continue
#         dcms = list(series_dir.glob("*.dcm"))
#         if not dcms:
#             continue
#         try:
#             ds  = pydicom.dcmread(str(dcms[0]), stop_before_pixels=True)
#             mod = getattr(ds, "Modality", "").upper()
#         except Exception:
#             continue

#         name = series_dir.name.lower()

#         if mod == "MR":
#             if "t2" in name and any(k in name for k in ("fat", "fs", "fatsat")):
#                 paths["mri_t2fs"] = series_dir
#             elif "t1" in name and "aligned" not in name:
#                 paths["mri_t1"]   = series_dir
#         elif mod == "PT":
#             paths["pet"] = series_dir
#         elif mod == "CT":
#             paths["ct"]  = series_dir
#         elif mod == "RTSTRUCT":
#             rtstruct_file = dcms[0]
#             if "t2" in name:
#                 paths["rtstruct_t2fs"] = rtstruct_file
#             elif "pet" in name and "ct" not in name:
#                 paths["rtstruct_pet"]  = rtstruct_file

#     return paths

def discover_modalities(patient_dir: Path) -> Dict[str, Optional[Path]]:
    """Walk patient folder → dict of key modality paths."""
    paths: Dict[str, Optional[Path]] = {
        "mri_t2fs": None, "mri_t1": None, "mri_any": None,
        "pet": None, "ct": None,
        "rtstruct_t2fs": None, "rtstruct_pet": None, "rtstruct_any": None,
    }

    for series_dir in sorted(Path(patient_dir).rglob("*")):
        if not series_dir.is_dir():
            continue
        dcms = list(series_dir.glob("*.dcm"))
        if not dcms:
            continue
        try:
            ds  = pydicom.dcmread(str(dcms[0]), stop_before_pixels=True)
            mod = getattr(ds, "Modality", "").upper()
            sdesc = getattr(ds, "SeriesDescription", "").lower()
        except Exception:
            continue

        name = series_dir.name.lower()
        # Combine folder name and SeriesDescription for robust matching
        combined = name + " " + sdesc

        if mod == "MR":
            paths["mri_any"] = series_dir
            is_t2 = any(k in combined for k in ("t2", "stir", "tirm"))
            is_fat = any(k in combined for k in ("fat", "fs", "fatsat", "dixon", "spair", "stir", "tirm"))
            
            if is_t2 and is_fat:
                paths["mri_t2fs"] = series_dir
            elif "t1" in combined and "aligned" not in combined:
                paths["mri_t1"]   = series_dir
        elif mod == "PT":
            paths["pet"] = series_dir
        elif mod == "CT":
            paths["ct"]  = series_dir
        elif mod == "RTSTRUCT":
            rtstruct_file = dcms[0]
            paths["rtstruct_any"] = rtstruct_file
            if any(k in combined for k in ("t2", "mri", "stir", "tirm")):
                paths["rtstruct_t2fs"] = rtstruct_file
            elif "pet" in combined and "ct" not in combined:
                paths["rtstruct_pet"]  = rtstruct_file

    return paths


# ─── Clinical Feature Builder ─────────────────────────────────────────────────

CLINICAL_DIMS = {"v1": 6, "v2": 3 + N_MSKCC_CLASSES}
GRADE_MISSING = -1


class ClinicalBuilder:
    """
    Parses INFOclinical_STS.xlsx and returns per-patient
    (clinical_vec, grade_label, surv_time, surv_event).

    array-revision changes (see BUGS.md B4, B5, PROTOCOL D-1, D-2, D-9):
      * grade "N/A" -> GRADE_MISSING (-1), never silently mapped to Low
      * endpoint selectable: "os" (default) or "lmfs"; the old hybrid
        (death event timed at recurrence/metastasis) is no longer available
      * clinical vector version "v2" (default) or "v1"
    """

    def __init__(self, xls_path: Path, endpoint: str = "os", clinical_version: str = "v2"):
        if endpoint not in ("os", "lmfs"):
            raise ValueError(f"unknown endpoint {endpoint!r}")
        if clinical_version not in CLINICAL_DIMS:
            raise ValueError(f"unknown clinical_version {clinical_version!r}")
        self.endpoint         = endpoint
        self.clinical_version = clinical_version
        self.CLINICAL_DIM     = CLINICAL_DIMS[clinical_version]
        df = pd.read_excel(str(xls_path), sheet_name=0)
        # Access columns by position to avoid encoding issues in names
        self.pid_col      = df.iloc[:, 0]
        self.age_col      = pd.to_numeric(df.iloc[:, 1],  errors="coerce").fillna(50)
        self.sex_col      = df.iloc[:, 2].str.strip().str.lower()
        self.mskcc_col    = df.iloc[:, 4].str.strip().str.lower()
        self.site_col     = df.iloc[:, 5].str.strip().str.lower()
        self.grade_col    = df.iloc[:, 6].str.strip().str.lower()
        self.mri2pet_col  = pd.to_numeric(df.iloc[:, 8],  errors="coerce").fillna(0)
        self.outcome_time = pd.to_numeric(df.iloc[:, 11], errors="coerce")
        self.status_col   = df.iloc[:, 12].str.strip().str.upper()
        self.followup_col = pd.to_numeric(df.iloc[:, 13], errors="coerce")
        self.outcome_col  = df.iloc[:, 10].astype(str).str.strip().str.lower()

        # Index by Patient ID
        self.index = {str(pid).strip(): i for i, pid in enumerate(self.pid_col)}

    def grade(self, pid: str) -> int:
        """0/1/2, or GRADE_MISSING for 'N/A' or any unrecognised value."""
        return GRADE_MAP.get(str(self.grade_col.iloc[self.index[pid]]), GRADE_MISSING)

    def survival(self, pid: str, endpoint: Optional[str] = None) -> Tuple[float, float]:
        """(time_days, event) for the requested endpoint (default: self.endpoint)."""
        ep = endpoint or self.endpoint
        i  = self.index[pid]
        followup = self.followup_col.iloc[i]
        if pd.isna(followup):
            raise ValueError(f"{pid}: missing last follow-up time")
        if ep == "os":
            event = 1.0 if str(self.status_col.iloc[i]) == "D" else 0.0
            time  = float(followup)
        else:  # lmfs
            event = 1.0 if "lung" in self.outcome_col.iloc[i] else 0.0
            ot    = self.outcome_time.iloc[i]
            if event and pd.isna(ot):
                raise ValueError(f"{pid}: lung metastasis without outcome time")
            time  = float(ot) if event else float(followup)
        return max(time, 1.0), event

    def clinical_vector(self, pid: str, version: Optional[str] = None) -> np.ndarray:
        ver = version or self.clinical_version
        i   = self.index[pid]
        age    = float(self.age_col.iloc[i]) / 100.0
        sex    = 1.0 if self.sex_col.iloc[i] == "male" else 0.0
        mskcc  = MSKCC_MAP.get(str(self.mskcc_col.iloc[i]), 6)
        site_b = 1.0 if any(k in str(self.site_col.iloc[i]) for k in LOWER_LIMB_KEYWORDS) else 0.0
        if ver == "v1":
            mri2pet = float(self.mri2pet_col.iloc[i]) / 100.0
            return np.array([age, sex, mskcc / 6.0, site_b, mri2pet, 0.0], dtype=np.float32)
        onehot = np.zeros(N_MSKCC_CLASSES, dtype=np.float32)
        onehot[mskcc] = 1.0
        return np.concatenate([np.array([age, sex, site_b], dtype=np.float32), onehot])

    def get(self, pid: str) -> Tuple[np.ndarray, int, float, float]:
        """Return (clinical_vec, grade_label (-1 = missing), surv_time, surv_event)."""
        if pid not in self.index:
            raise KeyError(f"{pid} not in clinical spreadsheet")
        time, event = self.survival(pid)
        return self.clinical_vector(pid), self.grade(pid), time, event


# ─── Dataset ──────────────────────────────────────────────────────────────────

class SarcomaDataset(Dataset):
    """
    TCIA Soft-Tissue-Sarcoma PyTorch Dataset.

    Fast path: if cache/<pid>/image.npy and mask.npy exist (written by
    preprocess.py), they are loaded directly — much faster than re-reading DICOMs.
    Slow path: falls back to live DICOM loading.

    Returns dict keys:
        image       : (3, D, H, W) float32  — [T2FS, PET, CT]
        seg_mask    : (D, H, W)    int64    — binary tumour mask
        cls_label   : ()           int64    — grade  (0=Low,1=Int,2=High)
        surv_time   : ()           float32  — days to event / last follow-up
        surv_event  : ()           float32  — 1=dead, 0=censored
        clinical    : (C,)         float32  — tabular clinical features (no grade leakage)
        has_mask    : ()           float32  — 1 if the patient is in the SEG analysis set
        has_grade   : ()           float32  — 1 if the grade is known (GRADE set)
        patient_id  : str

    array-revision: `has_mask` comes from splits.json (PROTOCOL D-3), never from
    the cache content; cls_label is clamped to 0 when missing and must be ignored
    via has_grade. With require_cache=True (training default) a missing cache
    entry is an error instead of a silent live-DICOM fallback.
    """

    CACHE_DIR = Path("cache")   # relative to CWD; overridden by SarcomaDataset.cache_dir

    def __init__(
        self,
        root:         Path,
        clinical_xls: Path,
        patient_ids:  List[str],
        augment:      bool               = False,
        target_shape: Tuple[int,int,int] = (48, 96, 96),
        cache_dir:    Optional[Path]     = None,
        has_mask:     Optional[Dict[str, bool]] = None,
        endpoint:     str                = "os",
        clinical_version: str            = "v2",
        require_cache: bool              = True,
    ):
        self.root         = Path(root)
        self.patient_ids  = patient_ids
        self.augment      = augment
        self.target_shape = tuple(target_shape)
        self.clinical     = ClinicalBuilder(Path(clinical_xls), endpoint=endpoint,
                                            clinical_version=clinical_version)
        self.cache_dir    = Path(cache_dir) if cache_dir else self.CACHE_DIR
        self.has_mask     = has_mask or {}
        self.require_cache = require_cache

    def __len__(self) -> int:
        return len(self.patient_ids)

    # ── Loading helpers ───────────────────────────────────────────────────────

    def _load_channel(self, folder: Optional[Path], norm_fn) -> np.ndarray:
        if folder is None:
            return np.zeros(self.target_shape, dtype=np.float32)
        try:
            vol, _, _ = load_dicom_series(folder)
            return resize_volume(norm_fn(vol), self.target_shape).astype(np.float32)
        except Exception as e:
            warnings.warn(f"Load failed {folder}: {e}")
            return np.zeros(self.target_shape, dtype=np.float32)

    # ── Augmentation ──────────────────────────────────────────────────────────

    def _augment(
        self, image: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Random axis flip
        for ax in range(1, 4):
            if random.random() > 0.5:
                image = np.flip(image, axis=ax)
                mask  = np.flip(mask,  axis=ax - 1)

        # Random rotation ±15° in axial plane
        if random.random() > 0.5:
            angle = random.uniform(-15.0, 15.0)
            for c in range(image.shape[0]):
                image[c] = ndi.rotate(image[c], angle, axes=(1, 2), reshape=False, order=1)
            mask = (
                ndi.rotate(mask.astype(np.float32), angle, axes=(1, 2), reshape=False, order=0) > 0.5
            ).astype(np.int64)

        # Gaussian noise
        if random.random() > 0.5:
            image = np.clip(
                image + np.random.normal(0, 0.02, image.shape).astype(np.float32),
                0.0, 1.0,
            )

        # Brightness/contrast scale
        if random.random() > 0.5:
            image = np.clip(image * random.uniform(0.85, 1.15), 0.0, 1.0)

        return image.copy().astype(np.float32), mask.copy()

    # ── Main __getitem__ ─────────────────────────────────────────────────────

    def __getitem__(self, idx: int) -> Dict:
        pid      = self.patient_ids[idx]
        img_path = self.cache_dir / pid / "image.npy"
        msk_path = self.cache_dir / pid / "mask.npy"

        # ── Fast path: load from preprocessed cache ───────────────────────────
        if img_path.exists() and msk_path.exists():
            image    = np.load(str(img_path)).astype(np.float32)   # (3,D,H,W)
            seg_mask = np.load(str(msk_path)).astype(np.int64)     # (D,H,W)
            # Resize if cached at different resolution
            if tuple(image.shape[1:]) != self.target_shape:
                ch = [resize_volume(image[c], self.target_shape) for c in range(3)]
                image    = np.stack(ch, axis=0).astype(np.float32)
                seg_mask = (
                    resize_volume(seg_mask.astype(np.float32), self.target_shape) > 0.5
                ).astype(np.int64)

        # ── Slow path: live DICOM loading ─────────────────────────────────────
        elif self.require_cache:
            raise FileNotFoundError(f"{pid}: no cache entry in {self.cache_dir} (run preprocess.py)")
        else:
            paths = discover_modalities(self.root / pid)

            t2fs = self._load_channel(paths["mri_t2fs"], normalise_mri)
            pet  = self._load_channel(paths["pet"],      normalise_pet)
            ct   = self._load_channel(paths["ct"],       normalise_ct)
            image = np.stack([t2fs, pet, ct], axis=0).astype(np.float32)

            # seg_mask = np.zeros(self.target_shape, dtype=np.int64)
            # rtstruct = paths.get("rtstruct_t2fs") or paths.get("rtstruct_pet")

            # if rtstruct is not None and paths["mri_t2fs"] is not None:
            #     try:
            #         raw_vol, _, _ = load_dicom_series(paths["mri_t2fs"])
            #         ref_slices    = _sorted_slices(paths["mri_t2fs"])
            #         raw_mask      = parse_rtstruct(rtstruct, raw_vol.shape, ref_slices)
            #         seg_mask      = (
            #             resize_volume(raw_mask.astype(np.float32), self.target_shape) > 0.5
            #         ).astype(np.int64)
            #     except Exception as e:
            #         warnings.warn(f"{pid} rtstruct: {e}")

            seg_mask = np.zeros(self.target_shape, dtype=np.int64)
            
            # Fetch best available rtstruct
            rtstruct = paths.get("rtstruct_t2fs") or paths.get("rtstruct_pet") or paths.get("rtstruct_any")
            # 3-level fallback for the reference volume
            ref_path = paths.get("mri_t2fs") or paths.get("mri_any") or paths.get("pet") or paths.get("ct")

            if rtstruct is not None and ref_path is not None:
                try:
                    raw_vol, _, _ = load_dicom_series(ref_path)
                    ref_slices    = _sorted_slices(ref_path)
                    raw_mask      = parse_rtstruct(rtstruct, raw_vol.shape, ref_slices)
                    seg_mask      = (
                        resize_volume(raw_mask.astype(np.float32), self.target_shape) > 0.5
                    ).astype(np.int64)
                except Exception as e:
                    warnings.warn(f"{pid} rtstruct: {e}")

        # Clinical / labels
        clinical, cls_label, surv_time, surv_event = self.clinical.get(pid)
        has_grade = cls_label != GRADE_MISSING
        has_mask  = bool(self.has_mask.get(pid, False))
        if not has_mask:
            seg_mask = np.zeros_like(seg_mask)   # never used: seg loss/metrics are masked

        # Augment
        if self.augment:
            image, seg_mask = self._augment(image, seg_mask)

        return {
            "image":      torch.from_numpy(np.ascontiguousarray(image)),
            "seg_mask":   torch.from_numpy(np.ascontiguousarray(seg_mask)).long(),
            "has_mask":   torch.tensor(float(has_mask), dtype=torch.float32),
            "has_grade":  torch.tensor(float(has_grade), dtype=torch.float32),
            "cls_label":  torch.tensor(max(cls_label, 0), dtype=torch.long),
            "surv_time":  torch.tensor(surv_time,   dtype=torch.float32),
            "surv_event": torch.tensor(surv_event,  dtype=torch.float32),
            "clinical":   torch.tensor(clinical,    dtype=torch.float32),
            "patient_id": pid,
        }


# ─── Quick sanity check ───────────────────────────────────────────────────────

if __name__ == "__main__":
    ROOT    = Path("archive/manifest-MjbMt99Q1553106146386120388/Soft-tissue-Sarcoma")
    XLS     = Path("archive/INFOclinical_STS.xlsx")
    IDS     = sorted(p.name for p in ROOT.iterdir() if p.is_dir())[:3]

    ds = SarcomaDataset(ROOT, XLS, IDS, augment=True, target_shape=(32, 64, 64))
    print(f"Dataset size: {len(ds)}")

    sample = ds[0]
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:12s}: {tuple(v.shape)}  dtype={v.dtype}  "
                  f"min={v.float().min().item():.3f}  max={v.float().max().item():.3f}")
        else:
            print(f"  {k:12s}: {v}")
