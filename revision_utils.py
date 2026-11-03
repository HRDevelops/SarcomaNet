"""Shared helpers for the array-revision scripts: seeding, frozen splits, run metadata."""

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_splits(path) -> dict:
    sp = json.loads(Path(path).read_text(encoding="utf-8"))
    sp["_sha256"] = splits_sha256(sp)
    return sp


def splits_sha256(sp: dict) -> str:
    """Hash of the canonical JSON (independent of CRLF/LF and key order)."""
    clean = {k: v for k, v in sp.items() if not k.startswith("_")}
    return hashlib.sha256(json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def fold_ids(sp: dict, seed: int, fold: int):
    """(train_ids, val_ids) for one (seed, fold) of the frozen splits."""
    f = sp["folds"][str(seed)]
    train = [p for p in sp["patients"] if f[p] != fold]
    val   = [p for p in sp["patients"] if f[p] == fold]
    if not val:
        raise ValueError(f"empty validation fold {fold} for seed {seed}")
    return train, val


def _read_git_head(repo_dir: Path):
    """Commit and branch from .git without the git binary (HPC compute nodes have none)."""
    g = repo_dir / ".git"
    try:
        head = (g / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head, None
        ref = head[5:]
        f = g / ref
        if f.exists():
            return f.read_text().strip(), ref.split("/")[-1]
        for line in (g / "packed-refs").read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split()[0], ref.split("/")[-1]
    except OSError:
        pass
    return None, None


def git_state(repo_dir=None) -> dict:
    repo_dir = Path(repo_dir or Path(__file__).resolve().parent)
    def run(*a):
        try:
            return subprocess.check_output(["git", *a], cwd=repo_dir, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None
    commit = run("rev-parse", "HEAD")
    if commit is not None:
        return {"commit": commit, "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
                "dirty": bool(run("status", "--porcelain")), "source": "git"}
    commit, branch = _read_git_head(repo_dir)
    # dirty state unknown without git; the Slurm scripts refuse to start on a dirty tree (checked at submission)
    return {"commit": commit, "branch": branch, "dirty": None, "source": ".git files"}


def env_state() -> dict:
    return {"python": sys.version.split()[0], "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "host": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID")}


def write_run_meta(out_dir: Path, cfg: dict, splits: dict, extra: dict = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"config": cfg, "splits_sha256": splits["_sha256"], "git": git_state(), "env": env_state()}
    if extra:
        meta.update(extra)
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
