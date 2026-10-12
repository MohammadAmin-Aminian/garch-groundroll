"""Loading the bundled datasets and configurations."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.io as sio

__all__ = ["Dataset", "repo_root", "load_dataset", "load_config", "list_datasets", "save_mat"]


def repo_root() -> Path:
    """Repository root (``GARCH_GROUNDROLL_ROOT`` overrides auto-detection)."""
    env = os.environ.get("GARCH_GROUNDROLL_ROOT")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "configs").is_dir() and (parent / "data").is_dir():
            return parent
    return Path.cwd()


@dataclass
class Dataset:
    name: str
    data: np.ndarray
    dt: float
    dx: float
    description: str = ""
    fk_mask: np.ndarray | None = None
    extras: dict = field(default_factory=dict)

    @property
    def shape(self):
        return self.data.shape


def _scalar(v):
    return float(np.asarray(v).ravel()[0])


def _text(v):
    if isinstance(v, str):
        return v
    a = np.asarray(v).ravel()
    return str(a[0]) if a.size else ""


def load_dataset(name_or_path: str | os.PathLike) -> Dataset:
    """Load ``data/<name>.mat`` (or an explicit ``.mat`` path).

    The file must contain ``data`` (nt x nx), ``dt`` (s) and ``dx`` (m), and
    may contain ``fk_mask``, ``description`` and any other arrays (returned in
    ``extras``, e.g. ``signal`` for synthetic data).
    """
    p = Path(name_or_path)
    if p.suffix != ".mat":
        p = repo_root() / "data" / f"{name_or_path}.mat"
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")
    m = sio.loadmat(p)
    known = {"data", "dt", "dx", "fk_mask", "description"}
    extras = {k: np.asarray(v, dtype=float) for k, v in m.items()
              if not k.startswith("__") and k not in known}
    return Dataset(
        name=p.stem,
        data=np.asarray(m["data"], dtype=float),
        dt=_scalar(m["dt"]),
        dx=_scalar(m["dx"]),
        description=_text(m.get("description", "")),
        fk_mask=np.asarray(m["fk_mask"], dtype=float) if "fk_mask" in m else None,
        extras=extras,
    )


def save_mat(path: str | os.PathLike, data: np.ndarray, dt: float, dx: float,
             description: str = "", **arrays) -> None:
    """Write a dataset in the format read by both the Python and MATLAB code."""
    out = {"data": np.asarray(data, dtype=np.float32), "dt": float(dt), "dx": float(dx),
           "description": description}
    for k, v in arrays.items():
        v = np.asarray(v)
        out[k] = v.astype(np.uint8) if v.dtype == bool else v.astype(np.float32)
    sio.savemat(path, out, do_compression=True)


def list_datasets() -> list[str]:
    return sorted(p.stem for p in (repo_root() / "data").glob("*.mat"))


def load_config(name_or_path: str | os.PathLike) -> dict:
    """Load ``configs/<name>.json`` (or an explicit ``.json`` path)."""
    p = Path(name_or_path)
    if p.suffix != ".json":
        p = repo_root() / "configs" / f"{name_or_path}.json"
    with open(p, encoding="utf-8") as fh:
        cfg = json.load(fh)
    cfg.pop("_comment", None)
    return cfg
