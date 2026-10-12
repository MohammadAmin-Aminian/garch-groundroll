"""End-to-end workflow of Aminian & Riahi (2023).

1. Attenuate ground roll with a conventional filter (Ormsby or f-k)
   -> ``filtered``; the removed part is ``noise = data - filtered``.
2. Fit a GARCH(p, q) model to every trace of ``noise`` and compute the
   conditional variance and conditional standard deviation (CSD).
3. Segment the CSD image with Gabor texture features + k-means.
4. Clusters with low CSD contain reflection energy that leaked into the
   noise: ``mask = 1`` there.
5. Restore it: ``recovered = noise * mask``,
   ``corrected = filtered + recovered``,
   ``corrected_noise = noise - recovered``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .clustering import cluster_statistics, kmeans, select_signal_clusters
from .features import build_features
from .filters import fk_fan_filter, fk_mask_filter, ormsby_filter
from .garch import fit_garch_traces
from .metrics import snr_report

__all__ = ["DEFAULT_CONFIG", "PipelineResult", "apply_filter", "run_pipeline", "merge_config"]

DEFAULT_CONFIG: dict[str, Any] = {
    "filter": {"type": "fk_fan", "v_reject": 1000.0, "v_pass": None, "f_max": None,
               "f_taper": 5.0},
    "garch": {"p": 1, "q": 1},
    "features": {
        "wavelengths": [3, 6, 12, 24, 48, 96],
        "orientations": [0, 45, 90, 135],
        "smoothing": 1.5,
        "scaling": "clip",
        "coordinates": True,
    },
    "clustering": {
        "n_clusters": 5,
        "n_init": 10,
        "seed": 0,
        "selection": "gap",
        "n_keep": None,
        "threshold": None,
        "clusters": None,
    },
    "n_jobs": 1,
}


def merge_config(base: dict, override: dict | None) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_config(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


@dataclass
class PipelineResult:
    data: np.ndarray
    filtered: np.ndarray
    noise: np.ndarray
    variance: np.ndarray
    csd: np.ndarray
    loglik: np.ndarray
    labels: np.ndarray
    mask: np.ndarray
    kept_clusters: np.ndarray
    recovered: np.ndarray
    corrected: np.ndarray
    corrected_noise: np.ndarray
    snr: dict
    cluster_stats: dict
    config: dict = field(default_factory=dict)

    def summary(self) -> str:
        s = self.snr
        lines = [
            f"GARCH({self.config['garch']['p']},{self.config['garch']['q']}) on "
            f"{self.data.shape[1]} traces x {self.data.shape[0]} samples",
            "clusters (id: mean CSD, fraction): "
            + ", ".join(f"{i}: {m:.3g}, {f:.1%}" for i, m, f in zip(
                self.cluster_stats["ids"], self.cluster_stats["mean_csd"],
                self.cluster_stats["fraction"])),
            f"clusters restored as signal: {list(map(int, self.kept_clusters))}",
            f"SNR filtered  : {s['snr_before_db']:+.3f} dB",
            f"SNR corrected : {s['snr_after_db']:+.3f} dB",
            f"SNR gain      : {s['gain_db']:+.3f} dB   (paper-style ratio {s['ratio']:.3f})",
        ]
        return "\n".join(lines)


def apply_filter(data: np.ndarray, dt: float, dx: float, cfg: dict,
                 fk_mask: np.ndarray | None = None) -> np.ndarray:
    """Apply the conventional filter described by ``cfg`` and return the filtered data."""
    kind = cfg["type"]
    if kind == "ormsby":
        return ormsby_filter(data, dt, tuple(cfg["corners"]), cfg.get("mode", "bandstop"))
    if kind == "fk_fan":
        return fk_fan_filter(data, dt, dx, cfg["v_reject"], cfg.get("v_pass"),
                             cfg.get("f_max"), cfg.get("f_taper", 5.0))
    if kind == "fk_mask":
        if fk_mask is None:
            raise ValueError("filter type 'fk_mask' needs the dataset's f-k mask.")
        return fk_mask_filter(data, fk_mask)
    raise ValueError(f"Unknown filter type {kind!r}.")


def run_pipeline(
    data: np.ndarray,
    dt: float,
    dx: float,
    config: dict | None = None,
    filtered: np.ndarray | None = None,
    fk_mask: np.ndarray | None = None,
    verbose: bool = False,
) -> PipelineResult:
    """Run the complete GARCH signal-recovery workflow on one gather.

    Parameters
    ----------
    data : (nt, nx) array
        Input gather (AGC applied if desired).
    dt, dx : float
        Sampling in seconds and metres.
    config : dict, optional
        Overrides of :data:`DEFAULT_CONFIG`.
    filtered : (nt, nx) array, optional
        Output of any conventional filter. If given, the ``filter`` section
        of the configuration is ignored.
    fk_mask : (nt, nx) array, optional
        Noise mask for ``filter.type == "fk_mask"``.
    """
    cfg = merge_config(DEFAULT_CONFIG, config)
    data = np.asarray(data, dtype=float)
    log = print if verbose else (lambda *a, **k: None)

    if filtered is None:
        log("1/4 conventional filter:", cfg["filter"]["type"])
        filtered = apply_filter(data, dt, dx, cfg["filter"], fk_mask)
    filtered = np.asarray(filtered, dtype=float)
    noise = data - filtered

    p, q = int(cfg["garch"]["p"]), int(cfg["garch"]["q"])
    log(f"2/4 GARCH({p},{q}) on {noise.shape[1]} traces")
    g = fit_garch_traces(noise, p, q, n_jobs=cfg.get("n_jobs", 1))
    csd = g["std"]

    fc = cfg["features"]
    log("3/4 Gabor features + k-means")
    X = build_features(csd, fc["wavelengths"], fc["orientations"], fc["smoothing"],
                       fc["coordinates"], fc["scaling"])
    cc = cfg["clustering"]
    labels = kmeans(X, cc["n_clusters"], cc["n_init"], random_state=cc["seed"])
    labels = labels.reshape(csd.shape, order="F")

    mask, kept = select_signal_clusters(labels, csd, cc["selection"], cc.get("n_keep"),
                                        cc.get("threshold"), cc.get("clusters"))
    log("4/4 signal recovery, clusters kept:", list(kept))
    recovered = noise * mask
    corrected = filtered + recovered
    corrected_noise = noise - recovered

    return PipelineResult(
        data=data, filtered=filtered, noise=noise, variance=g["variance"], csd=csd,
        loglik=g["loglik"], labels=labels, mask=mask, kept_clusters=kept,
        recovered=recovered, corrected=corrected, corrected_noise=corrected_noise,
        snr=snr_report(filtered, noise, corrected, corrected_noise),
        cluster_stats=cluster_statistics(labels, csd), config=cfg,
    )
