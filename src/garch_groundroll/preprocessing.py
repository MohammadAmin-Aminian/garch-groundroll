"""Pre-processing utilities."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d

__all__ = ["agc"]


def agc(d: np.ndarray, window: int = 100, eps: float = 1e-10) -> np.ndarray:
    """Automatic gain control with a sliding RMS window.

    Each sample is divided by the RMS amplitude of a centred window of
    ``window`` samples along the time axis.

    Parameters
    ----------
    d : (nt,) or (nt, nx) array
    window : int
        Window length in samples (the original study used 100 samples).
    eps : float
        Stabilisation relative to the global RMS (avoids division by zero in
        muted zones).
    """
    d = np.asarray(d, dtype=float)
    rms = np.sqrt(uniform_filter1d(d * d, size=int(window), axis=0, mode="nearest"))
    floor = eps * (np.sqrt(np.mean(d * d)) + np.finfo(float).tiny)
    return d / np.maximum(rms, floor)
