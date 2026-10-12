"""Quality measures used in the paper."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

__all__ = ["snr_db", "snr_report", "local_correlation", "true_snr_db"]


def snr_db(signal: np.ndarray, noise: np.ndarray) -> float:
    """Signal-to-noise ratio in dB, ``10 log10(sum s^2 / sum n^2)``.

    Identical to MATLAB's ``snr(x, y)`` for a signal ``x`` and noise ``y``.
    """
    ps = float(np.sum(np.asarray(signal, float) ** 2))
    pn = float(np.sum(np.asarray(noise, float) ** 2))
    return 10.0 * np.log10(ps / pn)


def snr_report(filtered, noise, corrected, corrected_noise) -> dict:
    """SNR before and after signal recovery.

    ``snr_before = snr(filtered, noise)`` and
    ``snr_after = snr(corrected, corrected_noise)`` as in the paper.

    ``gain_db`` (after - before) is the physically meaningful change.
    ``ratio`` (after / before) is the figure quoted in the paper as a
    percentage (e.g. 1.08 -> "8 %"); a ratio of decibel values is only
    meaningful when both are positive, so prefer ``gain_db``.
    """
    s1 = snr_db(filtered, noise)
    s2 = snr_db(corrected, corrected_noise)
    return {"snr_before_db": s1, "snr_after_db": s2, "gain_db": s2 - s1,
            "ratio": s2 / s1 if s1 != 0 else np.nan}


def true_snr_db(estimate: np.ndarray, truth: np.ndarray) -> float:
    """Reconstruction SNR against a known clean signal (synthetic data)."""
    return snr_db(truth, np.asarray(estimate) - np.asarray(truth))


def local_correlation(a: np.ndarray, b: np.ndarray, window: int = 9) -> np.ndarray:
    """Local Pearson correlation of two images in a ``window x window`` box.

    Values near 1 mean the two sections are locally identical, values near 0
    that they are unrelated. The paper uses it to show how much of the input
    data each filter preserves outside the ground-roll cone.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = lambda x: uniform_filter(x, size=window, mode="nearest")  # noqa: E731
    ma, mb = m(a), m(b)
    cov = m(a * b) - ma * mb
    va = np.maximum(m(a * a) - ma**2, 0)
    vb = np.maximum(m(b * b) - mb**2, 0)
    den = np.sqrt(va * vb)
    out = np.zeros_like(cov)
    ok = den > 1e-12 * (den.max() + 1e-30)
    out[ok] = cov[ok] / den[ok]
    return np.clip(out, -1.0, 1.0)
