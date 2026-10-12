"""Texture features of the conditional-standard-deviation image.

The clustering in the paper is a 2-D (image) segmentation of the conditional
standard deviation (CSD) of the removed noise. Each pixel is described by

* its CSD value,
* Gabor magnitude responses for a bank of wavelengths and orientations,
  each smoothed by a Gaussian with ``sigma = smoothing * wavelength``,
* its row and column coordinates,

and the features are standardised (zero mean, unit variance) before
k-means. The Gabor kernel follows the MATLAB ``gabor`` definition with
spatial-frequency bandwidth 1 octave and spatial aspect ratio 0.5, so the
Python and MATLAB versions of this package produce the same features.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

__all__ = ["gabor_kernel", "gabor_magnitude", "build_features", "scale_csd"]


def gabor_kernel(
    wavelength: float,
    orientation_deg: float,
    bandwidth: float = 1.0,
    aspect_ratio: float = 0.5,
) -> np.ndarray:
    """Complex Gabor kernel.

    ``wavelength`` is in pixels/cycle; ``orientation_deg`` is the direction
    of the wave vector, counter-clockwise from the column (x) axis with the
    row axis pointing down, as in MATLAB's ``gabor``.
    """
    sigma = wavelength / np.pi * np.sqrt(np.log(2) / 2) * (2**bandwidth + 1) / (2**bandwidth - 1)
    sx, sy = sigma, sigma / aspect_ratio
    r = int(np.ceil(3 * max(sx, sy)))
    y, x = np.mgrid[-r : r + 1, -r : r + 1].astype(float)
    th = np.deg2rad(orientation_deg)
    xr = x * np.cos(th) - y * np.sin(th)   # y axis points down => minus sign
    yr = x * np.sin(th) + y * np.cos(th)
    env = np.exp(-0.5 * (xr**2 / sx**2 + yr**2 / sy**2))
    return env * np.exp(1j * 2 * np.pi * xr / wavelength)


def gabor_magnitude(image: np.ndarray, wavelength: float, orientation_deg: float) -> np.ndarray:
    """Magnitude of the Gabor response (symmetric padding at the edges)."""
    g = gabor_kernel(wavelength, orientation_deg)
    r = g.shape[0] // 2
    padded = np.pad(image, r, mode="symmetric")
    out = fftconvolve(padded, g, mode="same")[r:-r, r:-r]
    return np.abs(out)


def scale_csd(csd: np.ndarray, mode: str = "clip") -> np.ndarray:
    """Map the CSD to [0, 1] before feature extraction.

    ``"clip"`` reproduces MATLAB's ``im2uint16`` on a double image used in
    the original study (values above 1 saturate); ``"minmax"`` rescales
    linearly; ``"none"`` leaves the values unchanged.
    """
    csd = np.asarray(csd, dtype=float)
    if mode == "clip":
        return np.clip(csd, 0.0, 1.0)
    if mode == "minmax":
        lo, hi = np.nanmin(csd), np.nanmax(csd)
        return (csd - lo) / (hi - lo) if hi > lo else np.zeros_like(csd)
    if mode == "none":
        return csd
    raise ValueError("mode must be 'clip', 'minmax' or 'none'.")


def build_features(
    csd: np.ndarray,
    wavelengths: Sequence[float] = (3, 6, 12, 24, 48, 96),
    orientations: Sequence[float] = (0, 45, 90, 135),
    smoothing: float = 1.5,
    include_coordinates: bool = True,
    scaling: str = "clip",
) -> np.ndarray:
    """Standardised feature matrix of shape ``(nt * nx, n_features)``.

    Pixels are ordered column-major (``order="F"``) so that labels can be
    reshaped identically in Python and MATLAB.

    Parameters
    ----------
    csd : (nt, nx) array
        Conditional standard deviation image.
    wavelengths, orientations
        Gabor filter bank (pixels/cycle, degrees). Use empty sequences to
        cluster on the CSD (and coordinates) alone.
    smoothing : float
        Gaussian smoothing of each Gabor magnitude with
        ``sigma = smoothing * wavelength`` (the original study used
        ``3 * 0.5 * wavelength``, i.e. ``smoothing=1.5``). ``0`` disables it.
    include_coordinates : bool
        Append the row and column index as features (spatial coherence).
    scaling : {"clip", "minmax", "none"}
        See :func:`scale_csd`.
    """
    img = scale_csd(csd, scaling)
    nt, nx = img.shape
    feats = [img]
    for lam in wavelengths:
        for th in orientations:
            mag = gabor_magnitude(img, lam, th)
            if smoothing > 0:
                mag = gaussian_filter(mag, sigma=smoothing * lam, mode="nearest", truncate=2.0)
            feats.append(mag)
    if include_coordinates:
        rows, cols = np.mgrid[0:nt, 0:nx]
        feats += [cols.astype(float), rows.astype(float)]
    X = np.stack([f.ravel(order="F") for f in feats], axis=1)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd
