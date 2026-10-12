"""Conventional ground-roll attenuation filters and spectral tools.

All functions take a 2-D gather ``d`` of shape ``(nt, nx)`` (time samples x
traces) and return arrays of the same shape. Every filter returns the
*filtered data*; the removed noise is ``d - filtered``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "ormsby_weights",
    "ormsby_filter",
    "fk_fan_filter",
    "fk_mask_filter",
    "fk_spectrum",
    "amplitude_spectrum",
]


# --------------------------------------------------------------------------
# Ormsby (trapezoidal) frequency filter
# --------------------------------------------------------------------------
def ormsby_weights(freqs: np.ndarray, corners: tuple[float, float, float, float]) -> np.ndarray:
    """Trapezoidal Ormsby band-pass weights for the absolute frequencies ``freqs``.

    ``corners = (f1, f2, f3, f4)`` with ``f1 <= f2 <= f3 <= f4`` (Hz): zero
    below ``f1``, linear ramp ``f1 -> f2``, flat ``f2 -> f3``, linear ramp
    ``f3 -> f4`` and zero above ``f4``.
    """
    f1, f2, f3, f4 = map(float, corners)
    if not (0 <= f1 <= f2 <= f3 <= f4):
        raise ValueError("Ormsby corners must satisfy 0 <= f1 <= f2 <= f3 <= f4.")
    f = np.abs(np.asarray(freqs, dtype=float))
    w = np.zeros_like(f)
    w[(f >= f2) & (f <= f3)] = 1.0
    if f2 > f1:
        m = (f > f1) & (f < f2)
        w[m] = (f[m] - f1) / (f2 - f1)
    if f4 > f3:
        m = (f > f3) & (f < f4)
        w[m] = (f4 - f[m]) / (f4 - f3)
    return w


def ormsby_filter(
    d: np.ndarray,
    dt: float,
    corners: tuple[float, float, float, float],
    mode: str = "bandpass",
) -> np.ndarray:
    """Zero-phase Ormsby filter applied trace by trace.

    Parameters
    ----------
    d : (nt, nx) array
    dt : float
        Sampling interval in seconds.
    corners : (f1, f2, f3, f4)
        Trapezoid corner frequencies in Hz.
    mode : {"bandpass", "bandstop"}
        ``"bandpass"`` keeps the trapezoid; ``"bandstop"`` removes it
        (weights ``1 - W``). For ground-roll attenuation use a band-stop
        around the ground-roll band, e.g. ``corners=(0, 0, 12, 18)`` removes
        0-12 Hz with a 12-18 Hz taper (a low-cut).
    """
    d = np.asarray(d, dtype=float)
    nt = d.shape[0]
    nfft = int(2 ** np.ceil(np.log2(2 * nt)))  # zero padding avoids wrap-around
    freqs = np.fft.rfftfreq(nfft, dt)
    w = ormsby_weights(freqs, corners)
    if mode == "bandstop":
        w = 1.0 - w
    elif mode != "bandpass":
        raise ValueError("mode must be 'bandpass' or 'bandstop'.")
    D = np.fft.rfft(d, n=nfft, axis=0)
    return np.fft.irfft(D * w[:, None], n=nfft, axis=0)[:nt]


# --------------------------------------------------------------------------
# f-k filters
# --------------------------------------------------------------------------
def _fk_axes(nt: int, nx: int, dt: float, dx: float):
    f = np.fft.fftfreq(nt, dt)          # Hz
    k = np.fft.fftfreq(nx, dx)          # cycles / m
    return np.meshgrid(f, k, indexing="ij")


def _cos_taper(x: np.ndarray, x0: float, x1: float) -> np.ndarray:
    """0 for x <= x0, 1 for x >= x1, raised-cosine in between."""
    if x1 <= x0:
        return (x >= x0).astype(float)
    t = np.clip((x - x0) / (x1 - x0), 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * t)


def fk_fan_filter(
    d: np.ndarray,
    dt: float,
    dx: float,
    v_reject: float,
    v_pass: float | None = None,
    f_max: float | None = None,
    f_taper: float = 5.0,
    pad: int = 2,
) -> np.ndarray:
    """Velocity (fan) filter in the f-k domain.

    Energy with apparent velocity ``|f / k| < v_reject`` is removed, energy
    with ``|f / k| > v_pass`` is kept and a raised-cosine taper is applied in
    between. Ground roll is slow *and* low-frequency, so the rejection can be
    restricted to ``|f| < f_max`` (tapered over ``f_taper`` Hz).

    Parameters
    ----------
    d : (nt, nx) array
    dt, dx : float
        Time (s) and trace (m) sampling.
    v_reject, v_pass : float
        Velocities in m/s; ``v_pass`` defaults to ``1.3 * v_reject``.
    f_max : float, optional
        Only reject below this frequency (Hz). ``None`` = all frequencies.
    pad : int
        Zero-padding factor in both dimensions (reduces wrap-around).
    """
    d = np.asarray(d, dtype=float)
    nt, nx = d.shape
    v_pass = 1.3 * v_reject if v_pass is None else v_pass
    ntp = int(2 ** np.ceil(np.log2(pad * nt)))
    nxp = int(2 ** np.ceil(np.log2(pad * nx)))
    F, K = _fk_axes(ntp, nxp, dt, dx)
    with np.errstate(divide="ignore", invalid="ignore"):
        vapp = np.where(K == 0, np.inf, np.abs(F) / np.abs(K))
    keep = _cos_taper(vapp, v_reject, v_pass)
    if f_max is not None:
        # outside the low-frequency band everything is kept
        keep = 1.0 - (1.0 - keep) * (1.0 - _cos_taper(np.abs(F), f_max, f_max + f_taper))
    D = np.fft.fft2(d, s=(ntp, nxp))
    return np.real(np.fft.ifft2(D * keep))[:nt, :nx]


def fk_mask_filter(d: np.ndarray, noise_mask: np.ndarray) -> np.ndarray:
    """f-k filter with an explicit noise mask (as drawn in the original study).

    ``noise_mask`` has the shape of ``d`` and is defined on the *centred*
    spectrum ``fftshift(fft2(d))``; it is 1 on the region to reject. This
    reproduces the original MATLAB workflow
    ``noise = real(ifft2(ifftshift(fftshift(fft2(d)) .* mask)))``.
    """
    d = np.asarray(d, dtype=float)
    noise_mask = np.asarray(noise_mask, dtype=float)
    if noise_mask.shape != d.shape:
        raise ValueError("noise_mask must have the same shape as d.")
    S = np.fft.fftshift(np.fft.fft2(d))
    noise = np.real(np.fft.ifft2(np.fft.ifftshift(S * noise_mask)))
    return d - noise


def fk_spectrum(d: np.ndarray, dt: float, dx: float, pad: int = 1):
    """Amplitude f-k spectrum for display.

    Returns ``(S, k, f)`` where ``S`` has shape ``(nf, nk)`` for the
    non-negative frequencies ``f`` (Hz) and centred wavenumbers ``k``
    (cycles/m).
    """
    d = np.asarray(d, dtype=float)
    nt, nx = d.shape
    ntp = int(2 ** np.ceil(np.log2(pad * nt)))
    nxp = int(2 ** np.ceil(np.log2(pad * nx)))
    S = np.abs(np.fft.fftshift(np.fft.fft2(d, s=(ntp, nxp)), axes=1))
    f = np.fft.fftfreq(ntp, dt)
    k = np.fft.fftshift(np.fft.fftfreq(nxp, dx))
    pos = f >= 0
    order = np.argsort(f[pos])
    return S[pos][order], k, f[pos][order]


def amplitude_spectrum(d: np.ndarray, dt: float, reduce: str = "mean"):
    """Amplitude spectrum of a trace or gather.

    For a gather the per-trace spectra are combined with ``reduce`` (``"mean"``
    or ``"max"``; the original scripts used the maximum over traces).
    Returns ``(f, A)``.
    """
    d = np.asarray(d, dtype=float)
    if d.ndim == 1:
        d = d[:, None]
    A = np.abs(np.fft.rfft(d, axis=0))
    A = A.max(axis=1) if reduce == "max" else A.mean(axis=1)
    return np.fft.rfftfreq(d.shape[0], dt), A
