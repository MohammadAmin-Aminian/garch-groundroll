"""Synthetic shot gather with hyperbolic reflections and dispersive ground roll.

The default parameters reproduce the synthetic model of the original study
(five reflections, 20 Hz Ricker wavelet, 2 ms sampling, 3 s record, a
split-spread with 20 m receiver spacing and 100 channels). Because the
clean reflections and the ground roll are generated separately, the true
signal is known and the recovery can be scored objectively.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ricker", "SyntheticGather", "make_synthetic_shot"]


def ricker(f0: float, dt: float, length: float | None = None) -> np.ndarray:
    """Zero-phase Ricker wavelet with peak frequency ``f0`` (Hz).

    The wavelet has ``2 * round(length / (2 dt)) + 1`` samples (odd, centred).
    """
    length = 2.5 / f0 if length is None else length
    half = int(round(length / (2.0 * dt)))
    t = np.arange(-half, half + 1) * dt
    a = (np.pi * f0 * t) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


@dataclass
class SyntheticGather:
    data: np.ndarray        # signal + ground roll (+ random noise)
    signal: np.ndarray      # clean reflections
    ground_roll: np.ndarray
    offsets: np.ndarray     # m
    dt: float
    dx: float


def _place_wavelet(trace: np.ndarray, t: float, dt: float, w: np.ndarray, amp: float) -> None:
    """Add ``amp * w`` centred at time ``t`` with sub-sample (sinc-free, linear) accuracy."""
    nt = trace.size
    half = w.size // 2
    pos = t / dt
    i0 = int(np.floor(pos))
    frac = pos - i0
    for shift, weight in ((0, 1.0 - frac), (1, frac)):
        c = i0 + shift
        lo, hi = c - half, c - half + w.size
        a, b = max(lo, 0), min(hi, nt)
        if a < b:
            trace[a:b] += amp * weight * w[a - lo : b - lo]


def make_synthetic_shot(
    nt: int = 1501,
    dt: float = 0.002,
    n_side: int = 50,
    dx: float = 20.0,
    t0: tuple[float, ...] = (0.5, 0.9, 1.5, 1.7, 2.0),
    velocities: tuple[float, ...] = (1800, 1800, 1700, 1700, 1600),
    amplitudes: tuple[float, ...] = (3, 3, 3, 3, 3),
    f0: float = 20.0,
    gr_t0: float = 0.4,
    gr_duration: float = 0.5,
    gr_velocity: tuple[float, float] = (1000.0, 600.0),
    gr_freq: tuple[float, float] = (6.0, 14.0),
    gr_amplitude: float = 3.0,
    noise_std: float = 0.0,
    seed: int | None = 0,
) -> SyntheticGather:
    """Generate a split-spread synthetic shot gather.

    Ground roll is modelled as a dispersive linear wave train: at offset
    ``x`` it starts at ``gr_t0 + |x|/v_fast`` and ends at
    ``gr_t0 + gr_duration + |x|/v_slow``; within that window the
    instantaneous frequency sweeps from ``gr_freq[0]`` (early, fast) to
    ``gr_freq[1]`` (late, slow) under a Hann envelope.
    """
    offsets = np.concatenate([-np.arange(n_side, 0, -1), np.arange(1, n_side + 1)]) * dx
    nx = offsets.size
    t = np.arange(nt) * dt
    w = ricker(f0, dt)

    signal = np.zeros((nt, nx))
    for tau, v, a in zip(t0, velocities, amplitudes):
        tt = np.sqrt(tau**2 + (offsets / v) ** 2)
        for j in range(nx):
            _place_wavelet(signal[:, j], tt[j], dt, w, a)

    v_fast, v_slow = gr_velocity
    fa, fb = gr_freq
    gr = np.zeros((nt, nx))
    for j, x in enumerate(np.abs(offsets)):
        ts = gr_t0 + x / v_fast
        te = gr_t0 + gr_duration + x / v_slow
        m = (t >= ts) & (t <= te)
        if not np.any(m):
            continue
        tau = (t[m] - ts) / (te - ts)                 # 0 -> 1 across the window
        phase = 2 * np.pi * (te - ts) * (fa * tau + 0.5 * (fb - fa) * tau**2)
        gr[m, j] = np.sin(phase) * np.sin(np.pi * tau) ** 2
    gr *= gr_amplitude

    data = signal + gr
    if noise_std > 0:
        rng = np.random.default_rng(seed)
        data = data + noise_std * rng.standard_normal(data.shape)
    return SyntheticGather(data=data, signal=signal, ground_roll=gr, offsets=offsets, dt=dt, dx=dx)
