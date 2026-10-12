"""Publication figures (matplotlib).

Conventions: seismic sections in grey (symmetric clip), magnitudes (CSD,
variance, f-k amplitude, local correlation) in a single-hue blue ramp,
categories (clusters, curves) in a fixed colour-blind-checked order.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402

from .filters import amplitude_spectrum, fk_spectrum  # noqa: E402

__all__ = [
    "SERIES", "BLUES", "plot_sections", "plot_csd", "plot_clusters", "plot_spectra",
    "plot_fk", "plot_trace_comparison", "plot_loglik_grid", "plot_local_correlation",
    "plot_workflow", "save",
]

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUES = LinearSegmentedColormap.from_list(
    "blues_seq", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.titleweight": "regular", "figure.dpi": 110,
    "savefig.dpi": 150, "savefig.bbox": "tight", "axes.spines.top": False,
    "axes.spines.right": False,
})


def _extent(shape, dt, dx):
    nt, nx = shape
    return [0, nx * dx, nt * dt * 1000.0, 0]  # x in m, t in ms


def _clip(arrs, pct=99.0):
    v = np.percentile(np.abs(np.concatenate([np.ravel(a) for a in arrs])), pct)
    return v if v > 0 else 1.0


def _section(ax, d, dt, dx, vmax, title=None, cmap="gray"):
    im = ax.imshow(d, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto",
                   extent=_extent(d.shape, dt, dx), interpolation="bilinear")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Time (ms)")
    if title:
        ax.set_title(title, loc="left")
    return im


def plot_sections(panels: Sequence[np.ndarray], titles: Sequence[str], dt: float, dx: float,
                  ncols: int | None = None, clip: float = 99.0, shared_clip: bool = True,
                  colorbar: bool = True, panel_size=(3.6, 3.4)):
    """Grid of seismic sections with a common (or individual) amplitude clip."""
    n = len(panels)
    ncols = ncols or n
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
                             squeeze=False, constrained_layout=True)
    vmax = _clip(panels, clip) if shared_clip else None
    for i, (ax, d, t) in enumerate(zip(axes.flat, panels, titles)):
        im = _section(ax, d, dt, dx, vmax if shared_clip else _clip([d], clip),
                      f"({chr(97 + i)}) {t}")
        if colorbar:
            fig.colorbar(im, ax=ax, shrink=0.85, label="Amplitude")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    return fig


def plot_csd(variance: np.ndarray, csd: np.ndarray, dt: float, dx: float):
    """Conditional variance and conditional standard deviation (paper Fig. 6)."""
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), constrained_layout=True)
    for ax, img, t, lab in ((axes[0], variance, "(a) Conditional variance", "Variance"),
                            (axes[1], csd, "(b) Conditional standard deviation", "CSD")):
        im = ax.imshow(img, cmap=BLUES, vmin=0, vmax=np.percentile(img, 99.5), aspect="auto",
                       extent=_extent(img.shape, dt, dx), interpolation="bilinear")
        ax.set_title(t, loc="left")
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Time (ms)")
        fig.colorbar(im, ax=ax, shrink=0.85, label=lab)
    return fig


def plot_clusters(noise, labels, mask, kept, stats, dt, dx):
    """Noise, cluster labels and the restored-signal mask (paper Fig. 7)."""
    k = int(labels.max())
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.6), constrained_layout=True)
    _section(axes[0], noise, dt, dx, _clip([noise]), "(a) Removed noise")
    cmap = ListedColormap(SERIES[:k])
    im = axes[1].imshow(labels, cmap=cmap, vmin=0.5, vmax=k + 0.5, aspect="auto",
                        extent=_extent(labels.shape, dt, dx), interpolation="nearest")
    axes[1].set_title("(b) k-means clusters of the CSD", loc="left")
    axes[1].set_xlabel("Distance (m)")
    axes[1].set_ylabel("Time (ms)")
    cb = fig.colorbar(im, ax=axes[1], shrink=0.85, ticks=range(1, k + 1))
    cb.ax.set_yticklabels([f"{i}  (CSD {m:.2f})" for i, m in zip(stats["ids"], stats["mean_csd"])])
    cb.set_label("Cluster (mean CSD)")
    im2 = axes[2].imshow(mask, cmap=ListedColormap(["#f0efec", SERIES[0]]), vmin=0, vmax=1,
                         aspect="auto", extent=_extent(mask.shape, dt, dx),
                         interpolation="nearest")
    axes[2].set_title(f"(c) Restored as signal: clusters {', '.join(map(str, kept))}", loc="left")
    axes[2].set_xlabel("Distance (m)")
    axes[2].set_ylabel("Time (ms)")
    cb2 = fig.colorbar(im2, ax=axes[2], shrink=0.85, ticks=[0.25, 0.75])
    cb2.ax.set_yticklabels(["noise", "signal"])
    return fig


def plot_spectra(curves: dict, dt: float, fmax: float = 100.0, reduce: str = "mean",
                 normalize: str = "first"):
    """Overlay of average amplitude spectra (paper Figs. 11, 14, 15).

    ``normalize="first"`` scales every curve by the maximum of the first one
    (the input), so restored amplitude is visible; ``"each"`` scales every
    curve to its own maximum (as in the paper); ``"none"`` keeps amplitudes.
    """
    fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
    ref = None
    for i, (name, d) in enumerate(curves.items()):
        f, A = amplitude_spectrum(d, dt, reduce)
        if normalize == "first":
            ref = A.max() if ref is None else ref
            A = A / ref
        elif normalize == "each":
            A = A / A.max()
        ax.plot(f, A, color=SERIES[i % len(SERIES)], lw=1.6, label=name)
    ax.set_xlim(0, fmax)
    ax.set_ylim(0, None)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel({"first": "Amplitude (relative to input)", "each": "Normalised amplitude"}
                  .get(normalize, "Amplitude"))
    ax.grid(True, color=GRID, lw=0.6)
    ax.legend(frameon=False)
    return fig


def plot_fk(panels: Sequence[np.ndarray], titles: Sequence[str], dt: float, dx: float,
            fmax: float = 60.0):
    """f-k amplitude spectra (paper Figs. 2, 9, 16)."""
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.7 * n, 3.4), squeeze=False, constrained_layout=True)
    specs = [fk_spectrum(d, dt, dx, pad=2) for d in panels]
    vmax = max(np.percentile(S[f <= fmax], 99.7) for S, k, f in specs)
    for i, (ax, (S, k, f), t) in enumerate(zip(axes[0], specs, titles)):
        sel = f <= fmax
        im = ax.imshow(S[sel], cmap=BLUES, vmin=0, vmax=vmax, aspect="auto", origin="lower",
                       extent=[k[0] * 1000, k[-1] * 1000, f[sel][0], f[sel][-1]])
        ax.set_title(f"({chr(97 + i)}) {t}", loc="left")
        ax.set_xlabel("Wavenumber (cycles/km)")
        ax.set_ylabel("Frequency (Hz)")
        fig.colorbar(im, ax=ax, shrink=0.85, label="Amplitude")
    return fig


def plot_trace_comparison(traces: dict, dt: float, trace_no: int, tmax: float | None = None):
    """One trace through several processing stages (paper Fig. 10)."""
    n = len(traces)
    fig, axes = plt.subplots(n, 1, figsize=(7.2, 1.35 * n + 0.4), sharex=True, sharey=True,
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    for i, (ax, (name, tr)) in enumerate(zip(axes, traces.items())):
        t = np.arange(tr.size) * dt * 1000
        ax.plot(t, tr, color=SERIES[0], lw=1.0)
        ax.axhline(0, color=GRID, lw=0.6, zorder=0)
        ax.set_title(f"({chr(97 + i)}) {name}", loc="left", fontsize=9)
        ax.set_ylabel("Amp.")
    axes[-1].set_xlabel(f"Time (ms) - trace {trace_no}")
    if tmax:
        axes[-1].set_xlim(0, tmax * 1000)
    return fig


def plot_loglik_grid(orders, loglik: np.ndarray, best: np.ndarray | None = None):
    """Per-trace log-likelihood of each GARCH(p, q) model (paper Fig. 5)."""
    n = len(orders)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(9.6, 2.3 * nrows), sharex=True,
                             sharey=True, squeeze=False, constrained_layout=True)
    x = np.arange(1, loglik.shape[1] + 1)
    for i, (ax, (p, q)) in enumerate(zip(axes.flat, orders)):
        ax.plot(x, loglik[i], color=SERIES[0], lw=0.9)
        title = f"GARCH({p},{q})"
        if best is not None:
            title += f"  - best on {np.mean(best == i):.0%} of traces"
        ax.set_title(title, loc="left", fontsize=9)
        ax.grid(True, color=GRID, lw=0.5)
    for ax in axes[-1]:
        ax.set_xlabel("Trace number")
    for ax in axes[:, 0]:
        ax.set_ylabel("Log-likelihood")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    return fig


def plot_local_correlation(panels: Sequence[np.ndarray], titles: Sequence[str], dt, dx):
    """Local correlation maps (paper Fig. 12); 1 = identical to the input data."""
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.7 * n, 3.4), squeeze=False, constrained_layout=True)
    for i, (ax, c, t) in enumerate(zip(axes[0], panels, titles)):
        im = ax.imshow(c, cmap=BLUES, vmin=0, vmax=1, aspect="auto",
                       extent=_extent(c.shape, dt, dx), interpolation="bilinear")
        ax.set_title(f"({chr(97 + i)}) {t}\nmean {np.mean(c):.3f}", loc="left")
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Time (ms)")
        fig.colorbar(im, ax=ax, shrink=0.85, label="Local correlation")
    return fig


def plot_workflow(result, dt, dx, title_prefix=""):
    """Six-panel overview of the whole method (paper Fig. 2)."""
    r = result
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 7.0), constrained_layout=True)
    v = _clip([r.data])
    _section(axes[0, 0], r.data, dt, dx, v, "(a) Input data")
    _section(axes[0, 1], r.filtered, dt, dx, v, "(b) Conventional filter")
    im = axes[0, 2].imshow(r.csd, cmap=BLUES, vmin=0, vmax=np.percentile(r.csd, 99.5),
                           aspect="auto", extent=_extent(r.csd.shape, dt, dx))
    axes[0, 2].set_title("(c) CSD of removed noise", loc="left")
    fig.colorbar(im, ax=axes[0, 2], shrink=0.85)
    k = int(r.labels.max())
    axes[1, 0].imshow(r.labels, cmap=ListedColormap(SERIES[:k]), vmin=0.5, vmax=k + 0.5,
                      aspect="auto", extent=_extent(r.labels.shape, dt, dx),
                      interpolation="nearest")
    axes[1, 0].set_title(f"(d) Clusters (kept: {', '.join(map(str, r.kept_clusters))})", loc="left")
    _section(axes[1, 1], r.corrected, dt, dx, v, "(e) After GARCH recovery")
    _section(axes[1, 2], r.recovered, dt, dx, v, "(f) Recovered signal")
    for ax in axes.flat:
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Time (ms)")
    if title_prefix:
        fig.suptitle(title_prefix, x=0.01, ha="left")
    return fig


def save(fig, path):
    fig.savefig(path, facecolor="white")
    plt.close(fig)
