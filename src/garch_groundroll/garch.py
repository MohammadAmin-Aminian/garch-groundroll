"""GARCH(p, q) estimation by Gaussian maximum likelihood.

The model used throughout the paper is a zero-mean GARCH(p, q) process

    y_t      = sigma_t * z_t,             z_t ~ N(0, 1)
    sigma2_t = omega + sum_{i=1..q} alpha_i * y_{t-i}^2
                     + sum_{j=1..p} beta_j  * sigma2_{t-j}

with omega > 0, alpha_i >= 0, beta_j >= 0 and sum(alpha) + sum(beta) < 1
(covariance stationarity). ``p`` is the number of GARCH (lagged variance)
terms and ``q`` the number of ARCH (lagged squared innovation) terms, which
matches the convention of MATLAB's ``garch(P, Q)``.

Pre-sample values of ``y^2`` and ``sigma^2`` are set to the sample mean of
``y^2`` (back-casting by the unconditional sample variance).

The implementation is self-contained (NumPy/SciPy only). Each trace is
scaled to unit RMS before fitting; because GARCH is scale-equivariant the
estimates are mapped back exactly (omega and the variances are multiplied by
the squared scale; the log-likelihood is shifted by ``-n*log(scale)``).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter, lfiltic

__all__ = [
    "GarchFit",
    "garch_conditional_variance",
    "garch_loglik",
    "fit_garch",
    "fit_garch_traces",
    "select_garch_order",
]

_LOG_2PI = np.log(2.0 * np.pi)
# Upper bound for the persistence sum(alpha)+sum(beta) (strictly < 1).
_MAX_PERSISTENCE = 0.9999


@dataclass
class GarchFit:
    """Result of fitting a GARCH(p, q) model to one series."""

    p: int
    q: int
    omega: float
    alpha: np.ndarray
    beta: np.ndarray
    loglik: float
    variance: np.ndarray = field(repr=False)
    nobs: int = 0
    converged: bool = True

    @property
    def n_params(self) -> int:
        return 1 + self.p + self.q

    @property
    def aic(self) -> float:
        return -2.0 * self.loglik + 2.0 * self.n_params

    @property
    def bic(self) -> float:
        return -2.0 * self.loglik + np.log(max(self.nobs, 1)) * self.n_params

    @property
    def std(self) -> np.ndarray:
        """Conditional standard deviation sigma_t."""
        return np.sqrt(self.variance)

    @property
    def persistence(self) -> float:
        return float(np.sum(self.alpha) + np.sum(self.beta))


# --------------------------------------------------------------------------
# Core recursion
# --------------------------------------------------------------------------
def garch_conditional_variance(
    y: np.ndarray,
    omega: float,
    alpha: Sequence[float],
    beta: Sequence[float],
    backcast: float | None = None,
) -> np.ndarray:
    """Conditional variance sigma_t^2 of a zero-mean GARCH(p, q) series.

    Parameters
    ----------
    y : 1-D array
        Innovations (here: one trace of the removed noise).
    omega, alpha, beta
        Model parameters; ``len(alpha) == q`` and ``len(beta) == p``.
    backcast : float, optional
        Pre-sample value for both ``y^2`` and ``sigma^2``. Defaults to
        ``mean(y**2)``.
    """
    y = np.asarray(y, dtype=float)
    alpha = np.atleast_1d(np.asarray(alpha, dtype=float))
    beta = np.atleast_1d(np.asarray(beta, dtype=float))
    q, p = alpha.size, beta.size
    y2 = y * y
    if backcast is None:
        backcast = float(np.mean(y2)) if y2.size else 1.0

    # x_t = omega + sum_i alpha_i * y^2_{t-i}  (pre-sample y^2 = backcast)
    y2_ext = np.concatenate([np.full(q, backcast), y2])
    x = np.full(y.size, omega, dtype=float)
    for i in range(1, q + 1):
        x += alpha[i - 1] * y2_ext[q - i : q - i + y.size]

    if p == 0:
        return x
    # sigma2_t = x_t + sum_j beta_j * sigma2_{t-j}  (pre-sample sigma^2 = backcast)
    a = np.concatenate([[1.0], -beta])
    zi = lfiltic([1.0], a, y=np.full(p, backcast))
    sigma2, _ = lfilter([1.0], a, x, zi=zi)
    return sigma2


def garch_loglik(y: np.ndarray, sigma2: np.ndarray) -> float:
    """Gaussian log-likelihood of ``y`` given conditional variances."""
    sigma2 = np.maximum(sigma2, np.finfo(float).tiny)
    return float(-0.5 * np.sum(_LOG_2PI + np.log(sigma2) + y * y / sigma2))


# --------------------------------------------------------------------------
# Unconstrained re-parameterisation (shared with the MATLAB implementation)
#   omega          = exp(theta_0)
#   [alpha, beta]  = c * exp(theta_k) / (1 + sum_k exp(theta_k)),  c < 1
# guarantees omega > 0, alpha, beta >= 0 and sum(alpha)+sum(beta) < 1.
# --------------------------------------------------------------------------
def _unpack(theta: np.ndarray, p: int, q: int):
    omega = np.exp(np.clip(theta[0], -50.0, 50.0))
    e = np.exp(np.clip(theta[1:], -50.0, 50.0))
    w = _MAX_PERSISTENCE * e / (1.0 + e.sum())
    return omega, w[:q], w[q:]


def _pack(omega: float, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    w = np.concatenate([alpha, beta]) / _MAX_PERSISTENCE
    w = np.clip(w, 1e-8, None)
    rest = max(1.0 - w.sum(), 1e-8)
    return np.concatenate([[np.log(omega)], np.log(w / rest)])


# (sum alpha, sum beta) of the starting points; the best optimum is kept.
_STARTS = ((0.10, 0.80), (0.50, 0.45), (0.90, 0.05))


def _start_values(p: int, q: int, var: float, a_sum: float = 0.10, b_sum: float = 0.80) -> np.ndarray:
    alpha = np.full(q, a_sum / q)
    beta = np.full(p, b_sum / p) if p else np.zeros(0)
    omega = var * (1.0 - alpha.sum() - beta.sum())
    return _pack(omega, alpha, beta)


def fit_garch(y: np.ndarray, p: int = 1, q: int = 1, maxiter: int = 500,
              n_starts: int = 3) -> GarchFit:
    """Fit a zero-mean GARCH(p, q) model to a 1-D series by maximum likelihood.

    The likelihood is maximised with L-BFGS-B from ``n_starts`` starting
    points (1-3) and the best optimum is returned.

    A series that is identically zero (e.g. a muted trace) returns a fit with
    zero variance and ``converged=False``.
    """
    if q < 1 or p < 0:
        raise ValueError("Require q >= 1 (ARCH terms) and p >= 0 (GARCH terms).")
    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    scale = float(np.sqrt(np.mean(y * y))) if n else 0.0
    if n < 3 * (p + q + 1) or scale == 0.0 or not np.isfinite(scale):
        return GarchFit(p, q, 0.0, np.zeros(q), np.zeros(p), 0.0,
                        np.zeros(n), nobs=n, converged=False)

    z = y / scale  # unit RMS => backcast = 1

    def nll(theta):
        omega, alpha, beta = _unpack(theta, p, q)
        s2 = garch_conditional_variance(z, omega, alpha, beta, backcast=1.0)
        val = -garch_loglik(z, s2)
        return val if np.isfinite(val) else 1e300

    res = None
    for a_sum, b_sum in _STARTS[: max(1, min(n_starts, len(_STARTS)))]:
        if p == 0:
            a_sum, b_sum = a_sum + b_sum, 0.0
        theta0 = _start_values(p, q, 1.0, a_sum, b_sum)
        r = minimize(nll, theta0, method="L-BFGS-B", options={"maxiter": maxiter})
        if not np.isfinite(r.fun):
            continue
        if res is None or r.fun < res.fun:
            res = r
    if res is None or not res.success:
        theta0 = res.x if res is not None else _start_values(p, q, 1.0)
        r = minimize(nll, theta0, method="Nelder-Mead",
                     options={"maxiter": 200 * (p + q + 1), "xatol": 1e-7, "fatol": 1e-9})
        if res is None or r.fun <= res.fun:
            res = r

    omega, alpha, beta = _unpack(res.x, p, q)
    s2 = garch_conditional_variance(z, omega, alpha, beta, backcast=1.0)
    ll_z = garch_loglik(z, s2)
    return GarchFit(
        p=p,
        q=q,
        omega=float(omega * scale**2),
        alpha=np.asarray(alpha),
        beta=np.asarray(beta),
        loglik=float(ll_z - n * np.log(scale)),
        variance=s2 * scale**2,
        nobs=n,
        converged=bool(res.success),
    )


def _fit_column(args):
    col, p, q = args
    f = fit_garch(col, p, q)
    return f.variance, f.loglik, f.omega, f.alpha, f.beta, f.converged


def fit_garch_traces(
    noise: np.ndarray,
    p: int = 1,
    q: int = 1,
    n_jobs: int = 1,
) -> dict:
    """Fit an independent GARCH(p, q) model to every trace (column).

    Returns a dict with ``variance`` (nt x nx), ``std`` (nt x nx),
    ``loglik`` (nx,), ``omega`` (nx,), ``alpha`` (nx x q), ``beta`` (nx x p)
    and ``converged`` (nx,).
    """
    noise = np.asarray(noise, dtype=float)
    if noise.ndim != 2:
        raise ValueError("noise must be a 2-D array (time x traces).")
    nt, nx = noise.shape
    jobs = [(noise[:, j], p, q) for j in range(nx)]
    if n_jobs is None or n_jobs == 1:
        results = [_fit_column(a) for a in jobs]
    else:
        workers = None if n_jobs < 0 else n_jobs
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_fit_column, jobs, chunksize=8))

    var = np.column_stack([r[0] for r in results])
    return {
        "variance": var,
        "std": np.sqrt(var),
        "loglik": np.array([r[1] for r in results]),
        "omega": np.array([r[2] for r in results]),
        "alpha": np.vstack([r[3] for r in results]),
        "beta": np.vstack([r[4] for r in results]) if p else np.zeros((nx, 0)),
        "converged": np.array([r[5] for r in results]),
        "p": p,
        "q": q,
    }


def select_garch_order(
    noise: np.ndarray,
    orders: Iterable[tuple[int, int]] = tuple((p, q) for p in (1, 2, 3) for q in (1, 2, 3)),
    criterion: str = "bic",
    n_jobs: int = 1,
) -> dict:
    """Compare GARCH(p, q) models trace by trace (paper, Figure 5).

    Parameters
    ----------
    orders : iterable of (p, q)
    criterion : {"bic", "aic", "loglik"}
        ``"loglik"`` reproduces the paper's comparison of raw log-likelihoods.
        Because the models are nested, the raw log-likelihood always favours
        the largest model; BIC (default) penalises the extra parameters.

    Returns
    -------
    dict with ``orders`` (list), ``loglik`` (n_orders x nx), ``score``
    (n_orders x nx, lower is better), ``best`` (nx,) index into ``orders``
    and ``best_overall`` (the order that wins on most traces).
    """
    orders = [tuple(o) for o in orders]
    noise = np.asarray(noise, dtype=float)
    nt, nx = noise.shape
    ll = np.zeros((len(orders), nx))
    score = np.zeros_like(ll)
    for k, (p, q) in enumerate(orders):
        res = fit_garch_traces(noise, p, q, n_jobs=n_jobs)
        ll[k] = res["loglik"]
        npar = 1 + p + q
        if criterion == "bic":
            score[k] = -2 * ll[k] + np.log(nt) * npar
        elif criterion == "aic":
            score[k] = -2 * ll[k] + 2 * npar
        elif criterion == "loglik":
            score[k] = -ll[k]
        else:
            raise ValueError("criterion must be 'bic', 'aic' or 'loglik'.")
    best = np.argmin(score, axis=0)
    counts = np.bincount(best, minlength=len(orders))
    return {
        "orders": orders,
        "loglik": ll,
        "score": score,
        "criterion": criterion,
        "best": best,
        "best_overall": orders[int(np.argmax(counts))],
        "counts": counts,
    }
