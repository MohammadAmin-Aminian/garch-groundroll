"""k-means segmentation of the CSD image and selection of signal clusters."""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = ["kmeans", "cluster_statistics", "select_signal_clusters"]


def kmeans(
    X: np.ndarray,
    n_clusters: int,
    n_init: int = 10,
    max_iter: int = 100,
    random_state: int | None = 0,
) -> np.ndarray:
    """k-means++ clustering; returns labels ``1..n_clusters``.

    Labels are re-ordered so that cluster 1 has the most members; this makes
    the labelling independent of the random initialisation order.
    Uses scikit-learn when available, otherwise a NumPy implementation.
    """
    try:
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter,
                    random_state=random_state)
        labels = km.fit_predict(X)
    except ImportError:  # pragma: no cover - exercised only without sklearn
        labels = _kmeans_numpy(X, n_clusters, n_init, max_iter, random_state)
    counts = np.bincount(labels, minlength=n_clusters)
    order = np.argsort(-counts, kind="stable")
    remap = np.empty(n_clusters, dtype=int)
    remap[order] = np.arange(1, n_clusters + 1)
    return remap[labels]


def _kmeans_numpy(X, k, n_init, max_iter, random_state):
    rng = np.random.default_rng(random_state)
    best, best_inertia = None, np.inf
    for _ in range(n_init):
        C = X[rng.integers(len(X))][None, :]
        for _ in range(1, k):  # k-means++ seeding
            d2 = np.min(((X[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
            C = np.vstack([C, X[rng.choice(len(X), p=d2 / d2.sum())]])
        for _ in range(max_iter):
            d2 = ((X[:, None, :] - C[None]) ** 2).sum(-1)
            lab = d2.argmin(1)
            newC = np.array([X[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(k)])
            if np.allclose(newC, C):
                break
            C = newC
        inertia = d2[np.arange(len(X)), lab].sum()
        if inertia < best_inertia:
            best, best_inertia = lab, inertia
    return best


def cluster_statistics(labels: np.ndarray, csd: np.ndarray) -> dict:
    """Mean CSD and pixel fraction of each cluster."""
    ids = np.unique(labels)
    return {
        "ids": ids,
        "mean_csd": np.array([csd[labels == i].mean() for i in ids]),
        "fraction": np.array([(labels == i).mean() for i in ids]),
    }


def select_signal_clusters(
    labels: np.ndarray,
    csd: np.ndarray,
    method: str = "gap",
    n_keep: int | None = None,
    threshold: float | None = None,
    clusters: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose the clusters that contain leaked reflection energy.

    Ground roll has a large conditional standard deviation, reflections a
    small one, so clusters are ranked by their mean CSD and the low-CSD
    clusters are classified as signal.

    Parameters
    ----------
    method : {"gap", "n_keep", "threshold", "manual"}
        * ``"gap"`` - sort the cluster means and split them at the largest
          jump; every cluster below the jump is signal, the high-CSD
          clusters above it are ground roll (automatic, default). Restoring
          a quiet background cluster is harmless because the removed noise
          is close to zero there.
        * ``"n_keep"`` - keep the ``n_keep`` lowest-CSD clusters.
        * ``"threshold"`` - keep clusters whose mean CSD is below
          ``threshold * max(cluster mean CSD)``.
        * ``"manual"`` - keep the cluster ids given in ``clusters`` (after
          visual review, as recommended in the paper).

    Returns
    -------
    mask : (nt, nx) float array, 1 where the noise is reclassified as signal
    kept : array of kept cluster ids
    """
    stats = cluster_statistics(labels, csd)
    ids, means = stats["ids"], stats["mean_csd"]
    order = np.argsort(means)
    if method == "manual":
        if clusters is None:
            raise ValueError("method='manual' requires 'clusters'.")
        kept = np.asarray(sorted(clusters))
    elif method == "n_keep":
        if n_keep is None or not 0 < n_keep < len(ids):
            raise ValueError("n_keep must be between 1 and n_clusters - 1.")
        kept = np.sort(ids[order[:n_keep]])
    elif method == "threshold":
        if threshold is None:
            raise ValueError("method='threshold' requires 'threshold'.")
        kept = np.sort(ids[means < threshold * means.max()])
    elif method == "gap":
        m = means[order]
        cut = int(np.argmax(np.diff(m))) + 1 if len(m) > 1 else 0
        kept = np.sort(ids[order[:cut]])
    else:
        raise ValueError("Unknown method %r." % method)
    mask = np.isin(labels, kept).astype(float)
    return mask, kept
