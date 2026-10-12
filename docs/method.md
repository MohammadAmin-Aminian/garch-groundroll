# Method

This page summarises the algorithm of Aminian & Riahi (2023) as implemented in
this repository, and the choices made where the original scripts were
ambiguous.

## 1. Conventional ground-roll attenuation

A gather $d(t, x)$ is split by a conventional filter into a filtered part
$s$ and the removed part $n = d - s$ ("noise"). Two filters are provided:

* **Ormsby** (`ormsby_filter`, `ggr_ormsby`): a zero-phase trapezoidal filter
  defined by four corner frequencies $f_1 \le f_2 \le f_3 \le f_4$. For ground
  roll it is used as a band-stop that starts at 0 Hz, i.e. a low-cut, for
  example `[0, 0, 15, 25]` removes 0-15 Hz with a linear taper to 25 Hz.
* **f-k filter**, either a parametric fan filter (`fk_fan_filter`,
  `ggr_fk_fan`) that rejects apparent velocities $|f/k| < v_\text{reject}$
  with a raised-cosine taper up to $v_\text{pass}$ (optionally only below
  $f_\text{max}$), or an explicit reject mask in the centred f-k plane
  (`fk_mask_filter`, `ggr_fk_mask`). The masks used in the paper are
  stored with the data, so the published f-k results are reproduced exactly
  instead of being redrawn by hand.

Both filters also remove part of the reflection energy: the overlap of the
signal and ground-roll spectra, together with tapering, makes some useful
signal leak into $n$. The method recovers it.

## 2. GARCH model of the removed noise

Each trace $y_t$ of $n$ is modelled as a zero-mean GARCH($p$, $q$) process
(Bollerslev, 1986):

$$
y_t = \sigma_t z_t,\quad z_t \sim \mathcal N(0, 1),\qquad
\sigma_t^2 = \omega + \sum_{i=1}^{q} \alpha_i\, y_{t-i}^2 + \sum_{j=1}^{p} \beta_j\, \sigma_{t-j}^2
$$

with $\omega > 0$, $\alpha_i, \beta_j \ge 0$ and $\sum\alpha_i + \sum\beta_j < 1$.
$p$ counts the GARCH (lagged variance) terms and $q$ the ARCH terms, as in
MATLAB's `garch(P,Q)`.

The parameters maximise the Gaussian log-likelihood

$$
\ell = -\tfrac12 \sum_t \left[\ln 2\pi + \ln \sigma_t^2 + y_t^2/\sigma_t^2\right],
$$

with pre-sample $y^2$ and $\sigma^2$ equal to the sample mean of $y^2$. The
constraints are enforced by the re-parameterisation
$\omega = e^{\theta_0}$, $(\alpha, \beta) = c\,e^{\theta}/(1+\sum e^{\theta})$ with
$c = 0.9999$, and the optimisation is started from three points
($\sum\alpha, \sum\beta$) = (0.1, 0.8), (0.5, 0.45), (0.9, 0.05), keeping the best.
Each trace is scaled to unit RMS first; GARCH is scale-equivariant, so the
estimates are mapped back exactly.

The **conditional standard deviation** (CSD) $\sigma_t$ of every trace forms
an image of the same size as the gather. Ground roll is high-amplitude, so its
CSD is large; leaked reflections have a small CSD.

### Choosing $p$ and $q$

`select_garch_order` / `ggr_select_garch_order` fits the nine models
$p, q \in \{1, 2, 3\}$ trace by trace (paper Fig. 5). The paper compared raw
log-likelihoods; because the models are nested, the raw likelihood never
decreases with model size, so the default criterion here is BIC
($-2\ell + k \ln N$). Use `criterion="loglik"` to reproduce the paper's
comparison.

## 3. Two-dimensional clustering of the CSD

The CSD image is segmented with k-means on per-pixel texture features
(as in the original MATLAB `imsegkmeans` workflow):

1. the CSD clipped to [0, 1] (the original scripts converted the CSD with
   `im2uint16`, which saturates values above 1; `scaling="minmax"` is
   available as an alternative),
2. Gabor magnitude responses for a bank of wavelengths $\lambda$ and
   orientations $\theta$ (1-octave bandwidth, aspect ratio 0.5, the
   definition of MATLAB's `gabor`), each smoothed by a Gaussian with
   $\sigma = 1.5\lambda$,
3. optionally the row and column index (spatial coherence).

Features are standardised and clustered with k-means++ (10 restarts by
default; fewer restarts gave seed-dependent partitions). Clusters are numbered
by decreasing size.

## 4. Selecting the signal clusters

Clusters are ranked by their mean CSD. With `selection="gap"` (default) the
sorted means are split at the largest jump: the high-CSD clusters above the
jump are ground roll; all clusters below it are restored. Restoring a quiet
background cluster is harmless, because $n \approx 0$ there. Alternatives:
`n_keep` (keep the $k$ lowest-CSD clusters), `threshold` (keep clusters below a
fraction of the largest mean) and `manual` (explicit ids). As the paper notes,
the selection should be reviewed visually; `plot_clusters` shows each
cluster with its mean CSD.

## 5. Recovery

With the binary mask $M$ of the selected clusters,

$$
r = n \odot M,\qquad s_\text{GARCH} = s + r,\qquad n_\text{GARCH} = n - r .
$$

## 6. Quality measures

* **SNR** as in MATLAB's `snr(x, y)`: $10\log_{10}(\sum s^2/\sum n^2)$, before
  ($s, n$) and after ($s_\text{GARCH}, n_\text{GARCH}$) recovery. The gain in
  dB is reported. The paper quoted the *ratio* of the two dB values
  (e.g. 1.08, "8 %"); that ratio is also reported, but it is only meaningful
  when both values are positive (with negative values a ratio below 1 is an
  improvement).
* **True SNR** on synthetic data, against the known clean reflections.
* **Local correlation**: Pearson correlation in a 9 x 9 window between a
  filtered section and the input (1 = locally identical).
* **Amplitude spectra** and **f-k spectra**.
