"""Recovery of reflection signals lost during ground-roll attenuation by
clustering the GARCH conditional standard deviation of the removed noise.

Reference
---------
Aminian, M. A. & Riahi, M. A. (2023). Enhanced data fidelity after ground
roll attenuation using conditional standard deviation clustering obtained
from the GARCH model. *Exploration Geophysics*, 54(3), 271-287.
https://doi.org/10.1080/08123985.2022.2135430
"""

from .clustering import cluster_statistics, kmeans, select_signal_clusters
from .features import build_features, gabor_kernel
from .filters import (amplitude_spectrum, fk_fan_filter, fk_mask_filter, fk_spectrum,
                      ormsby_filter)
from .garch import (GarchFit, fit_garch, fit_garch_traces, garch_conditional_variance,
                    select_garch_order)
from .io import Dataset, list_datasets, load_config, load_dataset
from .metrics import local_correlation, snr_db, snr_report, true_snr_db
from .pipeline import DEFAULT_CONFIG, PipelineResult, run_pipeline
from .preprocessing import agc
from .synthetic import make_synthetic_shot, ricker

__version__ = "1.0.0"

__all__ = [
    "agc", "amplitude_spectrum", "build_features", "cluster_statistics", "Dataset",
    "DEFAULT_CONFIG", "fit_garch", "fit_garch_traces", "fk_fan_filter", "fk_mask_filter",
    "fk_spectrum", "gabor_kernel", "GarchFit", "garch_conditional_variance", "kmeans",
    "list_datasets", "load_config", "load_dataset", "local_correlation",
    "make_synthetic_shot", "ormsby_filter", "PipelineResult", "ricker", "run_pipeline",
    "select_garch_order", "select_signal_clusters", "snr_db", "snr_report", "true_snr_db",
]
