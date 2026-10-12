"""Command-line interface.

Examples
--------
Run a bundled dataset with its paper configuration::

    garch-groundroll run --config shot_gather_502 --out results/shot502

Run your own gather (a .mat file with ``data``/``dt``/``dx`` or a .npy
array) with a custom configuration::

    garch-groundroll run --input my_gather.npy --dt 0.002 --dx 10 \\
        --config my_config.json --out results/my_gather

Generate and process the analytic synthetic::

    garch-groundroll run --config synthetic --out results/synthetic
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio

from . import __version__
from .io import list_datasets, load_config, load_dataset
from .metrics import true_snr_db
from .pipeline import run_pipeline
from .synthetic import make_synthetic_shot


def _load_input(args, cfg):
    truth = None
    if args.input:
        p = Path(args.input)
        if p.suffix == ".npy":
            if args.dt is None or args.dx is None:
                sys.exit("--dt and --dx are required for .npy input")
            return np.load(p), args.dt, args.dx, None, None, p.stem
        ds = load_dataset(p)
    else:
        name = cfg.get("dataset")
        if name is None:
            sys.exit("Give --input or use a configuration with a 'dataset' entry.")
        if name == "synthetic":
            g = make_synthetic_shot()
            return g.data, g.dt, g.dx, None, g.signal, "synthetic"
        ds = load_dataset(name)
        truth = ds.extras.get("signal")
    dt = args.dt if args.dt is not None else ds.dt
    dx = args.dx if args.dx is not None else ds.dx
    return ds.data, dt, dx, ds.fk_mask, truth, ds.name


def cmd_run(args):
    cfg = load_config(args.config) if args.config else {}
    cfg["n_jobs"] = args.jobs
    data, dt, dx, fk_mask, truth, name = _load_input(args, cfg)
    out = Path(args.out or f"results/{name}")
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    res = run_pipeline(data, dt, dx, cfg, fk_mask=fk_mask, verbose=not args.quiet)
    summary = {
        "dataset": name, "shape": list(data.shape), "dt": dt, "dx": dx,
        "garch_order": [res.config["garch"]["p"], res.config["garch"]["q"]],
        "kept_clusters": [int(k) for k in res.kept_clusters],
        "cluster_mean_csd": [float(v) for v in res.cluster_stats["mean_csd"]],
        "cluster_fraction": [float(v) for v in res.cluster_stats["fraction"]],
        **{k: float(v) for k, v in res.snr.items()},
        "runtime_s": round(time.time() - t0, 1),
    }
    if truth is not None:
        summary["true_snr_filtered_db"] = true_snr_db(res.filtered, truth)
        summary["true_snr_corrected_db"] = true_snr_db(res.corrected, truth)

    sio.savemat(out / "result.mat", {
        "data": res.data, "filtered": res.filtered, "noise": res.noise,
        "variance": res.variance, "csd": res.csd, "labels": res.labels.astype(np.uint8),
        "mask": res.mask.astype(np.uint8), "recovered": res.recovered,
        "corrected": res.corrected, "corrected_noise": res.corrected_noise,
        "loglik": res.loglik, "dt": dt, "dx": dx,
    }, do_compression=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "config_used.json").write_text(json.dumps(res.config, indent=2))

    if not args.no_figures:
        from . import plotting as P
        P.save(P.plot_workflow(res, dt, dx, name), out / "workflow.png")
        P.save(P.plot_csd(res.variance, res.csd, dt, dx), out / "csd.png")
        P.save(P.plot_clusters(res.noise, res.labels, res.mask, res.kept_clusters,
                               res.cluster_stats, dt, dx), out / "clusters.png")
        P.save(P.plot_spectra({"Input": res.data, "Filtered": res.filtered,
                               "GARCH corrected": res.corrected}, dt,
                              fmax=min(100.0, 0.5 / dt)), out / "spectra.png")
    print(res.summary())
    print(f"Results written to {out}/")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="garch-groundroll",
                                 description="Recover signal lost during ground-roll attenuation "
                                             "by GARCH conditional-standard-deviation clustering.")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="process one gather")
    r.add_argument("--config", help="configs/<name>.json or a path to a JSON file")
    r.add_argument("--input", help=".mat (data, dt, dx[, fk_mask]) or .npy file")
    r.add_argument("--dt", type=float, help="sampling interval in s (overrides file)")
    r.add_argument("--dx", type=float, help="trace spacing in m (overrides file)")
    r.add_argument("--out", help="output folder (default results/<dataset>)")
    r.add_argument("--jobs", type=int, default=1, help="parallel processes for GARCH (-1 = all)")
    r.add_argument("--no-figures", action="store_true")
    r.add_argument("--quiet", action="store_true")
    r.set_defaults(func=cmd_run)

    ls = sub.add_parser("datasets", help="list bundled datasets")
    ls.set_defaults(func=lambda a: print("\n".join(["synthetic (generated)"] + list_datasets())))

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
