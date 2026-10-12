"""Reproduce the figures and numbers of Aminian & Riahi (2023).

Usage
-----
    python examples/reproduce_paper.py                 # everything (~10 min, 1 core)
    python examples/reproduce_paper.py --jobs -1       # use all CPU cores
    python examples/reproduce_paper.py --fast          # skip the 9-model GARCH comparison
    python examples/reproduce_paper.py --only synthetic field_shot

Figures are written to ``docs/figures`` and the numbers to
``docs/results.json`` (both can be changed with ``--out``).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from garch_groundroll import (load_config, load_dataset, local_correlation, make_synthetic_shot,
                              ormsby_filter, run_pipeline, select_garch_order, true_snr_db)
from garch_groundroll import plotting as P

ROOT = Path(__file__).resolve().parents[1]


def _cfg(name, jobs):
    cfg = load_config(name)
    cfg.pop("dataset", None)
    cfg["n_jobs"] = jobs
    return cfg


def run_synthetic(out, jobs, results):
    """Paper Figure 2: the workflow on a synthetic gather (clean signal known)."""
    g = make_synthetic_shot()
    cfg = _cfg("synthetic", jobs)
    r = run_pipeline(g.data, g.dt, g.dx, cfg)
    P.save(P.plot_workflow(r, g.dt, g.dx), out / "fig01_synthetic_workflow.png")
    P.save(P.plot_sections([g.signal, r.filtered, r.corrected],
                           ["True reflections", "Ormsby low-cut", "Ormsby + GARCH recovery"],
                           g.dt, g.dx), out / "fig02_synthetic_vs_truth.png")
    entry = {"ormsby": {**r.snr,
                        "true_snr_filtered_db": true_snr_db(r.filtered, g.signal),
                        "true_snr_corrected_db": true_snr_db(r.corrected, g.signal),
                        "kept_clusters": r.kept_clusters.tolist()}}
    # same experiment with an f-k fan filter, which already preserves the reflections well
    fan = dict(cfg, filter={"type": "fk_fan", "v_reject": 1100, "v_pass": 1600,
                            "f_max": 20, "f_taper": 10})
    r2 = run_pipeline(g.data, g.dt, g.dx, fan)
    entry["fk_fan"] = {**r2.snr,
                       "true_snr_filtered_db": true_snr_db(r2.filtered, g.signal),
                       "true_snr_corrected_db": true_snr_db(r2.corrected, g.signal),
                       "kept_clusters": r2.kept_clusters.tolist()}
    results["synthetic"] = entry
    print(r.summary())


def run_shot502(out, jobs, results, fast):
    """Paper Figures 3-12: symmetric shot gather, f-k filter."""
    ds = load_dataset("shot_gather_502")
    cfg = _cfg("shot_gather_502", jobs)
    dt, dx = ds.dt, ds.dx
    r = run_pipeline(ds.data, dt, dx, cfg, fk_mask=ds.fk_mask)
    print(r.summary())

    P.save(P.plot_sections([ds.data, r.filtered, r.noise],
                           ["Shot gather", "f-k filtered", "Removed by f-k"], dt, dx),
           out / "fig03_shot_data_fk.png")
    entry = {**r.snr, "kept_clusters": r.kept_clusters.tolist(),
             "cluster_mean_csd": r.cluster_stats["mean_csd"].tolist()}
    if not fast:
        sel = select_garch_order(r.noise, criterion="bic", n_jobs=jobs)
        P.save(P.plot_loglik_grid(sel["orders"], sel["loglik"], sel["best"]),
               out / "fig04_shot_garch_orders.png")
        best_ll = np.argmax(sel["loglik"], axis=0)
        entry["garch_order_selection"] = {
            "orders": sel["orders"],
            "best_by_bic_counts": sel["counts"].tolist(),
            "best_by_loglik_counts": np.bincount(best_ll, minlength=len(sel["orders"])).tolist(),
            "best_overall_bic": sel["best_overall"],
        }
    P.save(P.plot_csd(r.variance, r.csd, dt, dx), out / "fig05_shot_csd.png")
    P.save(P.plot_clusters(r.noise, r.labels, r.mask, r.kept_clusters, r.cluster_stats, dt, dx),
           out / "fig06_shot_clusters.png")
    P.save(P.plot_sections([r.noise, r.corrected_noise, r.recovered],
                           ["Noise removed by f-k", "Noise after GARCH", "Restored signal"],
                           dt, dx), out / "fig07_shot_noise_vs_garch.png")
    P.save(P.plot_fk([ds.data, r.corrected, r.recovered],
                     ["Shot gather", "f-k + GARCH", "Restored signal"], dt, dx),
           out / "fig08_shot_fk_spectra.png")

    # Ormsby result for the comparisons of Figures 10-12
    orm = ormsby_filter(ds.data, dt, (0, 0, 15, 20), "bandstop")
    nn = 148
    P.save(P.plot_trace_comparison({"Input": ds.data[:, nn], "f-k filter": r.filtered[:, nn],
                                    "Ormsby filter": orm[:, nn],
                                    "f-k + GARCH recovery": r.corrected[:, nn]}, dt, nn),
           out / "fig09_shot_trace148.png")
    P.save(P.plot_spectra({"Input": ds.data, "f-k filter": r.filtered, "Ormsby filter": orm,
                           "f-k + GARCH recovery": r.corrected}, dt, fmax=100),
           out / "fig10_shot_spectra.png")
    lc = [local_correlation(x, ds.data, 9) for x in (orm, r.filtered, r.corrected)]
    P.save(P.plot_local_correlation(lc, ["Ormsby vs input", "f-k vs input", "GARCH vs input"],
                                    dt, dx), out / "fig11_shot_local_correlation.png")
    entry["mean_local_correlation"] = {"ormsby": float(lc[0].mean()), "fk": float(lc[1].mean()),
                                       "garch": float(lc[2].mean())}
    results["shot_gather_502"] = entry


def run_field(out, jobs, results):
    """Paper Figures 13-16: field shot gather."""
    ds = load_dataset("field_shot")
    cfg = _cfg("field_shot", jobs)
    r = run_pipeline(ds.data, ds.dt, ds.dx, cfg, fk_mask=ds.fk_mask)
    print(r.summary())
    P.save(P.plot_sections([ds.data, r.filtered, r.corrected, r.recovered],
                           ["Field shot gather", "f-k filtered", "f-k + GARCH recovery",
                            "Restored signal"], ds.dt, ds.dx, ncols=4),
           out / "fig12_field_shot.png")
    P.save(P.plot_clusters(r.noise, r.labels, r.mask, r.kept_clusters, r.cluster_stats,
                           ds.dt, ds.dx), out / "fig13_field_clusters.png")
    orm = ormsby_filter(ds.data, ds.dt, (0, 0, 15, 20), "bandstop")
    P.save(P.plot_spectra({"Input": ds.data, "f-k filter": r.filtered, "Ormsby filter": orm,
                           "f-k + GARCH recovery": r.corrected}, ds.dt, fmax=120),
           out / "fig14_field_spectra.png")
    P.save(P.plot_fk([ds.data, r.corrected, r.corrected_noise],
                     ["Field shot gather", "f-k + GARCH", "Final noise"], ds.dt, ds.dx),
           out / "fig15_field_fk.png")
    results["field_shot"] = {**r.snr, "kept_clusters": r.kept_clusters.tolist()}


def run_section(out, jobs, results):
    """Paper Figures 17-18: stacked section, Ormsby filter, GARCH(2,2)."""
    ds = load_dataset("stacked_section")
    cfg = _cfg("stacked_section", jobs)
    r = run_pipeline(ds.data, ds.dt, ds.dx, cfg)
    print(r.summary())
    P.save(P.plot_sections([ds.data, r.filtered, r.noise],
                           ["Stacked section", "Ormsby filtered", "Removed by Ormsby"],
                           ds.dt, ds.dx), out / "fig16_section_ormsby.png")
    P.save(P.plot_sections([r.recovered, r.corrected_noise, r.corrected],
                           ["Restored signal", "Noise after GARCH", "Ormsby + GARCH recovery"],
                           ds.dt, ds.dx), out / "fig17_section_garch.png")
    results["stacked_section"] = {**r.snr, "kept_clusters": r.kept_clusters.tolist()}


def run_crossline(out, jobs, results):
    """Paper Figures 19-20: cross-line of a 3-D survey, f-k fan filter."""
    ds = load_dataset("crossline_3d")
    cfg = _cfg("crossline_3d", jobs)
    r = run_pipeline(ds.data, ds.dt, ds.dx, cfg)
    print(r.summary())
    P.save(P.plot_sections([ds.data, r.filtered, r.corrected],
                           ["3-D cross-line", "f-k filtered", "f-k + GARCH recovery"],
                           ds.dt, ds.dx), out / "fig18_crossline.png")
    results["crossline_3d"] = {**r.snr, "kept_clusters": r.kept_clusters.tolist()}


STEPS = {"synthetic": run_synthetic, "shot_gather_502": run_shot502, "field_shot": run_field,
         "stacked_section": run_section, "crossline_3d": run_crossline}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "docs" / "figures"))
    ap.add_argument("--results", default=str(ROOT / "docs" / "results.json"))
    ap.add_argument("--jobs", type=int, default=1, help="parallel processes (-1 = all cores)")
    ap.add_argument("--fast", action="store_true", help="skip the 9-model GARCH comparison")
    ap.add_argument("--only", nargs="*", choices=list(STEPS), help="run a subset")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    for name in args.only or STEPS:
        t = time.time()
        print(f"\n=== {name} ===")
        if name == "shot_gather_502":
            run_shot502(out, args.jobs, results, args.fast)
        else:
            STEPS[name](out, args.jobs, results)
        print(f"({time.time() - t:.0f} s)")

    res_path = Path(args.results)
    old = json.loads(res_path.read_text()) if res_path.exists() else {}
    for k, v in results.items():
        old.setdefault(k, {}).update(v)
    res_path.write_text(json.dumps(old, indent=2, default=lambda o: o.tolist()
                                   if hasattr(o, "tolist") else str(o)))
    print(f"\nFigures: {out}\nNumbers: {res_path}")


if __name__ == "__main__":
    main()
