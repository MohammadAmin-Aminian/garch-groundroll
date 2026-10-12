"""Five-minute tour of the method on the synthetic gather.

    python examples/quickstart.py
"""

from garch_groundroll import load_config, make_synthetic_shot, run_pipeline, true_snr_db
from garch_groundroll.plotting import plot_workflow, save

# 1. A split-spread shot gather: 5 reflections + dispersive ground roll.
g = make_synthetic_shot()

# 2. Settings from configs/synthetic.json:
#    Ormsby low-cut (removes 0-15 Hz, tapered to 25 Hz), GARCH(1,1) on the
#    removed noise, Gabor + k-means clustering of the conditional standard
#    deviation (5 clusters) and restoration of the low-CSD clusters.
config = load_config("synthetic")
print("filter:", config["filter"])
result = run_pipeline(g.data, g.dt, g.dx, config, verbose=True)

print(result.summary())
print(f"True SNR after Ormsby          : {true_snr_db(result.filtered, g.signal):6.2f} dB")
print(f"True SNR after GARCH recovery  : {true_snr_db(result.corrected, g.signal):6.2f} dB")

save(plot_workflow(result, g.dt, g.dx, "Synthetic shot gather"), "quickstart_workflow.png")
print("Figure written to quickstart_workflow.png")
