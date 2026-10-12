%QUICKSTART Five-minute tour of the method on the synthetic gather.
%   >> cd matlab/examples; quickstart
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup.m'));

g = ggr_make_synthetic();                       % reflections + ground roll

% Settings from configs/synthetic.json (shared with Python): Ormsby low-cut
% 0-15 Hz tapered to 25 Hz, GARCH(1,1), Gabor + k-means with 5 clusters,
% automatic selection of the low-CSD clusters.
cfg = ggr_load_config('synthetic');

res = ggr_run_pipeline(g.data, g.dt, g.dx, cfg);
ggr_print_summary(res);
fprintf('True SNR after Ormsby         : %6.2f dB\n', ggr_true_snr_db(res.filtered, g.signal));
fprintf('True SNR after GARCH recovery : %6.2f dB\n', ggr_true_snr_db(res.corrected, g.signal));

ggr_plot_workflow(res, g.dt, g.dx, 'Synthetic shot gather');
