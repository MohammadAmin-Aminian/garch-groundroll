function cfg = ggr_default_config()
%GGR_DEFAULT_CONFIG Default processing parameters (same as the Python package).
cfg.filter = struct('type', 'fk_fan', 'v_reject', 1000, 'v_pass', [], 'f_max', [], ...
    'f_taper', 5);
cfg.garch = struct('p', 1, 'q', 1);
cfg.features = struct('wavelengths', [3 6 12 24 48 96], 'orientations', [0 45 90 135], ...
    'smoothing', 1.5, 'scaling', 'clip', 'coordinates', true);
cfg.clustering = struct('n_clusters', 5, 'n_init', 10, 'seed', 0, 'selection', 'gap', ...
    'n_keep', [], 'threshold', [], 'clusters', []);
end
