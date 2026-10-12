%REPRODUCE_PAPER Run every experiment of Aminian & Riahi (2023) in MATLAB.
%   >> cd matlab/examples; reproduce_paper
%   Figures are saved to <repo>/results/matlab and a summary table is
%   printed. Uses the same configs/*.json files as the Python package.
run(fullfile(fileparts(mfilename('fullpath')), '..', 'setup.m'));
outDir = fullfile(ggr_root(), 'results', 'matlab');
if ~exist(outDir, 'dir'), mkdir(outDir); end

names = {'synthetic', 'shot_gather_502', 'field_shot', 'stacked_section', 'crossline_3d'};
summary = cell(numel(names), 4);
for i = 1:numel(names)
    name = names{i};
    fprintf('\n=== %s ===\n', name);
    cfg = ggr_load_config(name);
    if strcmp(name, 'synthetic')
        g = ggr_make_synthetic();
        data = g.data; dt = g.dt; dx = g.dx; fkMask = [];
    else
        ds = ggr_load_dataset(name);
        data = ds.data; dt = ds.dt; dx = ds.dx; fkMask = ds.fkMask;
    end
    tic;
    res = ggr_run_pipeline(data, dt, dx, cfg, 'FkMask', fkMask);
    ggr_print_summary(res);
    fprintf('(%.0f s)\n', toc);
    if strcmp(name, 'synthetic')
        fprintf('True SNR: filtered %.2f dB -> corrected %.2f dB\n', ...
            ggr_true_snr_db(res.filtered, g.signal), ggr_true_snr_db(res.corrected, g.signal));
    end

    f1 = ggr_plot_workflow(res, dt, dx, strrep(name, '_', ' '));
    print(f1, fullfile(outDir, [name '_workflow.png']), '-dpng', '-r150');
    f2 = ggr_plot_spectra({data, res.filtered, res.corrected}, ...
        {'Input', 'Filtered', 'GARCH corrected'}, dt, min(100, 0.5 / dt));
    print(f2, fullfile(outDir, [name '_spectra.png']), '-dpng', '-r150');
    close([f1 f2]);
    summary(i, :) = {name, res.snr.snrBeforeDb, res.snr.snrAfterDb, res.snr.gainDb};
end

fprintf('\n%-18s %12s %12s %10s\n', 'dataset', 'SNR filt.', 'SNR corr.', 'gain');
for i = 1:size(summary, 1)
    fprintf('%-18s %+11.3f %+11.3f %+9.3f dB\n', summary{i, :});
end
fprintf('\nFigures saved to %s\n', outDir);
