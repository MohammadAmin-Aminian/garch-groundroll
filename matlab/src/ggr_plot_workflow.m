function fig = ggr_plot_workflow(res, dt, dx, ttl)
%GGR_PLOT_WORKFLOW Six-panel overview: data, filtered, CSD, clusters,
%   corrected data and recovered signal.
if nargin < 4, ttl = ''; end
[blues, series] = ggr_colormaps();
[nt, nx] = size(res.data);
xr = [0 nx * dx];
tr = [0 nt * dt * 1000];
a = sort(abs(res.data(:)));
v = a(round(0.99 * numel(a)));
fig = figure('Color', 'w', 'Position', [100 100 1150 700]);
panels = {res.data, res.filtered, res.csd, res.labels, res.corrected, res.recovered};
names = {'(a) Input data', '(b) Conventional filter', '(c) CSD of removed noise', ...
    sprintf('(d) Clusters (kept: %s)', mat2str(res.kept')), ...
    '(e) After GARCH recovery', '(f) Recovered signal'};
for i = 1:6
    ax = subplot(2, 3, i);
    switch i
        case 3
            c = sort(res.csd(:));
            imagesc(ax, xr, tr, res.csd, [0 c(round(0.995 * numel(c)))]);
            colormap(ax, blues);
            colorbar(ax);
        case 4
            k = max(res.labels(:));
            imagesc(ax, xr, tr, res.labels, [0.5 k + 0.5]);
            colormap(ax, series(1:k, :));
            colorbar(ax, 'Ticks', 1:k);
        otherwise
            imagesc(ax, xr, tr, panels{i}, [-v v]);
            colormap(ax, gray);
    end
    title(ax, names{i});
    xlabel(ax, 'Distance (m)');
    ylabel(ax, 'Time (ms)');
end
if ~isempty(ttl) && exist('sgtitle', 'file'), sgtitle(fig, ttl); end
end
