function fig = ggr_plot_sections(panels, titles, dt, dx, clipPct)
%GGR_PLOT_SECTIONS Side-by-side seismic sections with a common grey clip.
%   fig = GGR_PLOT_SECTIONS({d1, d2, ...}, {'title1', ...}, dt, dx)
if nargin < 5 || isempty(clipPct), clipPct = 99; end
n = numel(panels);
allv = [];
for i = 1:n, allv = [allv; abs(panels{i}(:))]; end %#ok<AGROW>
v = sortedPct(allv, clipPct);
fig = figure('Color', 'w', 'Position', [100 100 360 * n 340]);
for i = 1:n
    ax = subplot(1, n, i);
    [nt, nx] = size(panels{i});
    imagesc(ax, [0 nx * dx], [0 nt * dt * 1000], panels{i}, [-v v]);
    colormap(ax, gray);
    colorbar(ax);
    title(ax, sprintf('(%c) %s', 'a' + i - 1, titles{i}));
    xlabel(ax, 'Distance (m)');
    ylabel(ax, 'Time (ms)');
end
end

function v = sortedPct(x, p)
x = sort(x);
v = x(max(1, round(p / 100 * numel(x))));
if v == 0, v = 1; end
end
