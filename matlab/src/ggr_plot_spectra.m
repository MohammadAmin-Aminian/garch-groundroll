function fig = ggr_plot_spectra(curves, names, dt, fmax)
%GGR_PLOT_SPECTRA Overlay of mean amplitude spectra, scaled by the first curve's maximum.
%   fig = GGR_PLOT_SPECTRA({d1, d2, ...}, {'name1', ...}, dt, fmax)
if nargin < 4 || isempty(fmax), fmax = 100; end
[~, series] = ggr_colormaps();
fig = figure('Color', 'w', 'Position', [100 100 640 360]);
hold on
for i = 1:numel(curves)
    [f, A] = ggr_amplitude_spectrum(curves{i}, dt);
    if i == 1, ref = max(A); end
    plot(f, A / ref, 'LineWidth', 1.6, 'Color', series(i, :));
end
hold off
xlim([0 fmax]);
grid on
xlabel('Frequency (Hz)');
ylabel('Amplitude (relative to input)');
legend(names, 'Box', 'off');
end
