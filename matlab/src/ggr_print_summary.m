function ggr_print_summary(res)
%GGR_PRINT_SUMMARY Print clusters and SNR of a pipeline result.
fprintf('GARCH(%d,%d) on %d traces x %d samples\n', res.cfg.garch.p, res.cfg.garch.q, ...
    size(res.data, 2), size(res.data, 1));
for i = 1:numel(res.stats.ids)
    fprintf('  cluster %d: mean CSD %.3g, %.1f%% of pixels\n', res.stats.ids(i), ...
        res.stats.meanCsd(i), 100 * res.stats.fraction(i));
end
fprintf('  restored as signal: %s\n', mat2str(res.kept'));
fprintf('  SNR filtered  : %+.3f dB\n', res.snr.snrBeforeDb);
fprintf('  SNR corrected : %+.3f dB\n', res.snr.snrAfterDb);
fprintf('  SNR gain      : %+.3f dB (paper-style ratio %.3f)\n', res.snr.gainDb, res.snr.ratio);
end
