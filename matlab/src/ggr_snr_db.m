function s = ggr_snr_db(signal, noise)
%GGR_SNR_DB 10*log10(sum(signal.^2)/sum(noise.^2)), as snr(x,y).
s = 10 * log10(sum(signal(:) .^ 2) / sum(noise(:) .^ 2));
end
