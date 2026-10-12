function s = ggr_true_snr_db(estimate, truth)
%GGR_TRUE_SNR_DB Reconstruction SNR against a known clean signal.
s = ggr_snr_db(truth, estimate - truth);
end
