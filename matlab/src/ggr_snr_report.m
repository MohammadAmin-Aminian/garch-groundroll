function r = ggr_snr_report(filtered, noise, corrected, correctedNoise)
%GGR_SNR_REPORT SNR before/after recovery.
%   gainDb = after - before is the meaningful change; ratio = after/before
%   is the figure quoted in the paper (e.g. 1.08 -> "8 %"), valid only when
%   both values are positive.
r.snrBeforeDb = ggr_snr_db(filtered, noise);
r.snrAfterDb = ggr_snr_db(corrected, correctedNoise);
r.gainDb = r.snrAfterDb - r.snrBeforeDb;
r.ratio = r.snrAfterDb / r.snrBeforeDb;
end
