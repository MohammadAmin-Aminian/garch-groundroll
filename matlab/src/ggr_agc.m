function out = ggr_agc(d, window, epsRel)
%GGR_AGC Automatic gain control with a sliding RMS window (samples).
%   out = GGR_AGC(d, window) divides each sample by the RMS of a centred
%   window of WINDOW samples along time (default 100).
if nargin < 2 || isempty(window), window = 100; end
if nargin < 3 || isempty(epsRel), epsRel = 1e-10; end
d = double(d);
r = sqrt(ggr_box_mean(d .^ 2, [window 1]));
floorVal = epsRel * (sqrt(mean(d(:) .^ 2)) + realmin);
out = d ./ max(r, floorVal);
end
