function w = ggr_ricker(f0, dt, len)
%GGR_RICKER Zero-phase Ricker wavelet, 2*round(len/(2dt))+1 samples (column).
if nargin < 3 || isempty(len), len = 2.5 / f0; end
half = round(len / (2 * dt));
t = (-half:half)' * dt;
a = (pi * f0 * t) .^ 2;
w = (1 - 2 * a) .* exp(-a);
end
