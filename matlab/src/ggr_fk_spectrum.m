function [S, k, f] = ggr_fk_spectrum(d, dt, dx, pad)
%GGR_FK_SPECTRUM Amplitude f-k spectrum (non-negative frequencies).
%
%   [S, k, f] = GGR_FK_SPECTRUM(d, dt, dx) returns S (nf x nk), centred
%   wavenumbers k (cycles/m) and frequencies f (Hz).
if nargin < 4 || isempty(pad), pad = 1; end
[nt, nx] = size(d);
ntp = 2 ^ nextpow2(pad * nt);
nxp = 2 ^ nextpow2(pad * nx);
A = abs(fftshift(fft2(double(d), ntp, nxp), 2));
fa = ggr_fftfreq(ntp, dt);
k = fftshift(ggr_fftfreq(nxp, dx));
pos = find(fa >= 0);
[f, o] = sort(fa(pos));
S = A(pos(o), :);
end
