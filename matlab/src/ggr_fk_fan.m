function out = ggr_fk_fan(d, dt, dx, vReject, vPass, fMax, fTaper, pad)
%GGR_FK_FAN Velocity (fan) filter in the f-k domain.
%
%   out = GGR_FK_FAN(d, dt, dx, vReject, vPass, fMax, fTaper) removes energy
%   with apparent velocity |f/k| < vReject (m/s), keeps |f/k| > vPass and
%   tapers in between with a raised cosine. If FMAX is given (non-empty),
%   rejection is limited to |f| < fMax, tapered over FTAPER Hz (default 5).
%   VPASS defaults to 1.3*vReject. PAD is the zero-padding factor (default 2).

if nargin < 5 || isempty(vPass), vPass = 1.3 * vReject; end
if nargin < 6, fMax = []; end
if nargin < 7 || isempty(fTaper), fTaper = 5; end
if nargin < 8 || isempty(pad), pad = 2; end
[nt, nx] = size(d);
ntp = 2 ^ nextpow2(pad * nt);
nxp = 2 ^ nextpow2(pad * nx);
f = ggr_fftfreq(ntp, dt);
k = ggr_fftfreq(nxp, dx);
[K, F] = meshgrid(k, f);                 % F: ntp x nxp
vapp = abs(F) ./ abs(K);
vapp(K == 0) = Inf;
keep = cosTaper(vapp, vReject, vPass);
if ~isempty(fMax)
    keep = 1 - (1 - keep) .* (1 - cosTaper(abs(F), fMax, fMax + fTaper));
end
D = fft2(double(d), ntp, nxp);
out = real(ifft2(D .* keep));
out = out(1:nt, 1:nx);
end

function w = cosTaper(x, x0, x1)
if x1 <= x0
    w = double(x >= x0);
    return
end
t = min(max((x - x0) / (x1 - x0), 0), 1);
w = 0.5 - 0.5 * cos(pi * t);
end
