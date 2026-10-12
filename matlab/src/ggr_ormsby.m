function out = ggr_ormsby(d, dt, corners, mode)
%GGR_ORMSBY Zero-phase Ormsby (trapezoidal) frequency filter, trace by trace.
%
%   out = GGR_ORMSBY(d, dt, [f1 f2 f3 f4], mode) filters the columns of D
%   (time x traces) sampled at DT seconds. MODE is 'bandpass' (keep the
%   trapezoid) or 'bandstop' (default here; remove it). For ground roll use a
%   band-stop starting at 0 Hz, e.g. [0 0 12 18] removes 0-12 Hz with a
%   12-18 Hz taper.

if nargin < 4 || isempty(mode), mode = 'bandstop'; end
f1 = corners(1); f2 = corners(2); f3 = corners(3); f4 = corners(4);
if ~(0 <= f1 && f1 <= f2 && f2 <= f3 && f3 <= f4)
    error('ggr:ormsby', 'Ormsby corners must satisfy 0 <= f1 <= f2 <= f3 <= f4.');
end
nt = size(d, 1);
nfft = 2 ^ nextpow2(2 * nt);
f = abs(ggr_fftfreq(nfft, dt));
w = zeros(nfft, 1);
w(f >= f2 & f <= f3) = 1;
if f2 > f1
    m = f > f1 & f < f2;
    w(m) = (f(m) - f1) / (f2 - f1);
end
if f4 > f3
    m = f > f3 & f < f4;
    w(m) = (f4 - f(m)) / (f4 - f3);
end
switch lower(mode)
    case 'bandpass'
    case 'bandstop', w = 1 - w;
    otherwise, error('ggr:ormsby', 'mode must be bandpass or bandstop');
end
D = fft(double(d), nfft, 1);
out = real(ifft(D .* w, [], 1));   % implicit expansion (R2016b+)
out = out(1:nt, :);
end
