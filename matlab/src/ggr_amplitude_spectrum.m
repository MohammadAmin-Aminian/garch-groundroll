function [f, A] = ggr_amplitude_spectrum(d, dt, reduce)
%GGR_AMPLITUDE_SPECTRUM Mean (or max) amplitude spectrum over traces.
if nargin < 3 || isempty(reduce), reduce = 'mean'; end
if isvector(d), d = d(:); end
nt = size(d, 1);
X = abs(fft(double(d), [], 1));
nf = floor(nt / 2) + 1;
X = X(1:nf, :);
if strcmpi(reduce, 'max')
    A = max(X, [], 2);
else
    A = mean(X, 2);
end
f = (0:nf-1)' / (nt * dt);
end
